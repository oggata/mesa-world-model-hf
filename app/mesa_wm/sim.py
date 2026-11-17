"""世界モデルを「歩けるシミュレーター」として動かす。

1 回の指示 (例: 前進を 12 歩) ごとに
  ・世界モデルが、いま見えている特徴から先の特徴を想像する (imagine) → 復元器で画像にする
  ・比べるために、本物の描画器でも同じ行動を実行して描く
の 2 本の動画を作る。
"""
import copy
import json
import math
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file

from .models import Decoder, DinoEncoder, InverseDynamics, WorldModel, imagine
from .people import People, people_feats
from .render import Renderer
from .world import BUILDING, ROAD, TREE, build_city

# 指示の言葉 → 行動の名前 (config.json の actions に合わせる)
ALIASES = {"前進": "前進", "前": "前進", "forward": "前進", "f": "前進", "w": "前進",
           "左": "左", "left": "左", "l": "左", "a": "左",
           "右": "右", "right": "右", "r": "右", "d": "右",
           "後退": "後退", "後ろ": "後退", "back": "後退", "backward": "後退", "b": "後退", "s": "後退"}


class Simulator:
    def __init__(self, model_dir, device="cpu"):
        self.device = device
        with open(os.path.join(model_dir, "config.json")) as f:
            self.cfg = cfg = json.load(f)
        self.actions = cfg["actions"]
        self.hist, self.grid_tok = cfg["hist"], cfg["tok_grid"]
        self.move, self.rot = cfg["move"], math.radians(cfg["rot_deg"])
        self.img, self.thumb, self.size = cfg["img"], cfg["thumb"], cfg["grid"]
        self.render_px = cfg.get("render_px", self.img)   # 描く大きさ (DINOv2 には 224 に縮めて渡す)
        n_tok, D = self.grid_tok ** 2, cfg["feat_dim"]
        ld = lambda name: load_file(os.path.join(model_dir, name), device=device)

        self.wm = WorldModel(D, n_tok, self.hist, n_actions=len(self.actions), **cfg["world_model"]).to(device).eval()
        self.wm.load_state_dict(ld("world_model.safetensors"))
        self.span = self.wm.span                      # 覚えておく本物のフレームの数 (直近 hist + 長い記憶の届く範囲)
        self.dec = Decoder(D, self.grid_tok, **cfg["decoder"]).to(device).eval()
        self.dec.load_state_dict(ld("decoder.safetensors"))
        self.inv = None
        if os.path.exists(os.path.join(model_dir, "inverse_dynamics.safetensors")):
            self.inv = InverseDynamics(D, n_tok, len(self.actions), **cfg["inverse_dynamics"]).to(device).eval()
            self.inv.load_state_dict(ld("inverse_dynamics.safetensors"))
        st = ld("feat_stats.safetensors")
        self.mu, self.sd = st["mu"], st["sd"]
        self.dino = DinoEncoder(device)
        self.renderer = Renderer(cfg, device)
        self.npc = cfg.get("npc")                     # ほかの住民 (学習のときに NPCS > 0 だったときだけ)
        self._cities, self._cur = {}, None

    def to(self, device):
        """モデル・描画器のテクスチャ・いまの街の地図を、まとめて device に移す。
        ZeroGPU の Space では、GPU は @spaces.GPU の関数の中でしか使えないので、その中で "cuda" に移し、終わったら "cpu" に戻す"""
        if device == self.device:
            return self

        def mv(o):
            if isinstance(o, (torch.Tensor, torch.nn.Module)):
                return o.to(device)
            if isinstance(o, dict):
                return {k: mv(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return type(o)(mv(v) for v in o)
            return o

        for obj in (self, self.renderer, self.dino):
            for k, v in list(vars(obj).items()):
                if k not in ("renderer", "dino", "cfg", "_cities"):   # 街の地図 (numpy) と設定は動かさない
                    setattr(obj, k, mv(v))
            obj.device = device
        return self

    # ── 街と本物の環境 ──
    def city(self, seed):
        if seed not in self._cities:
            self._cities[seed] = build_city(seed, self.size, self.cfg["render"]["p_big"])
        if self._cur != seed:
            self.renderer.set_city(self._cities[seed]); self._cur = seed
        return self._cities[seed]

    def step_pose(self, city, x, y, th, a):
        """ノートブックの step_env と同じ動き方 (1 人ぶん)。建物と地図の外へは進めない"""
        name = self.actions[a]
        th = th + {"左": -self.rot, "右": self.rot}.get(name, 0.0)
        s = -1.0 if name == "後退" else 1.0
        nx, ny = x + s * self.move * math.cos(th), y + s * self.move * math.sin(th)
        G = self.size
        ok = 0.05 < nx < G - 0.05 and 0.05 < ny < G - 0.05 and city["bt"][int(nx), int(ny)] < 0
        return (nx, ny, th) if ok else (x, y, th)

    @torch.no_grad()
    def draw(self, poses, people=None):
        p = torch.tensor(poses, dtype=torch.float32, device=self.device)
        pp = None if people is None else torch.tensor(np.stack(people), dtype=torch.float32, device=self.device)
        return self.renderer.render(p[:, 0], p[:, 1], p[:, 2], self.render_px, self.render_px, people=pp)

    def npc_feats(self, pose, people_pose):
        """プレイヤーの位置と向き + 住民の本当の位置 → 世界モデルに渡す数の並び (K,6)"""
        return people_feats(*pose, people_pose, self.npc["range"])

    @torch.no_grad()
    def encode(self, imgs):
        return (self.dino(imgs, self.grid_tok) - self.mu) / self.sd

    # ── 1 回の散歩 ──
    def reset(self, seed, start_seed=None):
        """seed の街の道の上に立つ。最初の HIST フレームは前を向いて歩いて集める (世界モデルの文脈)"""
        city = self.city(seed)
        rng = np.random.default_rng(start_seed)
        g = city["grid"]; cand = np.argwhere(g == ROAD)
        r, c = cand[rng.integers(len(cand))]
        dirs = [k for k, (dr, dc) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1)))
                if 0 <= r + dr < self.size and 0 <= c + dc < self.size and g[r + dr, c + dc] == ROAD] or [0]
        pose = (r + 0.5, c + 0.5, int(rng.choice(dirs)) * math.pi / 2)
        poses = [pose]
        ppl = People(g, self.npc["count"], self.npc["speed"], seed=start_seed) if self.npc else None
        pps = [ppl.pose()] if ppl else None
        for _ in range(self.hist - 1):
            poses.append(self.step_pose(city, *poses[-1], 0))
            if ppl:
                ppl.step(); pps.append(ppl.pose())
        imgs = self.draw(poses, pps)
        z = self.encode(imgs)
        n_hist = [self.npc_feats(p, q) for p, q in zip(poses[1:], pps[1:])] if ppl else []   # 各行動の「あと」の住民
        return dict(seed=seed, poses=poses, z_real=z.cpu(), z_dream=z.cpu(), people=ppl, n_hist=n_hist,
                    a_hist=[0] * (self.hist - 1), last_img=imgs[-1].cpu(), trail=[p[:2] for p in poses])

    @property
    def has_hint(self):
        """復元器が手がかりの画像を受け取る形か (sharp 以上で学習した重み)"""
        return bool(getattr(self.dec, "ctx", False))

    @torch.no_grad()
    def decode(self, z, last_img, hint=True):
        """特徴 (K,N,D) → 画像 (K,3,h,w)。hint=True なら、最後に本物を見たコマ last_img (3,H,W) を手がかりに渡す
        (復元器が壁の模様や色をそこから写す)。False なら手がかりなし (学習でも 15% は手がかりなしで練習している)"""
        if self.has_hint and hint:
            return self.dec(z, last_img[None].to(self.device).expand(len(z), -1, -1, -1))
        return self.dec(z)

    @torch.no_grad()
    def run(self, state, acts, sync=True, hint=True):
        """acts: 行動番号の列。返り値: (本物の画像 (K+1,H,W,3), 想像の画像 (K+1,h,w,3), 新しい state)。
        sync=True なら「いま本当に見えているもの」から想像を始める。False なら前回の想像の続きから (夢の中を歩き続ける)。
        hint=False なら、復元器に本物の手がかり画像を渡さない (世界モデルの予測した特徴だけで絵にする)"""
        city = self.city(state["seed"])
        H = self.span                                 # 覚えておくフレームの数 (記憶の分も)
        z_ctx = (state["z_real"] if sync else state["z_dream"]).to(self.device)
        a_seq = torch.tensor([state["a_hist"] + list(acts)], device=self.device)
        # 本物の環境で同じ行動を実行する。住民もシミュレーターの決まりで歩く (世界モデルには、その結果を数の並びで渡す)
        poses = list(state["poses"])
        ppl = copy.deepcopy(state.get("people"))
        pps, n_new = [], []
        for a in acts:
            poses.append(self.step_pose(city, *poses[-1], a))
            if ppl is not None:
                ppl.step(); pps.append(ppl.pose()); n_new.append(self.npc_feats(poses[-1], pps[-1]))
        n_all = state.get("n_hist", []) + n_new
        n_seq = torch.stack(n_all)[None].to(self.device) if ppl is not None else None
        z_imag = imagine(self.wm, z_ctx[None], a_seq, n_seq)[0]                   # (K,N,D)
        z_all = torch.cat([z_ctx[-1:], z_imag]).float()                          # 最初の 1 枚 = 想像の出発点
        imag = self.decode(z_all, state["last_img"], hint)
        new = poses[-len(acts):]
        real = self.draw(new, pps if ppl is not None else None)
        z_new = self.encode(real[-H:]).cpu()
        z_real = torch.cat([state["z_real"], z_new])[-H:]
        z_dream = torch.cat([z_ctx.cpu(), z_imag.cpu()])[-H:]
        real = torch.cat([state["last_img"][None].to(self.device), real])
        st = dict(state, poses=poses[-H:], z_real=z_real, z_dream=z_dream, people=ppl, n_hist=n_all[-(H - 1):] if ppl else [],
                  a_hist=(state["a_hist"] + list(acts))[-(H - 1):], last_img=real[-1].cpu(),
                  trail=(state["trail"] + [p[:2] for p in new])[-400:])
        to_np = lambda t: t.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()
        return to_np(real), to_np(imag), st

    # ── 見せるための絵 ──
    def minimap(self, state, px=8):
        """真上から見た街と、歩いた道のり (赤) と、いまの向き"""
        city = self.city(state["seed"]); g = city["grid"]
        col = np.zeros(g.shape + (3,), np.uint8)
        col[g == ROAD] = (70, 72, 78); col[g == TREE] = (60, 120, 55); col[(g != ROAD) & (g != TREE)] = (150, 180, 110)
        h = np.clip(city["bh"] / 3.7, 0, 1)
        bmask = g == BUILDING
        col[bmask] = (np.stack([200 - 120 * h, 150 - 60 * h, 120 + 100 * h], -1)[bmask]).astype(np.uint8)
        im = Image.fromarray(col).resize((g.shape[1] * px, g.shape[0] * px), Image.NEAREST)
        d = ImageDraw.Draw(im)
        pts = [(yy * px, xx * px) for xx, yy in state["trail"]]
        if len(pts) > 1: d.line(pts, fill=(230, 40, 40), width=2)
        x, y, th = state["poses"][-1]
        cx, cy = y * px, x * px
        tip = (cx + math.sin(th) * px * 1.6, cy + math.cos(th) * px * 1.6)
        if state.get("people") is not None:                     # ほかの住民 (シャツの色の点)
            for k, (px_, py_, _) in enumerate(state["people"].pose()):
                c = self.npc["colors"][k % len(self.npc["colors"])]
                ex, ey = py_ * px, px_ * px
                d.ellipse((ex - 3, ey - 3, ex + 3, ey + 3), fill=((c >> 16) & 255, (c >> 8) & 255, c & 255), outline=(0, 0, 0))
        d.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(255, 220, 0), outline=(0, 0, 0))
        d.line([(cx, cy), tip], fill=(255, 220, 0), width=3)
        return im


def parse_commands(text, actions, default_steps=12, max_total=120):
    """「前進 10, 左 5, 後退 3」→ 行動番号の列"""
    out = []
    for part in text.replace("、", ",").replace("，", ",").split(","):
        words = part.strip().split()
        if not words: continue
        name = ALIASES.get(words[0].lower())
        if name is None or name not in actions:
            raise ValueError(f"分からない指示です: {part.strip()} (前進 / 左 / 右 / 後退 と歩数で書いてください)")
        n = int(words[1]) if len(words) > 1 else default_steps
        out += [actions.index(name)] * max(0, n)
    return out[:max_total]


def _font(px):
    try:
        return ImageFont.load_default(size=px)        # Pillow 10.1 以降は大きさを選べる
    except TypeError:
        return ImageFont.load_default()


def _label(d, xy, text, font, anchor):
    """白い文字に黒いふち。古い Pillow のビットマップ文字 (anchor やふちが使えない) でも落ちないようにする"""
    try:
        d.text(xy, text, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0), anchor=anchor)
    except (ValueError, TypeError):
        x, y = xy
        d.text((x, y - 12 if anchor == "lb" else y), text, font=font, fill=(255, 255, 255))


def _wide(a, w, h, resample):
    """正方形の目線の画像 (H,W,3) [0,1] の上下を切って横長 (w:h) にし、w×h に広げる"""
    im = Image.fromarray((a * 255).astype(np.uint8))
    W0, H0 = im.size
    ch = min(H0, round(W0 * h / w))
    top = (H0 - ch) // 2                              # 地平線のある真ん中を残す
    return im.crop((0, top, W0, top + ch)).resize((w, h), resample)


def save_gif(real, imag, path, fps=6, width=768, aspect=16 / 9, show_real=False, inset=0.28, margin=12, loop=False):
    """想像を横長 (aspect) の画面いっぱいに描いた GIF を書く。
    show_real=True なら右上の小窓に本物を重ねる (答え合わせ用)。loop=False なら 1 回だけ再生して最後のコマで止まる"""
    height = round(width / aspect)
    sw = int(width * inset); sh = round(sw / aspect)  # 小窓の大きさ
    x0, y0 = width - sw - margin, margin              # 小窓の左上
    font, big = _font(max(12, sh // 6)), _font(max(12, height // 22))
    frames = []
    for r, m in zip(real, imag):
        f = _wide(m, width, height, Image.BICUBIC)
        if show_real:
            d = ImageDraw.Draw(f)
            d.rectangle((x0 - 3, y0 - 3, x0 + sw + 2, y0 + sh + 2), fill=(255, 255, 255), outline=(0, 0, 0))
            f.paste(_wide(r, sw, sh, Image.BILINEAR), (x0, y0))
            _label(d, (x0 + 4, y0 + sh - 4), "REAL", font, "lb")
            _label(d, (margin, margin), "IMAGINED", big, "lt")
        frames.append(f)
    extra = dict(loop=0) if loop else {}              # loop を書かなければ、ブラウザは 1 回だけ再生する
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), **extra)
    return path
