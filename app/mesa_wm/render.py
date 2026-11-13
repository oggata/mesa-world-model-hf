"""目線の描画。colab/world_model_mesa.ipynb の「目線の描画」のセルと同じレイキャスタ。

mesa-clouds-renderer の 3D 表示に寄せたもの (建物の写真と高さ・路面標示・芝生・木・昼の空)。
設定は学習時の config.json の "render" をそのまま使うので、学習したときと同じ見え方になる。
ノートブックの描画を変えたら、ここも同じように直すこと。
"""
import io
import math
import os
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .people import draw_people, make_person_sprite
from .world import BLDG_TYPES

CACHE = os.environ.get("MESA_WM_CACHE", os.path.expanduser("~/.cache/mesa_wm"))


def fetch(base, path, mode="RGB"):
    """テクスチャを取ってくる (一度取ったら手元に置いておく)"""
    local = os.path.join(CACHE, "textures", path)
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        data = urllib.request.urlopen(base + path).read()
        with open(local + ".part", "wb") as f:
            f.write(data)
        os.replace(local + ".part", local)
    return Image.open(local).convert(mode)


def make_tree_sprite(n=64, seed=0):
    g = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / n
    crown = ((xx - 0.5) / 0.48) ** 2 + ((yy - 0.36) / 0.36) ** 2 < 1
    trunk = (abs(xx - 0.5) < 0.06) & (yy > 0.55)
    blot = np.kron(g.standard_normal((n // 8, n // 8)), np.ones((8, 8)))
    shade = 0.75 + 0.35 * (0.5 - yy) + 0.10 * blot
    rgb = np.where(crown[..., None], np.array([0x48, 0x90, 0x3f]) / 255 * shade[..., None],
                   np.array([0x7a, 0x52, 0x30]) / 255)
    a = (crown | trunk).astype(np.float32)
    return np.dstack([rgb.clip(0, 1), a]).astype(np.float32)


class Renderer:
    def __init__(self, cfg, device="cpu"):
        rc = cfg["render"]
        self.device, self.grid = device, cfg["grid"]
        self.FOV, self.EYE, self.SSAA = math.radians(rc["fov_deg"]), rc["eye"], rc["ssaa"]
        self.LAYERS, self.TLAYERS = rc["layers"], rc["tlayers"]
        self.TW, self.TH, self.GT, self.AT = rc["tw"], rc["th"], rc["gt"], rc["at"]
        self.TREE_H, self.TREE_R = rc["tree_h"], rc["tree_r"]
        self.FOG0, self.FOG1 = rc["fog"]
        base = rc["textures"]
        dev = device
        t = lambda im, size: torch.tensor(np.asarray(im.resize(size, Image.BILINEAR), np.float32) / 255, device=dev)
        TW, TH, GT, AT = self.TW, self.TH, self.GT, self.AT
        self.TEX = torch.stack([t(fetch(base, f"textures/v4/{n}.jpg"), (TW, TH)) for n, _, _ in BLDG_TYPES])
        dirt = t(fetch(base, "textures/base/other_256.jpg"), (GT, GT)) * torch.tensor([0xb5, 0xb0, 0x8a], device=dev) / 255 * 1.55
        asph = t(fetch(base, "textures/base/asphalt_256.jpg"), (GT, GT))
        asph = asph / asph.mean((0, 1)) * torch.tensor([0.265, 0.275, 0.295], device=dev)
        self.GTEX = torch.stack([asph, t(fetch(base, "textures/base/grass_256.jpg"), (GT, GT)), dirt.clamp(0, 1)])
        atlas = fetch(base, "textures/road/road_atlas.png", "RGBA")
        self.ATLAS = torch.stack([t(atlas.crop(((s % 8) * 128 + 8, (s // 8) * 128 + 8, (s % 8) * 128 + 120, (s // 8) * 128 + 120)), (AT, AT))
                                  for s in range(33)])
        self.TREE_TEX = torch.tensor(make_tree_sprite(128), device=dev)
        rgb = lambda h: torch.tensor([(h >> 16) & 255, (h >> 8) & 255, h & 255], device=dev, dtype=torch.float32) / 255
        self.SKY_H = rgb(0xeaf2f7)
        self.SKY_Z = self.SKY_H * 0.78 + (rgb(0x1b3a63) - self.SKY_H * 0.78) * 0.10
        npc = cfg.get("npc") or {}
        self.PERSON_H, self.PERSON_W = npc.get("person_h", 0.26), npc.get("person_w", 0.13)
        colors = npc.get("colors", [0xe03c31, 0x2f6fde, 0xf2c230, 0x2fae5a, 0xb04fd6, 0xff8a1f, 0x18b5b5, 0xf06aa8])
        self.PERSON_TEX = torch.tensor(np.stack([make_person_sprite(c) for c in colors]), device=dev)

    def set_city(self, city):
        """world.build_city の結果を載せる (1 つの街だけ持つ。地図の番号 mid は常に 0)"""
        T = lambda a, **k: torch.tensor(a[None], device=self.device, **k)
        self.BT, self.BH, self.BORG, self.BFP = T(city["bt"]), T(city["bh"], dtype=torch.float32), T(city["org"]), T(city["fp"])
        self.GK, self.RSLOT, self.TREEM = T(city["gk"]), T(city["slot"]), T(city["tree"])

    def _fog(self, col, dist):
        f = ((dist - self.FOG0) / (self.FOG1 - self.FOG0)).clamp(0, 1) * 0.7
        return col * (1 - f[..., None]) + self.SKY_H * f[..., None]

    @torch.no_grad()
    def render(self, xs, ys, ths, H=224, W=224, chunk=16, people=None):
        """xs, ys, ths: (N,) → 目線の画像 (N,3,H,W) [0,1]。people: (N,K,3) ほかの住民の x, y, 向き"""
        mid = torch.zeros(xs.shape[0], dtype=torch.long, device=self.device)
        out = []
        for i in range(0, xs.shape[0], chunk):
            sl = slice(i, i + chunk)
            im = self._render(xs[sl], ys[sl], ths[sl], mid[sl], H * self.SSAA, W * self.SSAA,
                              people=None if people is None else people[sl]).permute(0, 3, 1, 2)
            out.append(F.avg_pool2d(im, self.SSAA) if self.SSAA > 1 else im)
        return torch.cat(out).clamp(0, 1).contiguous()

    @torch.no_grad()
    def _render(self, xs, ys, ths, mid, H, W, people=None, MAXSTEPS=64):
        """(N,) → (N,H,W,3)。列 0 が th - FOV/2 を見る = th を減らすと画面の左を向く (既存方策の turn_left と同じ)"""
        DEVICE, GRID = self.device, self.grid
        BT, BH, BORG, BFP, GK, RSLOT, TREEM = self.BT, self.BH, self.BORG, self.BFP, self.GK, self.RSLOT, self.TREEM
        TEX, GTEX, ATLAS, TREE_TEX, SKY_H, SKY_Z = self.TEX, self.GTEX, self.ATLAS, self.TREE_TEX, self.SKY_H, self.SKY_Z
        FOV, EYE, LAYERS, TLAYERS, TREE_H, TREE_R = self.FOV, self.EYE, self.LAYERS, self.TLAYERS, self.TREE_H, self.TREE_R
        TW, TH, GT, AT, _fog = self.TW, self.TH, self.GT, self.AT, self._fog
        N = xs.shape[0]
        cam = 2 * torch.arange(W, device=DEVICE, dtype=torch.float32) / W - 1
        dirX, dirY = torch.cos(ths), torch.sin(ths)
        pl = math.tan(FOV / 2); planeX, planeY = -dirY * pl, dirX * pl
        FOC = (W / 2) / pl                      # 焦点距離 [px]。縦も横と同じ倍率で写す (縦に伸びない)
        hor = H / 2
        yc = torch.arange(H, device=DEVICE, dtype=torch.float32) + 0.5   # 画素の中心の行
        rdx = dirX[:, None] + planeX[:, None] * cam[None]; rdy = dirY[:, None] + planeY[:, None] * cam[None]
        m3 = mid[:, None, None]

        # ── 空 ──
        t = ((hor - yc) / hor).clamp(0, 1) ** 0.65
        img = (SKY_H + (SKY_Z - SKY_H) * t[:, None])[None, :, None, :].expand(N, H, W, 3).clone()

        # ── 地面 (床キャスト): 地平線より下の行は、その行が指す地面の 1 点の色 ──
        fr = yc > hor
        dist = FOC * EYE / (yc[fr] - hor)                                    # (Hf,) その行の地面までの距離
        fx = xs[:, None, None] + rdx[:, None, :] * dist[None, :, None]       # (N,Hf,W)
        fy = ys[:, None, None] + rdy[:, None, :] * dist[None, :, None]
        r, c = torch.floor(fx).long(), torch.floor(fy).long()
        inb = (r >= 0) & (r < GRID) & (c >= 0) & (c < GRID)
        rr, cc = r.clamp(0, GRID - 1), c.clamp(0, GRID - 1)
        fu, fv = fy - torch.floor(fy), fx - torch.floor(fx)                  # u = 列 (東) 方向, v = 行 (南) 方向
        base = GTEX[GK[m3, rr, cc], (fv * GT).long().clamp(0, GT - 1), (fu * GT).long().clamp(0, GT - 1)]
        slot = RSLOT[m3, rr, cc]
        mk = ATLAS[slot.clamp(min=0), (fv * AT).long().clamp(0, AT - 1), (fu * AT).long().clamp(0, AT - 1)]
        a = mk[..., 3:] * (slot >= 0)[..., None]
        floor = _fog(base * (1 - a) + mk[..., :3] * a, dist[None, :, None].expand_as(fx))
        img[:, fr] = torch.where(inb[..., None], floor, SKY_H)               # 地図の外は何も無い (renderer と同じく背景色)

        # ── 壁: DDA でレイを進め、「手前の壁の上端より上に見える壁」を最大 LAYERS 枚まで拾う ──
        mapX = torch.floor(xs).long()[:, None].expand(N, W).clone()
        mapY = torch.floor(ys).long()[:, None].expand(N, W).clone()
        ddx = torch.where(rdx == 0, torch.full_like(rdx, 1e30), (1 / rdx).abs())
        ddy = torch.where(rdy == 0, torch.full_like(rdy, 1e30), (1 / rdy).abs())
        stepX = torch.where(rdx < 0, -1, 1); stepY = torch.where(rdy < 0, -1, 1)
        pX, pY = xs[:, None], ys[:, None]
        sdx = torch.where(rdx < 0, (pX - mapX) * ddx, (mapX + 1 - pX) * ddx)
        sdy = torch.where(rdy < 0, (pY - mapY) * ddy, (mapY + 1 - pY) * ddy)
        side = torch.zeros((N, W), dtype=torch.long, device=DEVICE)
        active = torch.ones((N, W), dtype=torch.bool, device=DEVICE)
        occ = torch.full((N, W), float(H), device=DEVICE)        # この行より下は手前の壁で埋まっている
        cnt = torch.zeros((N, W), dtype=torch.long, device=DEVICE)
        L = {k: torch.zeros((LAYERS, N, W), device=DEVICE) for k in ("perp", "top", "bot", "u", "occ", "side")}
        Lt = torch.full((LAYERS, N, W), -1, dtype=torch.long, device=DEVICE)
        tcnt = torch.zeros((N, W), dtype=torch.long, device=DEVICE)
        TLr = {k: torch.zeros((TLAYERS, N, W), device=DEVICE) for k in ("dz", "u", "h", "occ")}
        TLv = torch.zeros((TLAYERS, N, W), dtype=torch.bool, device=DEVICE)
        m = mid[:, None].expand(N, W)
        for _ in range(MAXSTEPS):
            xlt = sdx < sdy; sx = active & xlt; sy = active & ~xlt
            sdx = torch.where(sx, sdx + ddx, sdx); mapX = torch.where(sx, mapX + stepX, mapX); side = torch.where(sx, 0, side)
            sdy = torch.where(sy, sdy + ddy, sdy); mapY = torch.where(sy, mapY + stepY, mapY); side = torch.where(sy, 1, side)
            inb = (mapX >= 0) & (mapX < GRID) & (mapY >= 0) & (mapY < GRID)
            mx, my = mapX.clamp(0, GRID - 1), mapY.clamp(0, GRID - 1)
            t = BT[m, mx, my]
            perp = torch.where(side == 0, sdx - ddx, sdy - ddy).clamp(min=1e-4)
            top = hor - FOC * (BH[m, mx, my] - EYE) / perp
            vis = active & inb & (t >= 0) & (top < occ) & (cnt < LAYERS)
            # 壁の横方向の位置 u (0..1)。2x2 の建物は 2 マスにまたがって 1 枚の写真を貼る
            o, fpw = BORG[m, mx, my], BFP[m, mx, my]
            u = torch.where(side == 0, pY + perp * rdy - o[..., 1], pX + perp * rdx - o[..., 0]) / fpw
            flip = ((side == 0) & (rdx > 0)) | ((side == 1) & (rdy < 0)); u = torch.where(flip, 1 - u, u)
            # 木: マスの中心に立つ、こちらを向いた板。通り抜けられるので壁と違って奥を隠さない (絵の透明部分は透ける)
            dz = (mx + 0.5 - pX) * dirX[:, None] + (my + 0.5 - pY) * dirY[:, None]            # 前方への距離
            s_ = (pX + rdx * dz - mx - 0.5) * -dirY[:, None] + (pY + rdy * dz - my - 0.5) * dirX[:, None]  # 横のずれ
            th_ = TREE_H * (0.85 + 0.075 * (((mx * 73856093) ^ (my * 19349663)) % 5))
            tvis = (active & inb & TREEM[m, mx, my] & (dz > 0.3) & (s_.abs() < TREE_R) & (tcnt < TLAYERS)
                    & (hor - FOC * (th_ - EYE) / dz.clamp(min=1e-4) < occ))
            for k in range(TLAYERS):
                sel = tvis & (tcnt == k)
                TLv[k] |= sel
                for name, v in (("dz", dz), ("u", s_ / (2 * TREE_R) + 0.5), ("h", th_.float()), ("occ", occ)):
                    TLr[name][k] = torch.where(sel, v, TLr[name][k])
            tcnt = tcnt + tvis.long()
            for k in range(LAYERS):
                sel = vis & (cnt == k)
                Lt[k] = torch.where(sel, t, Lt[k])
                for name, v in (("perp", perp), ("top", top), ("bot", hor + FOC * EYE / perp), ("u", u),
                                ("occ", occ), ("side", side.float())):
                    L[name][k] = torch.where(sel, v, L[name][k])
            occ = torch.where(vis, top, occ); cnt = cnt + vis.long()
            active = active & inb & (occ > 0) & (cnt < LAYERS)
            if not active.any(): break

        rows = yc[None, :, None]
        zbuf = torch.full((N, H, W), float("inf"), device=DEVICE)          # その画素に見えている物までの距離 (住民の隠れに使う)
        for k in range(LAYERS):
            tk = Lt[k]
            top, bot = L["top"][k][:, None, :], L["bot"][k][:, None, :]
            inwall = (tk >= 0)[:, None, :] & (rows >= top) & (rows < bot) & (rows < L["occ"][k][:, None, :])
            ty = ((rows - top) / (bot - top)).clamp(0, 0.999) * TH          # 写真は壁の高さ全体に貼る
            tx = (L["u"][k].clamp(0, 0.999) * TW).long()
            col = TEX[tk.clamp(min=0)[:, None, :].expand(N, H, W), ty.long(), tx[:, None, :].expand(N, H, W)]
            shade = torch.where(L["side"][k] > 0, 0.82, 1.0)                   # 面の向きで明るさを変える (日なた / 日かげ)
            col = _fog(col * shade[:, None, :, None], L["perp"][k][:, None, :].expand(N, H, W))
            img = torch.where(inwall[..., None], col, img)
            zbuf = torch.where(inwall, L["perp"][k][:, None, :].expand(N, H, W), zbuf)
        for k in reversed(range(TLAYERS)):                                     # 木は奥から順に重ねる
            dz = TLr["dz"][k].clamp(min=1e-4)[:, None, :]
            top = hor - FOC * (TLr["h"][k][:, None, :] - EYE) / dz; bot = hor + FOC * EYE / dz
            ty = ((rows - top) / (bot - top)).clamp(0, 0.999) * TREE_TEX.shape[0]
            tx = (TLr["u"][k].clamp(0, 0.999) * TREE_TEX.shape[1]).long()[:, None, :].expand(N, H, W)
            tc = TREE_TEX[ty.long().expand(N, H, W), tx]
            on = TLv[k][:, None, :] & (rows >= top) & (rows < bot) & (rows < TLr["occ"][k][:, None, :]) & (tc[..., 3] > 0.5)
            img = torch.where(on[..., None], _fog(tc[..., :3], dz.expand(N, H, W)), img)
            zbuf = torch.where(on, dz.expand(N, H, W), zbuf)
        if people is not None:                                             # ほかの住民 (遠い順に、壁や木に隠れる)
            img = draw_people(img, zbuf, xs, ys, ths, people, FOC, hor, self.EYE, self.PERSON_TEX, _fog,
                              self.PERSON_H, self.PERSON_W)
        return img

