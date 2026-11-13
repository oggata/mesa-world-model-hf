"""世界モデル・復元器・逆動力学と DINOv2。colab/world_model_mesa.ipynb の「共通の部品」と同じ形。

重みの名前 (state_dict のキー) がノートブックと一致していないと読めないので、クラスの中身は変えないこと。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    """順動力学: 「過去 T フレームの特徴 + 各フレームでの行動」→「それぞれの次のフレームの特徴」。

    1 フレーム = N 個のパッチトークン。行動の埋め込みを全トークンに足し、フレーム単位の因果マスクで
    「フレーム i はフレーム 0..i だけを見る」ようにする (DINO-WM と同じ形)。
    出力は「前のフレームからの差分」。最初は差分 0 = 「何も変わらない」と予測する状態から学習が始まる。

    n_npc > 0 のときは、ほかの住民 (NPC) の数の並び (B,T,n_npc,npc_f) も受け取る。1 人 = 1 トークンにして、
    各フレームの画像のトークンの後ろに並べる (位置 t には「次のフレーム t+1 で住民がどこにいるか」を入れる。
    住民の動きはシミュレーターが決めるので、世界モデルは「そこにいるときの見た目」を描くだけでよい)。
    n_npc = 0 なら前の版とまったく同じ形 (重みの名前も同じ)。
    """
    def __init__(self, feat_dim, n_tok, hist, act_dim=None, n_actions=None, d=256, depth=6, heads=8, n_npc=0, npc_f=6):
        super().__init__()
        self.hist, self.n_tok, self.n_npc = hist, n_tok, n_npc
        self.inp = nn.Linear(feat_dim, d)
        if n_actions is not None:
            self.act = nn.Embedding(n_actions, d)
        else:
            self.act = nn.Sequential(nn.Linear(act_dim, d), nn.GELU(), nn.Linear(d, d))
        self.pos_tok = nn.Parameter(torch.randn(1, 1, n_tok, d) * 0.02)
        self.pos_t = nn.Parameter(torch.randn(1, hist, 1, d) * 0.02)
        if n_npc:
            self.npc = nn.Sequential(nn.Linear(npc_f, d), nn.GELU(), nn.Linear(d, d))
            self.npc_id = nn.Parameter(torch.randn(1, 1, n_npc, d) * 0.02)      # 何番目の住民か (同じ人をフレームをまたいで追う)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, feat_dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        f = torch.arange(hist).repeat_interleave(n_tok + n_npc)
        self.register_buffer("mask", f[None, :] > f[:, None], persistent=False)  # True = 見てはいけない

    def forward(self, z, a, n=None):
        # z: (B,T,N,D) / a: (B,T) 行動番号 か (B,T,A) 連続値 / n: (B,T,K,F) 住民 (無ければ「誰もいない」)。位置 t の出力が t+1 の予測
        B, T, N, D = z.shape
        h = self.inp(z) + self.pos_tok + self.act(a)[:, :, None, :]
        if self.n_npc:
            if n is None:
                n = torch.zeros(B, T, self.n_npc, self.npc[0].in_features, device=z.device)
            h = torch.cat([h, self.npc(n.to(h.dtype)) + self.npc_id + self.act(a)[:, :, None, :]], 2)
        h = h + self.pos_t[:, -T:]
        L = h.shape[2]
        h = self.blocks(h.reshape(B, T * L, -1), mask=self.mask[:T * L, :T * L])
        return z + self.out(self.norm(h)).reshape(B, T, L, -1)[:, :, :N]


@torch.no_grad()
def imagine(model, z_ctx, a_seq, n_seq=None):
    """想像の中で先へ進める (シミュレータとしての使い方)。
    z_ctx: (B,C,N,D) 実際に見た直近 C フレーム / a_seq: (B, C-1+K, ...) 文脈の先頭フレームからの行動
    n_seq: (B, C-1+K, n_npc, F) 各行動の「あと」のフレームでの住民 (a_seq と同じ並び。無ければ住民なし)
    返り値: (B,K,N,D) 想像した K フレーム先までの特徴。予測を次の入力に使い回すので、先ほど誤差が積もる"""
    H = model.hist
    zs = list(z_ctx.float().unbind(1)); C = len(zs)
    K = a_seq.shape[1] - (C - 1)
    for _ in range(K):
        t = len(zs) - 1; lo = max(0, t - H + 1)
        pred = model(torch.stack(zs[lo:], 1), a_seq[:, lo:t + 1],
                     None if n_seq is None else n_seq[:, lo:t + 1])[:, -1]
        zs.append(pred.float())
    return torch.stack(zs[C:], 1)


class ResBlock(nn.Module):
    """残差ブロック: 入力に「少しの直し」を足す。深くしても学習が崩れにくい"""
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.GroupNorm(8, c), nn.GELU(), nn.Conv2d(c, c, 3, 1, 1),
                               nn.GroupNorm(8, c), nn.GELU(), nn.Conv2d(c, c, 3, 1, 1))

    def forward(self, x):
        return x + self.b(x)


class Decoder(nn.Module):
    """特徴 (grid×grid トークン) → 画像 (3, out, out)。想像した未来を目で見るためだけに使う。
    世界モデルの学習や計画には一切使わない (画素ではなく特徴で考えるのが DINO-WM の要点)。
    out は grid の 2 の累乗倍 (8→64, 16→128 など)。2 倍に広げる層を、その回数だけ重ねる。

    res: 広げる段ごとに重ねる残差ブロックの数。ctx: 直前に見た本物の画像 (手がかり) も受け取る。
    res=0, ctx=False のときは前の版とまったく同じ形 (重みの名前も同じ) なので、前に保存した重みも読める。"""
    def __init__(self, feat_dim, grid, ch=128, out=None, res=0, ctx=False):
        super().__init__()
        self.grid = grid
        self.out_px = out = out or 8 * grid
        self.n_up = n_up = round(math.log2(out / grid))
        assert n_up >= 2 and grid * 2 ** n_up == out, f"out ({out}) は grid ({grid}) の 4 倍以上の 2 の累乗倍にする"
        def up(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.GroupNorm(8, o), nn.GELU(),
                                 nn.Conv2d(o, o, 3, 1, 1), nn.GroupNorm(8, o), nn.GELU())
        chs = [2 * ch] * (n_up - 1) + [ch, ch // 2]       # 広げるたびに細くする (8→64 なら 2ch, 2ch, ch, ch/2)
        self.ctx = ctx
        if not res and not ctx:                            # 前の版と同じ形
            self.net = nn.Sequential(nn.Conv2d(feat_dim, chs[0], 1), *[up(chs[k], chs[k + 1]) for k in range(n_up)],
                                     nn.Conv2d(ch // 2, 3, 3, 1, 1))
            return
        ce = ch // 2 if ctx else 0                          # 手がかりの画像から取り出す特徴の幅
        if ctx:
            # 手がかりの画像を、out → out/2 → … → grid と縮めながら特徴にする (各段の大きさで、復元の途中に足す)
            self.cenc = nn.ModuleList([nn.Sequential(nn.Conv2d(3, ce, 3, 1, 1), nn.GELU())] +
                                      [nn.Sequential(nn.Conv2d(ce, ce, 4, 2, 1), nn.GroupNorm(8, ce), nn.GELU())
                                       for _ in range(n_up)])
        self.stem = nn.Conv2d(feat_dim, chs[0], 1)
        self.ups = nn.ModuleList([up(chs[k] + ce, chs[k + 1]) for k in range(n_up)])
        self.blocks = nn.ModuleList([nn.Sequential(*[ResBlock(chs[k + 1]) for _ in range(res)]) for k in range(n_up)])
        self.head = nn.Sequential(nn.Conv2d(ch // 2 + ce, ch // 2, 3, 1, 1), nn.GELU(), nn.Conv2d(ch // 2, 3, 3, 1, 1))

    def forward(self, z, ctx_img=None):
        """z: (B,N,D) / ctx_img: (B,3,out,out) [0,1] の手がかりの画像 (ctx=True のとき。None なら手がかりなし)"""
        B, N, D = z.shape
        h = z.transpose(1, 2).reshape(B, D, self.grid, self.grid)
        if not hasattr(self, "stem"):
            return torch.sigmoid(self.net(h))
        h = self.stem(h)
        if self.ctx:
            if ctx_img is None:
                ctx_img = torch.zeros(B, 3, self.out_px, self.out_px, device=z.device)
            if ctx_img.shape[-1] != self.out_px:
                ctx_img = F.interpolate(ctx_img.float(), size=self.out_px, mode="bilinear", antialias=True)
            e = [self.cenc[0](ctx_img.to(h.dtype) if h.dtype != torch.float32 else ctx_img.float())]
            for m in self.cenc[1:]:
                e.append(m(e[-1]))
            e = e[::-1]                                    # e[0] = grid の大きさ … e[-1] = out の大きさ
        for k in range(self.n_up):
            if self.ctx:
                h = torch.cat([h, e[k].to(h.dtype)], 1)
            h = self.blocks[k](self.ups[k](h))
        if self.ctx:
            h = torch.cat([h, e[-1].to(h.dtype)], 1)
        return torch.sigmoid(self.head(h))


class InverseDynamics(nn.Module):
    """逆動力学: (今のフレーム, 次のフレーム) の特徴 → その間の行動"""
    def __init__(self, feat_dim, n_tok, out_dim, r=32):
        super().__init__()
        self.red = nn.Linear(feat_dim, r)
        self.mlp = nn.Sequential(nn.Linear(2 * n_tok * r, 512), nn.GELU(),
                                 nn.Linear(512, 512), nn.GELU(), nn.Linear(512, out_dim))

    def forward(self, z0, z1):
        return self.mlp(torch.cat([self.red(z0).flatten(1), self.red(z1).flatten(1)], 1))


class DinoEncoder:
    """凍結した DINOv2 (vits14)。224×224 の画像 → grid×grid のパッチ特徴 (384 次元)"""
    def __init__(self, device="cpu"):
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, imgs, grid):
        p = self.model.forward_features((imgs - self.mean) / self.std)["x_norm_patchtokens"]
        B, D = p.shape[0], p.shape[-1]
        p = p.float().transpose(1, 2).reshape(B, D, 16, 16)
        if grid != 16:
            p = F.adaptive_avg_pool2d(p, grid)
        return p.flatten(2).transpose(1, 2)
