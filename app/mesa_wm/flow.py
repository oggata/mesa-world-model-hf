"""拡散 (フローマッチング) 版の世界モデル。colab/world_model_mesa_flow.ipynb の「共通の部品」と同じ形。

重みの名前 (state_dict のキー) がノートブックと一致していないと読めないので、クラスの中身は変えないこと。
本体は models.WorldModel と同じで、出口 (out) の代わりに、ノイズを取り除く FlowHead を持つ。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import WorldModel


class FlowHead(nn.Module):
    """ノイズを取り除く部分 (拡散モデルの「描く」ところ)。
    1 フレーム分の「ノイズの混ざった変化」(N,D) と、混ざり具合 tau (0 = 全部ノイズ … 1 = 答え) を受け取り、
    ノイズを取り除いた「答えの変化」を当てる (進む向きは、そこから計算する)。
    向きを直接当てさせると、ノイズの 384 次元をすべて打ち消す必要があり、幅 d が 384 より狭いと消し切れない。
    答えはデータの中の限られた形なので、狭い幅でも当てられる。フレームの中の N トークンは互いに見合うので、1 枚の絵として
    矛盾のない未来を選べる。何を描くかは、世界モデルの本体が作ったトークンごとの条件 c で決まる"""
    def __init__(self, feat_dim, n_tok, d, depth=2, heads=8):
        super().__init__()
        self.inp = nn.Linear(feat_dim, d)
        self.cond = nn.Linear(d, d)
        self.pos = nn.Parameter(torch.randn(1, n_tok, d) * 0.02)
        self.time = nn.Sequential(nn.Linear(64, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, feat_dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, tau, c):
        """x: (B,N,D) ノイズの混ざった変化 / tau: (B,) 混ざり具合 / c: (B,N,d) 条件 → (B,N,D) 答えの変化の予想"""
        f = torch.exp(-math.log(1000.0) * torch.arange(32, device=x.device, dtype=torch.float32) / 32)
        e = tau.float()[:, None] * 1000.0 * f[None]
        t = self.time(torch.cat([e.sin(), e.cos()], 1))
        h = self.inp(x) + self.cond(c) + self.pos + t[:, None, :]
        return self.out(self.norm(self.blocks(h)))


class FlowWorldModel(WorldModel):
    """拡散 (フローマッチング) 版の世界モデル。Cosmos などの動画の拡散モデルと同じ考え方を、DINOv2 の特徴の上で行う。

    本体 (Transformer) は WorldModel と同じで、過去のフレーム・行動・住民・長い記憶を読んで、トークンごとに
    「次のフレームを描くための条件」を作る。そこから先が違う:
      ・WorldModel … 条件から次のフレームを 1 回で当てる (迷うと、ありえる未来の平均 = ぼやけた予測になる)
      ・この版     … ノイズから始めて、FlowHead で少しずつノイズを取り除き、次のフレームを「描く」。
                     迷う所では、ありえる未来のどれか 1 つを選んで描く (最初のノイズを変えると、別の未来になる)
    描くのは「今のフレームからの変化」を delta_sd で割ったもの (ノイズと同じ大きさにそろえる)。
    学習: 答えの変化 x1 とノイズ eps を tau : 1 - tau で混ぜたもの xt から、FlowHead が x1 を予想する。
    進む向き v = (予想 - xt) / (1 - tau) が、本当の向き x1 - eps に合うように学ぶ (rectified flow。x を当てて v で比べる形)。
    """
    def __init__(self, feat_dim, n_tok, hist, flow_depth=2, flow_heads=None, **kw):
        super().__init__(feat_dim, n_tok, hist, **kw)
        d = self.inp.out_features
        del self.out                                        # 1 回で当てる出力は使わない
        self.head = FlowHead(feat_dim, n_tok, d, flow_depth, flow_heads or kw.get("heads", 8))
        self.register_buffer("delta_sd", torch.ones(1))     # 1 歩で特徴がどれくらい変わるか (学習データから決める)

    def forward(self, z, a, n=None, mem=None):
        """本体: 過去と行動 → 各位置の「次のフレームを描く条件」(B,T,N,d)。WorldModel.forward と同じ計算で、最後の出力だけ違う"""
        B, T, N, D = z.shape
        h = self.inp(z) + self.pos_tok + self.act(a)[:, :, None, :]
        if self.n_npc:
            if n is None:
                n = torch.zeros(B, T, self.n_npc, self.npc[0].in_features, device=z.device)
            h = torch.cat([h, self.npc(n.to(h.dtype)) + self.npc_id + self.act(a)[:, :, None, :]], 2)
        h = h + self.pos_t[:, -T:]
        L = h.shape[2]
        h = h.reshape(B, T * L, -1)
        Lm = self.n_mem_tok
        if self.n_mem:
            if mem is None:
                mem = (torch.zeros(B, self.n_mem, self.mem_grid ** 2, D, device=z.device),
                       torch.zeros_like(a[:, :1]).unsqueeze(1).expand(B, self.n_mem, self.mem_stride, *a.shape[2:]),
                       torch.zeros(B, self.n_mem, dtype=torch.bool, device=z.device))
            mz, ma, ok = mem
            hm = (self.inp(mz) + self.mem_pos_tok + self.mem_pos_t + self.act(ma).sum(2)[:, :, None, :]
                  + self.mem_ok(ok.long())[:, :, None, :])
            h = torch.cat([hm.reshape(B, Lm, -1).to(h.dtype), h], 1)
        h = self.blocks(h, mask=self.mask[:Lm + T * L, :Lm + T * L])
        return self.norm(h[:, Lm:]).reshape(B, T, L, -1)[:, :, :N]

    def loss(self, c, z_base, z_next):
        """c: (B,T,N,d) 条件 / z_base: (B,T,N,D) 変化の起点 (入力のフレーム) / z_next: 答えのフレーム → フローマッチングの損失"""
        x1 = ((z_next.float() - z_base.float()) / self.delta_sd).reshape(-1, *z_next.shape[-2:])
        c = c.reshape(-1, *c.shape[-2:])
        tau = torch.sigmoid(torch.randn(len(x1), device=x1.device))[:, None, None]   # 真ん中あたり (いちばん難しい所) を多めに練習する
        eps = torch.randn_like(x1)
        xt = (1 - tau) * eps + tau * x1
        v = (self.head(xt, tau[:, 0, 0], c).float() - xt) / (1 - tau).clamp_min(0.05)
        return F.mse_loss(v, x1 - eps)

    @torch.no_grad()
    def sample(self, c, steps=8, temp=1.0, gen=None):
        """c: (B,N,d) → 描いた変化 (B,N,D)。ノイズから steps 回に分けて取り除く (オイラー法。最後の 1 回で予想そのものに着く)。
        temp: 最初のノイズの大きさ (0 = いつも同じ「無難な」未来) / gen: 乱数 (torch.Generator。同じなら同じ未来)"""
        B, N = c.shape[:2]
        x = torch.randn(B, N, self.head.out.out_features, generator=gen).to(c.device) * temp
        for k in range(steps):
            tau = torch.full((B,), k / steps, device=c.device)
            x = x + (self.head(x, tau, c).float() - x) / (steps - k)          # = 向き (予想 - x) / (1 - tau) × 1 / steps
        return x

    @torch.no_grad()
    def next(self, z, a, n=None, mem=None, steps=8, temp=1.0, gen=None):
        """直近の窓 → 次のフレームの特徴 (B,N,D) を 1 つ描く"""
        c = self(z, a, n, mem)[:, -1]
        return z[:, -1].float() + self.delta_sd * self.sample(c, steps, temp, gen)


@torch.no_grad()
def imagine_flow(model, z_ctx, a_seq, n_seq=None, steps=8, temp=1.0, seed=None):
    """拡散版で、想像の中で先へ進める。models.imagine と同じ引数に、描き方の 3 つを足したもの。
    z_ctx: (B,C,N,D) 実際に見た直近 C フレーム (model.span フレームまで渡すと全部使う)
    a_seq: (B, C-1+K) 文脈の先頭フレームからの行動 / n_seq: (B, C-1+K, n_npc, F) 各行動の「あと」の住民
    steps: 1 歩を何回に分けてノイズを取り除くか / temp: 最初のノイズの大きさ (0 = いつも同じ「無難な」未来) /
    seed: 乱数 (同じなら同じ未来。None なら毎回違う未来)。返り値: (B,K,N,D)"""
    gen = torch.Generator().manual_seed(int(seed)) if seed is not None else None
    H, M = model.hist, model.n_mem
    zs = list(z_ctx.float().unbind(1)); C = len(zs)
    pz = [model.pool(z) for z in zs] if M else None          # 記憶に使う、粗くしたフレーム
    K = a_seq.shape[1] - (C - 1)
    for _ in range(K):
        t = len(zs) - 1; lo = max(0, t - H + 1)
        mem = model.memory(pz, a_seq, lo) if M else None
        zs.append(model.next(torch.stack(zs[lo:], 1), a_seq[:, lo:t + 1],
                             None if n_seq is None else n_seq[:, lo:t + 1], mem, steps=steps, temp=temp, gen=gen).float())
        if M:
            pz.append(model.pool(zs[-1]))
    return torch.stack(zs[C:], 1)
