"""ほかの住民 (NPC)。colab/world_model_mesa.ipynb の「住民の動き」のセルと同じ動き方と、世界モデルに渡す数の並び。

住民の位置はこのシミュレーターが決める「本当の状態」。世界モデルには、プレイヤーから見た住民の数の並び (テンソル) を渡す。
ノートブックを変えたら、ここも同じように直すこと。
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from .world import ROAD

DIRS4 = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])   # 向きの番号 0..3 = 向き 0, 90, 180, 270 度


class People:
    """1 つの街で K 人を歩かせる。道のマスの真ん中を、マスからマスへまっすぐ歩き、着くたびに行き先を選び直す"""
    def __init__(self, grid, k, speed=0.08, seed=None):
        self.grid, self.k, self.speed = grid, k, speed
        self.rng = np.random.default_rng(seed)
        cand = np.argwhere(grid == ROAD)
        pick = cand[self.rng.integers(len(cand), size=k)]
        self.r, self.c = pick[:, 0].copy(), pick[:, 1].copy()
        self.d = np.array([self._choose(r, c, None) for r, c in zip(self.r, self.c)])
        self.p = self.rng.random(k)

    def _options(self, r, c):
        G = self.grid.shape[0]
        return np.array([0 <= r + dr < G and 0 <= c + dc < G and self.grid[r + dr, c + dc] == ROAD for dr, dc in DIRS4], float)

    def _choose(self, r, c, d):
        w = self._options(r, c)
        if d is not None:
            w = w * (1 + 3 * np.eye(4)[d]) * (1 - 0.97 * np.eye(4)[(d + 2) % 4])   # まっすぐ 4 倍・引き返し 0.03 倍
        if w.sum() == 0:
            w = np.ones(4)
        return int(self.rng.choice(4, p=w / w.sum()))

    def step(self):
        self.p = self.p + self.speed
        for i in np.nonzero(self.p >= 1)[0]:
            self.r[i] += DIRS4[self.d[i], 0]; self.c[i] += DIRS4[self.d[i], 1]; self.p[i] -= 1
            self.d[i] = self._choose(self.r[i], self.c[i], self.d[i])

    def pose(self):
        """(K,3) の x, y, 向き"""
        x = self.r + 0.5 + self.p * DIRS4[self.d, 0]
        y = self.c + 0.5 + self.p * DIRS4[self.d, 1]
        return np.stack([x, y, self.d * math.pi / 2], -1).astype(np.float32)


def people_feats(x, y, th, pose, rng=8.0):
    """プレイヤー (1 人) から見た住民の数の並び (K,6): 前への距離・右への距離 (÷rng)・向きの sin, cos・距離 (÷rng)・いるか"""
    pose = torch.as_tensor(pose, dtype=torch.float32)
    c, s = math.cos(th), math.sin(th)
    rx, ry = pose[:, 0] - x, pose[:, 1] - y
    fwd, right = rx * c + ry * s, -rx * s + ry * c
    dist = torch.sqrt(rx ** 2 + ry ** 2)
    rel = pose[:, 2] - th
    here = (dist < rng).float()
    f = torch.stack([fwd / rng, right / rng, torch.sin(rel), torch.cos(rel), dist / rng, here], -1)
    return f * here[:, None]


def make_person_sprite(shirt, n=64):
    """人の絵 (RGBA)。頭・シャツ・ズボン (2 本の脚)"""
    yy, xx = (np.mgrid[0:n, 0:n // 2] + 0.5) / np.array([n, n // 2])[:, None, None]
    head = ((xx - 0.5) / 0.24) ** 2 + ((yy - 0.11) / 0.10) ** 2 < 1
    body = (abs(xx - 0.5) < 0.30) & (yy > 0.20) & (yy < 0.60)
    legs = (yy >= 0.60) & ((abs(xx - 0.36) < 0.11) | (abs(xx - 0.64) < 0.11))
    col = np.array([(shirt >> 16) & 255, (shirt >> 8) & 255, shirt & 255]) / 255
    rgb = np.where(head[..., None], np.array([0.93, 0.76, 0.62]),
                   np.where(body[..., None], col * (0.85 + 0.25 * (0.6 - yy))[..., None], np.array([0.20, 0.22, 0.30])))
    return np.dstack([rgb.clip(0, 1), (head | body | legs).astype(np.float32)]).astype(np.float32)


def draw_people(img, zbuf, xs, ys, ths, people, FOC, hor, eye, tex, fog, person_h=0.26, person_w=0.13):
    """img: (N,H,W,3) / zbuf: (N,H,W) その画素に見えている物までの距離 / people: (N,K,3) 住民の x, y, 向き。
    遠い人から順に描き、壁や木より奥にいる部分は描かない (zbuf と比べる)"""
    N, H, W, _ = img.shape
    K = people.shape[1]
    dirX, dirY = torch.cos(ths)[:, None], torch.sin(ths)[:, None]
    rx, ry = people[..., 0] - xs[:, None], people[..., 1] - ys[:, None]
    dz = rx * dirX + ry * dirY                                  # 前方への距離
    s = -rx * dirY + ry * dirX                                  # 右へのずれ
    order = torch.argsort(dz, dim=1, descending=True)           # 遠い順
    cols = torch.arange(W, device=img.device, dtype=torch.float32) + 0.5
    rows = torch.arange(H, device=img.device, dtype=torch.float32) + 0.5
    T = tex.shape
    for j in range(K):
        k = order[:, j]
        d = dz.gather(1, k[:, None])[:, 0]; sk = s.gather(1, k[:, None])[:, 0]
        dd = d.clamp(min=1e-3)
        cx = W / 2 + FOC * sk / dd
        hw = FOC * person_w / 2 / dd
        top = hor - FOC * (person_h - eye) / dd; bot = hor + FOC * eye / dd
        u = (cols[None] - (cx - hw)[:, None]) / (2 * hw)[:, None]                 # (N,W)
        v = (rows[None] - top[:, None]) / (bot - top)[:, None]                    # (N,H)
        inside = ((d > 0.15)[:, None, None] & (u >= 0)[:, None, :] & (u < 1)[:, None, :]
                  & (v >= 0)[:, :, None] & (v < 1)[:, :, None])
        ui = (u.clamp(0, 0.999) * T[2]).long()[:, None, :].expand(N, H, W)
        vi = (v.clamp(0, 0.999) * T[1]).long()[:, :, None].expand(N, H, W)
        tc = tex[(k % T[0])[:, None, None].expand(N, H, W), vi, ui]               # (N,H,W,4)
        on = inside & (tc[..., 3] > 0.5) & (d[:, None, None] < zbuf)
        img = torch.where(on[..., None], fog(tc[..., :3], d[:, None, None].expand(N, H, W)), img)
        zbuf = torch.where(on, d[:, None, None].expand(N, H, W), zbuf)
    return img
