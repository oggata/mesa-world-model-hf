"""街の地図。colab/world_model_mesa.ipynb の「地図」のセルと同じ作り方 (同じ seed なら同じ街になる)。

ノートブックを変えたら、ここも同じように直すこと。
"""
import random
from collections import deque

import numpy as np

OTHER, ROAD, BUILDING, TREE = 0, 1, 2, 3

# mesa-clouds-renderer の BLDG_TYPES (server.js) と同じ。(名前 = textures/v4/<名前>.jpg, 敷地, 高さ [マス])
BLDG_TYPES = [
    ("kiosk", 1, 0.7), ("conbini", 1, 0.9), ("pharmacy", 1, 0.9), ("cafe", 1, 1.1), ("gyudon", 1, 1.1),
    ("ramen", 1, 1.1), ("bento", 1, 1.1), ("shop", 1, 1.4), ("house", 1, 1.4), ("post", 1, 1.4),
    ("bank", 1, 1.7), ("apartment", 1, 2.1), ("hotel", 1, 2.1), ("office", 1, 2.6), ("tower", 1, 3.3),
    ("police", 1, 1.7), ("elementary", 1, 1.1), ("junior", 1, 1.4), ("warehouse", 1, 1.4),
    ("supermarket", 2, 1.1), ("temple", 2, 1.1), ("school", 2, 1.4), ("station", 2, 1.4), ("library", 2, 1.4),
    ("hospital", 2, 1.7), ("cityhall", 2, 1.7), ("museum", 2, 1.7), ("stadium", 2, 2.1), ("mall", 2, 2.6),
    ("high", 2, 1.7), ("university", 2, 2.1),
]
SMALL = [i for i, t in enumerate(BLDG_TYPES) if t[1] == 1]
BIG = [i for i, t in enumerate(BLDG_TYPES) if t[1] == 2]

G_ASPHALT, G_GRASS, G_DIRT = 0, 1, 2
SLOT_TWOLANE, SLOT_ONEWAY, SLOT_PATH = 0, 16, 32


def make_map_organic(size, seed):
    rng = random.Random(seed)
    grid = np.full((size, size), OTHER, dtype=np.int32)
    rows = cols = list(range(0, size, 4))
    for r in rows: grid[r, :] = ROAD
    for c in cols: grid[:, c] = ROAD
    for ri in range(len(rows) - 1):
        for ci in range(len(cols) - 1):
            cells = [(r, c) for r in range(rows[ri] + 1, rows[ri + 1]) for c in range(cols[ci] + 1, cols[ci + 1])]
            if not cells: continue
            b = rng.choice(cells); grid[b] = BUILDING
            for rc in cells:
                if rc == b: continue
                v = rng.random()
                grid[rc] = TREE if v < 0.25 else BUILDING if v < 0.45 else grid[rc]
    for r in rows: grid[r, :] = ROAD
    for c in cols: grid[:, c] = ROAD

    def connected(g):
        cells = list(zip(*np.where(g == ROAD)))
        seen = {cells[0]}; q = deque([cells[0]])
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                n = (r + dr, c + dc)
                if 0 <= n[0] < size and 0 <= n[1] < size and n not in seen and g[n] == ROAD:
                    seen.add(n); q.append(n)
        return len(seen) == len(cells)

    cand = [(r, c) for r, c in zip(*np.where(grid == ROAD)) if not (r in rows and c in cols)]
    rng.shuffle(cand)
    quota = int(len(cand) * (0.25 + rng.random() * 0.25)); removed = 0
    for rc in cand:
        if removed >= quota: break
        grid[rc] = OTHER
        if connected(grid):
            grid[rc] = TREE if rng.random() < 0.4 else OTHER; removed += 1
        else:
            grid[rc] = ROAD
    return grid


def bldg_height(t, r, c):
    k = ((r * 73856093) ^ (c * 19349663)) % 5
    return BLDG_TYPES[t][2] * (0.88 + 0.06 * k)


def place_buildings(grid, seed, p_big=0.3):
    rng = np.random.default_rng(seed)
    grid = grid.copy(); S = grid.shape[0]
    bt = np.full(grid.shape, -1, np.int64); bh = np.zeros(grid.shape, np.float32)
    org = np.zeros(grid.shape + (2,), np.int64); fp = np.ones(grid.shape, np.int64)
    for r0 in range(1, S - 3, 4):
        for c0 in range(1, S - 3, 4):
            if rng.random() >= p_big: continue
            r, c = r0 + int(rng.integers(0, 2)), c0 + int(rng.integers(0, 2))
            t = int(rng.choice(BIG)); sl = (slice(r, r + 2), slice(c, c + 2))
            grid[sl] = BUILDING; bt[sl] = t; org[sl] = (r, c); fp[sl] = 2; bh[sl] = bldg_height(t, r, c)
    for r, c in zip(*np.where((grid == BUILDING) & (bt < 0))):
        t = int(rng.choice(SMALL)); bt[r, c] = t; org[r, c] = (r, c); bh[r, c] = bldg_height(t, r, c)
    return grid, bt, bh, org, fp


def ground_layout(grid):
    S = grid.shape[0]
    gk = np.full(grid.shape, G_GRASS, np.int64); slot = np.full(grid.shape, -1, np.int64)
    for r in range(S):
        for c in range(S):
            t = grid[r, c]
            if t in (ROAD, BUILDING): gk[r, c] = G_ASPHALT
            elif t == OTHER and ((r * 92837111) ^ (c * 689287499)) % 10 < 3: gk[r, c] = G_DIRT
            if t != ROAD: continue
            R = lambda rr, cc: 0 <= rr < S and 0 <= cc < S and grid[rr, cc] == ROAD
            mask = R(r - 1, c) | R(r, c + 1) << 1 | R(r + 1, c) << 2 | R(r, c - 1) << 3
            slot[r, c] = SLOT_PATH if bin(mask).count("1") <= 1 else SLOT_ONEWAY + mask
    return gk, slot


def build_city(seed, size=30, p_big=0.3):
    """seed → 1 つの街の配列一式 (numpy)"""
    grid, bt, bh, org, fp = place_buildings(make_map_organic(size, seed), seed, p_big)
    gk, slot = ground_layout(grid)
    return dict(grid=grid, bt=bt, bh=bh, org=org, fp=fp, gk=gk, slot=slot, tree=(grid == TREE))
