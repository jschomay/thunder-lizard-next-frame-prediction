"""Entity-centric targets (H13): instead of "is there a dino in this cell next tick?", ask each
existing entity "what happens to you?" and move it ourselves. Entities can't duplicate or vanish
by accident this way.

Fates (one per entity in the last frame):
  0 stay, 1 up, 2 right, 3 down, 4 left   moved one cell in the world (the view may also scroll)
  5 die                                   became a skeleton
  6 gone                                  removed (eaten, or a skeleton cleared)
  -1                                      unknown: it left the view, so we can't tell (no loss)

The view follows the player, so screen positions also shift by the scroll. The scroll is found
from the terrain: the shift that best lines up the two frames.
"""

from __future__ import annotations

import numpy as np

STAY, DIE, GONE, UNKNOWN = 0, 5, 6, -1
N_FATES = 7
MOVES = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]  # (dy, dx) for stay, up, right, down, left
N = 45


def kind(e: np.ndarray) -> np.ndarray:
    """0 empty, 1 player, 2 dino, 3 skeleton"""
    return np.where(e == 0, 0, np.where(e == 41, 3, np.where(e <= 20, 1, 2)))


def find_scroll(t0: np.ndarray, t1: np.ndarray) -> tuple[int, int]:
    """(dy, dx) such that t1[y, x] ~= t0[y + dy, x + dx]: the view moved by (dy, dx)"""
    best, best_err = (0, 0), None
    for dy, dx in MOVES:
        a = t0[max(dy, 0):N + min(dy, 0), max(dx, 0):N + min(dx, 0)]
        b = t1[max(-dy, 0):N + min(-dy, 0), max(-dx, 0):N + min(-dx, 0)]
        err = (a != b).mean()
        if best_err is None or err < best_err - 1e-9:
            best, best_err = (dy, dx), err
    return best


def fates(f0: np.ndarray, f1: np.ndarray) -> np.ndarray:
    """f0, f1: [2, 45, 45] frames -> int8 [45, 45] fate per entity cell of f0 (-1 elsewhere)"""
    out = np.full((N, N), UNKNOWN, np.int8)
    sy, sx = find_scroll(f0[0], f1[0])
    k0, k1 = kind(f0[1]), kind(f1[1])
    taken = np.zeros((N, N), bool)
    ys, xs = np.nonzero(k0)
    # entities that stay put claim their cell first, so a neighbour can't steal it
    order = sorted(zip(ys, xs), key=lambda p: 0 if _at(k1, p[0] - sy, p[1] - sx) == k0[p] else 1)
    for y, x in order:
        k = k0[y, x]
        any_inside = False
        for m, (dy, dx) in enumerate(MOVES):
            ty, tx = y + dy - sy, x + dx - sx
            if not (0 <= ty < N and 0 <= tx < N):
                continue
            any_inside = True
            if not taken[ty, tx] and k1[ty, tx] == k:
                out[y, x] = m; taken[ty, tx] = True
                break
        else:
            sty, stx = y - sy, x - sx
            if k in (1, 2) and _at(k1, sty, stx) == 3 and not taken[sty, stx]:
                out[y, x] = DIE; taken[sty, stx] = True
            elif any_inside and all(0 <= y + dy - sy < N and 0 <= x + dx - sx < N for dy, dx in MOVES):
                out[y, x] = GONE   # every place it could have gone is in view, and it isn't there
            # else: it may have stepped out of view -> unknown
    return out


def _at(a, y, x):
    return a[y, x] if 0 <= y < N and 0 <= x < N else -1


def episode_fates(obs: np.ndarray) -> np.ndarray:
    """obs [T+1, 2, 45, 45] -> int8 [T, 45, 45]"""
    return np.stack([fates(obs[t], obs[t + 1]) for t in range(len(obs) - 1)])
