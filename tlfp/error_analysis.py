"""Slice a checkpoint's validation errors into categories until a pattern shows up.

    uv run python -m tlfp.error_analysis checkpoints/<run>.pt

1. Per tick: did the real frame scroll? did the prediction scroll? -> 4 boxes, share of wrong cells in each
2. Invented scrolls: did the player move on the previous tick? (rhythm vs something else)
3. Invented scrolls with no previous move: what terrain is in the key's direction? (blocked/slowed?)
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np
import torch

from tlfp.data import TERRAIN, find_episodes, split_by_episode
from tlfp.model import CNN
from tlfp.train import Frames

DIRS = {1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1)}
SCROLL = 20  # more than this many terrain cells differing = the view scrolled


def main(path: str):
    ck = torch.load(path, map_location="cpu")
    c = ck["config"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CNN(c["k"], c["width"], c["depth"], copy_prior=c.get("copy_prior", False)).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    _, val = split_by_episode(find_episodes(*c["data"]), 0.1, seed=0)
    data = Frames(val, c["k"], device)

    boxes, wrong = Counter(), Counter()
    cause, toward = Counter(), Counter()
    for i in range(0, len(data), 512):
        ctx, keys, tgt = data.batch(torch.arange(i, min(i + 512, len(data)), device=device))
        pred = model.predict(ctx, keys)
        for j in range(len(tgt)):
            prev = ctx[j, -1]
            real = bool((prev[0] != tgt[j, 0]).any())
            guess = int((prev[0] != pred[j, 0]).sum()) > SCROLL
            box = ("real move" if real else "no move") + (", model scrolled" if guess else ", model stayed")
            boxes[box] += 1
            wrong[box] += int((pred[j] != tgt[j]).sum())
            if guess and not real:  # invented scroll: why?
                prev_moved = c["k"] > 1 and bool((ctx[j, -2, 0] != prev[0]).any())
                if prev_moved:
                    cause["rhythm (moved last tick)"] += 1
                    continue
                cause["tried to move, didn't"] += 1
                ent = prev[1].cpu().numpy()
                ys, xs = np.nonzero((ent >= 1) & (ent <= 20))
                if len(ys) and int(keys[j]) in DIRS:
                    dy, dx = DIRS[int(keys[j])]
                    y, x = ys[0] + dy, xs[0] + dx
                    toward["off view" if not (0 <= y < 45 and 0 <= x < 45) else TERRAIN[int(prev[0, y, x])]] += 1

    total = sum(wrong.values())
    print("1. ticks by (reality, prediction):")
    for b in sorted(boxes, key=lambda b: -wrong[b]):
        print(f"   {b:30s} ticks {boxes[b]:6d}   share of wrong cells {wrong[b] / total:6.1%}")
    print("2. invented scrolls, by cause:", dict(cause))
    print("3. ...'tried to move, didn't': terrain in the key's direction:", dict(toward.most_common()))


if __name__ == "__main__":
    main(sys.argv[1])
