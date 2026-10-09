"""One fixed report for any fates model, so runs and architectures compare like for like.

    uv run python -m tlfp.scorecard checkpoints/<run>.pt [--name label]

Always scored on the stage-3 validation games (the split of data/stage3-100k + data/stage3-100k-b),
whatever the model was trained on. Prints a table and saves scorecards/<name>.json.

One step (the net sees real frames):
  terrain_changed     terrain accuracy on cells that really changed
  player_move_right   player moved: predicted the right move
  ocean_block         key pushes into ocean and the player is blocked: predicted "stay"
  dino_move_right     dino moved: predicted the right move
  dino_moves_pred     share of dinos predicted to move (real: ~40%)
  chase_near          predicted toward/away/stay for dinos within 6 cells (real ~27/13/60)
Rollouts (8 games x 150 ticks, the app's step: dice for dinos, edge spawns, lava dice):
  static              isolated terrain cells (real ~0.02%)
  max_share           biggest single terrain type, worst frame (real ~59%)
  dinos               dinos per frame (real ~1.5)
  scroll_when_key     ticks with a key held where the view scrolled (real ~44%)
Speed: CPU ms per frame.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tlfp.data import find_episodes, split_by_episode
from tlfp.model import build_model
from tlfp.moves import MOVES, find_scroll, kind
from tlfp.rollout import model_step, reset_memory
from tlfp.train import Frames

VAL_DATA = ["data/stage3-100k", "data/stage3-100k-b"]
DIRS = {1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1)}


@torch.no_grad()
def one_step(model, data, dev) -> dict:
    c = dict.fromkeys(["tc", "tn", "pm", "pn", "ob", "on", "dm", "dn", "dp", "dall"], 0)
    near = np.zeros((2, 3))
    yy, xx = np.mgrid[0:45, 0:45]
    for i in range(0, len(data), 256):
        ctx, keys, tgt, fate = data.batch(torch.arange(i, min(i + 256, len(data)), device=dev))
        t, _, f = model.forward_parts(ctx, keys)
        pt, pf = t.argmax(1), f.argmax(1)
        ch = ctx[:, -1, 0] != tgt[:, 0]
        c["tc"] += ((pt == tgt[:, 0]) & ch).sum().item(); c["tn"] += ch.sum().item()
        last, fate, pf = ctx[:, -1].cpu().numpy(), fate.cpu().numpy(), pf.cpu().numpy()
        keys = keys.cpu().numpy()
        for b in range(len(last)):
            k = kind(last[b][1])
            pl = np.argwhere(k == 1)
            if len(pl):
                py, px = pl[0]
                lf = fate[b, py, px]
                if 1 <= lf <= 4:
                    c["pn"] += 1; c["pm"] += pf[b, py, px] == lf
                if keys[b] and lf == 0:
                    dy, dx = DIRS[int(keys[b])]
                    if 0 <= py + dy < 45 and 0 <= px + dx < 45 and last[b][0][py + dy, px + dx] == 0:
                        c["on"] += 1; c["ob"] += pf[b, py, px] == 0
            dm = (k == 2) & (fate[b] >= 0)
            c["dall"] += dm.sum(); c["dp"] += ((1 <= pf[b]) & (pf[b] <= 4) & dm).sum()
            mv = dm & (fate[b] >= 1) & (fate[b] <= 4)
            c["dn"] += mv.sum(); c["dm"] += (mv & (pf[b] == fate[b])).sum()
            if len(pl):
                d0 = np.abs(yy - py) + np.abs(xx - px)
                for r, ff in enumerate([fate[b], pf[b]]):
                    m = dm & (d0 <= 6) & (ff <= 4)
                    for y, x in zip(*np.nonzero(m)):
                        mvv = ff[y, x]
                        if mvv == 0:
                            near[r, 2] += 1
                        else:
                            dy, dx = MOVES[mvv]
                            near[r, 0 if abs(y + dy - py) + abs(x + dx - px) < d0[y, x] else 1] += 1
    pct = lambda a, b: round(100 * a / max(b, 1), 1)
    fmt = lambda v: "/".join(str(round(100 * x / max(v.sum(), 1))) for x in v)
    return {"terrain_changed": pct(c["tc"], c["tn"]), "player_move_right": pct(c["pm"], c["pn"]),
            "ocean_block": pct(c["ob"], c["on"]), "dino_move_right": pct(c["dm"], c["dn"]),
            "dino_moves_pred": pct(c["dp"], c["dall"]), "chase_near": fmt(near[1]), "chase_near_real": fmt(near[0])}


@torch.no_grad()
def rollouts(model, val, n_eps=8, ticks=150) -> dict:
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    iso, share, dinos, sc = [], [], [], []
    for ep in [e for e in val if len(e.actions) > 60][:n_eps]:
        obs, acts = np.asarray(ep.obs), np.asarray(ep.actions)
        s = 20
        reset_memory(model)
        hist = [obs[s - i] for i in range(model.k - 1, -1, -1)]
        for t in range(s, min(s + ticks, len(acts))):
            nxt = model_step(model, hist, int(acts[t]), "cpu", "fates", gen)
            tr = nxt[0]
            c = tr[1:-1, 1:-1]
            iso.append(((c != tr[:-2, 1:-1]) & (c != tr[2:, 1:-1]) & (c != tr[1:-1, :-2]) & (c != tr[1:-1, 2:])).mean())
            share.append(np.bincount(tr.ravel(), minlength=8).max() / 2025)
            dinos.append(((nxt[1] >= 21) & (nxt[1] <= 40)).sum())
            if acts[t]:
                sc.append(find_scroll(hist[-1][0], tr) != (0, 0))
            hist.append(nxt)
    return {"static": round(100 * float(np.mean(iso)), 3), "max_share": round(100 * float(np.max(share))),
            "dinos": round(float(np.mean(dinos)), 2), "scroll_when_key": round(100 * float(np.mean(sc)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--name", default=None)
    args = p.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(args.checkpoint, map_location="cpu")
    c = ck["config"]
    model = build_model(c).eval()
    model.load_state_dict(ck["state"])
    model.dino_weight = c.get("dino_weight", 1.0)
    _, val = split_by_episode(find_episodes(*VAL_DATA), 0.1, seed=0)
    out = one_step(model.to(dev), Frames(val, model.k, dev, fates=True), dev)
    model = model.cpu()
    out |= rollouts(model, val)
    x, k = torch.zeros(1, model.k, 2, 45, 45, dtype=torch.long), torch.tensor([0])
    for _ in range(5):
        model.forward_parts(x, k)
    t0 = time.time()
    for _ in range(30):
        model.forward_parts(x, k)
    out["cpu_ms"] = round((time.time() - t0) / 30 * 1000, 1)
    name = args.name or Path(args.checkpoint).stem
    Path("scorecards").mkdir(exist_ok=True)
    Path(f"scorecards/{name}.json").write_text(json.dumps(out, indent=1))
    print(f"{name}: " + "  ".join(f"{k}={v}" for k, v in out.items()))


if __name__ == "__main__":
    main()
