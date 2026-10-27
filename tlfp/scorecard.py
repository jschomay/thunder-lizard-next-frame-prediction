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
  max_share           biggest single terrain type, worst frame (real ~59%; one frame, so noisy)
  share_mean          biggest single terrain type, averaged over frames (steadier smear measure);
                      share_mean_real is the same for the real game over the same ticks
  dinos               dinos per frame (real ~1.5)
  scroll_when_key     ticks with a key held where the view scrolled (real ~44%)
  continuity          walk away and come back (backtrack games): share of terrain where the model's
                      frame on return matches the frame it left from (real ~95%+; current models ~0?)
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
from tlfp.model import build_model, diffuse_terrain
from tlfp.moves import MOVES, find_scroll, kind
from tlfp.rollout import model_step, reset_memory
from tlfp.train import Frames

VAL_DATA = ["data/stage3-100k", "data/stage3-100k-b"]
BACKTRACK_DATA = ["data/stage3-backtrack", "data/stage3-backtrack-b"]


def return_pairs(ep, max_pairs=2):
    """(a, b): the real view at b is ~the view at a, and it went elsewhere in between"""
    t = np.asarray(ep.obs)[:, 0]
    out = []
    a = 4
    while a < len(t) - 25 and len(out) < max_pairs:
        for b in range(a + 20, min(a + 120, len(t))):
            if (t[a] == t[b]).mean() > 0.9 and min((t[a] == t[c]).mean() for c in range(a + 1, b)) < 0.5:
                out.append((a, b)); a = b
                break
        a += 5
    return out


@torch.no_grad()
def continuity(model, n_pairs=20) -> float:
    """roll out from a with the real keys until b; compare the model's frame at b to the frame at a"""
    _, val = split_by_episode(find_episodes(*BACKTRACK_DATA), 0.1, seed=0)
    gen = torch.Generator().manual_seed(0)
    scores = []
    for ep in val:
        obs, acts = np.asarray(ep.obs), np.asarray(ep.actions)
        for a, b in return_pairs(ep):
            reset_memory(model)
            hist = [obs[a - i] for i in range(model.k - 1, -1, -1)]
            for t in range(a, b):
                hist.append(model_step(model, hist, int(acts[t]), "cpu", "fates", gen))
            start, back = obs[a][0], hist[-1][0]
            keep = (start != 6) & (start != 7) & (back != 6) & (back != 7)  # lava changes on its own
            scores.append(float((start == back)[keep].mean()))
        if len(scores) >= n_pairs:
            break
    return round(100 * float(np.mean(scores)), 1) if scores else float("nan")
DIRS = {1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1)}


@torch.no_grad()
def one_step(model, data, dev) -> dict:
    c = dict.fromkeys(["tc", "tn", "pm", "pn", "ob", "on", "dm", "dn", "dp", "dall"], 0)
    near = np.zeros((2, 3))
    yy, xx = np.mgrid[0:45, 0:45]
    bs = 32 if getattr(model, "memory", None) is not None else 256  # transformer: keep memory use small
    for i in range(0, len(data), bs):
        ctx, keys, tgt, fate = data.batch(torch.arange(i, min(i + bs, len(data)), device=dev))
        model.rel = data.batch_rel
        if getattr(model, "diffuse", False):  # H16: the full few-pass decode, argmax everywhere
            pt, _, (t, _, f) = diffuse_terrain(model, ctx, keys)
            pf = f.argmax(1)
        else:
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
def rollouts(model, val, n_eps=8, ticks=150) -> dict:  # n_eps: --eps
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    iso, share, dinos, sc, real_share = [], [], [], [], []
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
            real_share.append(np.bincount(obs[t + 1][0].ravel(), minlength=8).max() / 2025)
    return {"static": round(100 * float(np.mean(iso)), 3), "max_share": round(100 * float(np.max(share))),
            "share_mean": round(100 * float(np.mean(share)), 1),
            "share_mean_real": round(100 * float(np.mean(real_share)), 1), "dinos": round(float(np.mean(dinos)), 2), "scroll_when_key": round(100 * float(np.mean(sc)))}


def edges(t: np.ndarray) -> float:
    """share of neighbouring cell pairs with different terrain: blobs keep edges, smear erases them"""
    return float(((t[1:] != t[:-1]).mean() + (t[:, 1:] != t[:, :-1]).mean()) / 2)


@torch.no_grad()
def walk(model, val, n_eps=12, ticks=300, window=50) -> dict:
    """the long walk: the model plays `ticks` ticks on its own with a real game's keys. Compares the
    first and last `window` ticks with the real game over the same ticks: terrain edges (smear erases
    them), the terrain-type mix (total variation distance from the real mix, 0 = same), stray cells."""
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    early, late, real_e, real_l, mix, stray = [], [], [], [], [], []
    for ep in [e for e in val if len(e.actions) > ticks + 25][:n_eps]:
        obs, acts = np.asarray(ep.obs), np.asarray(ep.actions)
        s = 20
        reset_memory(model)
        hist = [obs[s - i] for i in range(model.k - 1, -1, -1)]
        frames = []
        for t in range(s, s + ticks):
            hist.append(model_step(model, hist, int(acts[t]), "cpu", "fates", gen))
            frames.append(hist[-1][0])
            if len(hist) > 200:
                hist = hist[-140:]
        real = [obs[t + 1][0] for t in range(s, s + ticks)]
        early.append(np.mean([edges(f) for f in frames[:window]]))
        late.append(np.mean([edges(f) for f in frames[-window:]]))
        real_e.append(np.mean([edges(f) for f in real[:window]]))
        real_l.append(np.mean([edges(f) for f in real[-window:]]))
        hm = np.bincount(np.concatenate([f.ravel() for f in frames[-window:]]), minlength=8) / (window * 2025)
        hr = np.bincount(np.concatenate([f.ravel() for f in real[-window:]]), minlength=8) / (window * 2025)
        mix.append(0.5 * np.abs(hm - hr).sum())
        stray.append(np.mean([((c := f[1:-1, 1:-1]) != f[:-2, 1:-1]) & (c != f[2:, 1:-1]) & (c != f[1:-1, :-2]) & (c != f[1:-1, 2:])
                              for f in frames[-window:]]))
    r = lambda x: round(100 * float(np.mean(x)), 2)
    return {"walk_edges_early": r(early), "walk_edges_late": r(late), "walk_edges_real_early": r(real_e),
            "walk_edges_real_late": r(real_l), "walk_mix_late": r(mix), "walk_static_late": r(stray),
            "walk_games": len(early)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--name", default=None)
    p.add_argument("--eps", type=int, default=8, help="rollout games (8 is quick but noisy)")
    p.add_argument("--steps", type=int, default=4, help="diffusion passes per frame (H16/H17)")
    p.add_argument("--temp", type=float, default=0.0, help="diffusion: draw every terrain cell at this temperature (0 = argmax)")
    p.add_argument("--order-noise", type=float, default=0.0, help="diffusion: randomness in which cells are fixed first (MaskGIT)")
    p.add_argument("--device", default=None, help="cpu for transformer models (MPS runs out of shared memory)")
    p.add_argument("--walk-only", action="store_true", help="only the long-walk score (quick)")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()
    dev = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location="cpu")
    c = ck["config"]
    model = build_model(c).eval()
    model.load_state_dict(ck["state"])
    model.dino_weight = c.get("dino_weight", 1.0)
    model.steps, model.diffuse_temp, model.order_noise = args.steps, args.temp, args.order_noise
    _, val = split_by_episode(find_episodes(*VAL_DATA), 0.1, seed=0)
    if args.walk_only:
        print(walk(model.cpu(), val))
        return
    out = one_step(model.to(dev), Frames(val, model.k, dev, fates=True, offsets=model.offsets,
                                        world=bool(model.memory is not None and model.memory.world)), dev)
    model = model.cpu()
    out |= rollouts(model, val, n_eps=args.eps)
    out["continuity"] = continuity(model)
    out |= walk(model, val)
    x, k = torch.zeros(1, len(model.offsets), 2, 45, 45, dtype=torch.long), torch.tensor([0])
    frame = (lambda: diffuse_terrain(model, x, k)) if getattr(model, "diffuse", False) else (lambda: model.forward_parts(x, k))
    for _ in range(5):
        frame()
    t0 = time.time()
    for _ in range(30):
        frame()
    out["cpu_ms"] = round((time.time() - t0) / 30 * 1000, 1)
    name = args.name or Path(args.checkpoint).stem
    Path("scorecards").mkdir(exist_ok=True)
    Path(f"scorecards/{name}.json").write_text(json.dumps(out, indent=1))
    print(f"{name}: " + "  ".join(f"{k}={v}" for k, v in out.items()))
    if not args.no_wandb:
        push_to_wandb(c, out)


def push_to_wandb(config: dict, scores: dict):
    """attach the scorecard to the training run in W&B as score/* summary values (for the
    parallel-coordinates chart); finds the run by its saved id, else by name"""
    try:
        import wandb
        api = wandb.Api()
        path = f"{api.default_entity}/thunder-lizard-frame-prediction"
        if config.get("wandb_id"):
            runs = [api.run(f"{path}/{config['wandb_id']}")]
        else:
            runs = list(api.runs(path, filters={"display_name": config.get("name")}))
        for r in runs:
            for k, v in scores.items():
                if isinstance(v, (int, float)):
                    r.summary[f"score/{k}"] = v
            r.summary.update()
        print(f"  attached to W&B run(s): {[r.name for r in runs]}")
    except Exception as err:  # never lose a scorecard over W&B
        print(f"  (W&B upload skipped: {err})")


if __name__ == "__main__":
    main()
