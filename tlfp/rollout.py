"""Play inside the net: feed a predictor its own output, tick after tick, with keys from a real game.

    uv run python -m tlfp.rollout checkpoints/<run>.pt --ticks 150

Starts every predictor from the same real frames of a held-out episode, replays that episode's
keys, and never shows them the real game again. Exports to the whiteboard "Play in the net" tab:
  real    the actual game
  argmax  model, most likely class per cell
  sample  model, a random draw from its probabilities (H5)
  hybrid  argmax, except lava cells and their neighbors are sampled (spreading and burning out
          are both per-cell dice rolls in the game)
  rule    the hand-written shift_rhythm rule
  fates   (fate-head models, H13) terrain as in hybrid; every existing entity is moved by its
          predicted fate, so none can duplicate or vanish by accident; new dinos are sampled only
          in the 2-cell ring at the screen edge, where they can walk or scroll into view
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tlfp.baselines import predict as rule_predict
from tlfp.data import find_episodes, split_by_episode, KEYS
from tlfp.model import build_model, combine
from tlfp.moves import MOVES, DIE, find_scroll, kind

SPAWN = "revealed"  # where new dinos may appear in fates mode: ring | revealed | none (--spawn)
OUT = Path(".whiteboard/frame-prediction/rollouts.json")  # overridden by --out
B36 = "0123456789abcdefghijklmnopqrstuvwxyz!@#$%^"
enc = lambda f: ["".join(map(str, f[0].ravel())), "".join(B36[v] for v in f[1].ravel())]


def next_to_lava(terrain: np.ndarray) -> np.ndarray:
    lava = terrain == 6
    adj = np.zeros_like(lava)
    adj[1:] |= lava[:-1]; adj[:-1] |= lava[1:]; adj[:, 1:] |= lava[:, :-1]; adj[:, :-1] |= lava[:, 1:]
    return adj & ~lava


@torch.no_grad()
def model_step(model, history: list[np.ndarray], key: int, device, sample, gen: torch.Generator):
    """sample: False = argmax, True = sample every cell, "hybrid" = sample only lava and its neighbors"""
    ctx = torch.from_numpy(np.stack(history[-model.k:]).astype(np.int64))[None].to(device)
    t_logits, e_logits = model(ctx, torch.tensor([key], device=device))
    arg = np.stack([t_logits.argmax(1)[0].cpu().numpy(), e_logits.argmax(1)[0].cpu().numpy()]).astype(np.uint8)
    if not sample:
        return arg
    def draw(logits):  # [1, C, 45, 45] -> sample a class per cell
        probs = logits.softmax(1)[0].permute(1, 2, 0).reshape(-1, logits.shape[1]).cpu()
        return torch.multinomial(probs, 1, generator=gen).reshape(45, 45).numpy()
    if sample == "fates":
        return fates_step(model, history, key, device, gen, draw)
    if sample == "hybrid":
        out = arg.copy()
        m = next_to_lava(history[-1][0]) | (history[-1][0] == 6)
        out[0][m] = draw(t_logits)[m]
        return out
    return np.stack([draw(t_logits), draw(e_logits)]).astype(np.uint8)


@torch.no_grad()
def fates_step(model, history, key, device, gen, draw):
    ctx = torch.from_numpy(np.stack(history[-model.k:]).astype(np.int64))[None].to(device)
    t_logits, e_parts, f_logits = model.forward_parts(ctx, torch.tensor([key], device=device))
    e_logits = combine(*e_parts) if model.split_entity else e_parts
    last = history[-1]
    fate = f_logits.argmax(1)[0].cpu().numpy()
    # H14: the scroll is the player's predicted move (the view follows the player). Terrain only
    # changes through lava in the game, so shift what was visible; the net draws only the newly
    # revealed row/column, plus dice for lava and its neighbours
    # the player: dice for *whether* it moves (from standstill the net often says ~30%, and argmax
    # would then never get going), the net's top choice for *which way*
    sy, sx = 0, 0
    pl = np.argwhere((last[1] >= 1) & (last[1] <= 20))
    if len(pl):
        py, px = pl[0]
        pf = f_logits[0, :, py, px].softmax(0).cpu()
        moved = torch.rand(1, generator=gen).item() < pf[1:5].sum().item()
        fate[py, px] = 1 + int(pf[1:5].argmax()) if moved else (int(pf.argmax()) if pf.argmax() > 4 else 0)
        if 1 <= fate[py, px] <= 4:
            sy, sx = MOVES[fate[py, px]]
    net_t = t_logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    terrain = net_t.copy()
    ys, xs = np.mgrid[0:45, 0:45]
    oy, ox = ys + sy, xs + sx
    inside = (oy >= 0) & (oy < 45) & (ox >= 0) & (ox < 45)
    terrain[inside] = last[0][oy[inside], ox[inside]]
    m = next_to_lava(terrain) | (terrain == 6)
    terrain[m] = draw(t_logits)[m]
    # dinos: draw the fate from its probabilities (dino moves are partly random in the game, and
    # argmax nearly always says "stay"); the player keeps the most likely fate
    fate_draw = torch.multinomial(f_logits.softmax(1)[0].permute(1, 2, 0).reshape(-1, 7).cpu(), 1,
                                  generator=gen).reshape(45, 45).numpy()
    fate = np.where((last[1] >= 21) & (last[1] <= 40), fate_draw, fate)
    ent = np.zeros((45, 45), np.uint8)
    for y, x in zip(*np.nonzero(last[1])):
        e, f = int(last[1][y, x]), int(fate[y, x])
        if f == 6:  # gone
            continue
        if f == DIE or e == 41:
            dy, dx, e = 0, 0, 41 if f == DIE else e
        else:
            dy, dx = MOVES[f]
            if f > 0:  # moved: face the way it went (id = base + level*4 + dir)
                base = 1 if e <= 20 else 21
                e = base + ((e - base) // 4) * 4 + (f - 1)
        ty, tx = y + dy - sy, x + dx - sx
        if e <= 20 and (y, x) == (22, 22):
            ty, tx = 22, 22  # the view follows the player
        if not (0 <= ty < 45 and 0 <= tx < 45) or ent[ty, tx]:
            ty, tx = y - sy, x - sx  # blocked or off-screen: stay
            if not (0 <= ty < 45 and 0 <= tx < 45) or ent[ty, tx]:
                continue
        ent[ty, tx] = e
    ring = np.ones((45, 45), bool); ring[1:-1, 1:-1] = False  # outermost row/column only
    if SPAWN == "revealed":  # only the row/column the scroll just revealed
        ring[:] = False
        if sy: ring[-1 if sy > 0 else 0] = True
        if sx: ring[:, -1 if sx > 0 else 0] = True
    elif SPAWN == "none":
        ring[:] = False
    # training weighted dino cells by dino_weight, which multiplies the model's odds of "dino" by
    # that factor; undo it before rolling dice, or the edge fills up with dinos
    lp = e_logits.log_softmax(1)[0].cpu()
    w = getattr(model, "dino_weight", 1.0)
    lp[21:41] -= np.log(w)
    new = torch.multinomial(lp.softmax(0).permute(1, 2, 0).reshape(-1, lp.shape[0]), 1, generator=gen).reshape(45, 45).numpy()
    spawn = ring & (ent == 0) & (kind(new) == 2)
    ent[spawn] = new[spawn]
    return np.stack([terrain, ent]).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--ticks", type=int, default=150)
    p.add_argument("--start", type=int, default=20, help="start tick (-1 = the tick with the most lava edge in view)")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--data", default=None, help="roll out these episodes instead of the run's validation split")
    p.add_argument("--spawn", default="revealed", choices=["ring", "revealed", "none"])
    p.add_argument("--no-rule", action="store_true", help="skip the hand-written rule (it doesn't model lava)")
    args = p.parse_args()
    global SPAWN
    SPAWN = args.spawn

    ck = torch.load(args.checkpoint, map_location="cpu")
    c = ck["config"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_model(c).to(device).eval()
    model.load_state_dict(ck["state"])
    model.dino_weight = c.get("dino_weight", 1.0)
    gen = torch.Generator().manual_seed(0)

    _, val = split_by_episode(find_episodes(*c["data"]), 0.1, seed=0)
    if args.data:
        val = find_episodes(args.data)
    out = {"run": c.get("name"), "episodes": []}
    for ep in [e for e in val if len(e.actions) > 60][:args.episodes]:
        obs, acts = np.asarray(ep.obs), np.asarray(ep.actions)
        s = args.start
        if s < 0:  # pick the moment with the most lava frontier in view
            s = max(range(model.k, len(acts) - args.ticks), key=lambda t: next_to_lava(obs[t][0]).sum(), default=model.k)
        if len(acts) - s < 20:  # the game ended too soon after the start
            continue
        n = min(args.ticks, len(acts) - s)
        seed_frames = [obs[s - i] for i in range(model.k - 1, -1, -1)]  # same real starting history for all
        names = ["argmax", "hybrid", "sample"] + (["fates"] if model.fates else []) + ([] if args.no_rule else ["rule"])
        hist = {name: list(seed_frames) for name in names}
        rule_acts = list(acts[s - model.k + 1:s])  # the rule looks at the previous key too
        frames = {name: [enc(seed_frames[-1])] for name in ["real"] + names}
        for t in range(s, s + n):
            key = int(acts[t])
            hist["argmax"].append(model_step(model, hist["argmax"], key, device, False, gen))
            hist["sample"].append(model_step(model, hist["sample"], key, device, True, gen))
            hist["hybrid"].append(model_step(model, hist["hybrid"], key, device, "hybrid", gen))
            if "fates" in hist:
                hist["fates"].append(model_step(model, hist["fates"], key, device, "fates", gen))
            if "rule" in hist:
                rule_acts.append(key)
                h = np.stack(hist["rule"][-2:])
                hist["rule"].append(rule_predict("shift_rhythm", h, np.array(rule_acts[-2:]), 1).copy())
            for name in names:
                frames[name].append(enc(hist[name][-1]))
            frames["real"].append(enc(obs[t + 1]))
        out["episodes"].append({"name": ep.path.name, "start": s,
                                "actions": [KEYS[a] for a in acts[s:s + n]], "frames": frames})
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {len(out['episodes'])} rollouts x {n} ticks -> {args.out}")


if __name__ == "__main__":
    main()
