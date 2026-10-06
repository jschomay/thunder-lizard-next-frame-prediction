"""Play inside the net: feed a predictor its own output, tick after tick, with keys from a real game.

    uv run python -m tlfp.rollout checkpoints/<run>.pt --ticks 150

Starts every predictor from the same real frames of a held-out episode, replays that episode's
keys, and never shows them the real game again. Exports to the whiteboard "Play in the net" tab:
  real    the actual game
  argmax  model, most likely class per cell
  sample  model, a random draw from its probabilities (H5)
  rule    the hand-written shift_rhythm rule
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tlfp.baselines import predict as rule_predict
from tlfp.data import find_episodes, split_by_episode, KEYS
from tlfp.model import CNN

OUT = Path(".whiteboard/frame-prediction/rollouts.json")
B36 = "0123456789abcdefghijklmnopqrstuvwxyz!@#$%^"
enc = lambda f: ["".join(map(str, f[0].ravel())), "".join(B36[v] for v in f[1].ravel())]


@torch.no_grad()
def model_step(model, history: list[np.ndarray], key: int, device, sample: bool, gen: torch.Generator):
    ctx = torch.from_numpy(np.stack(history[-model.k:]).astype(np.int64))[None].to(device)
    t_logits, e_logits = model(ctx, torch.tensor([key], device=device))
    if not sample:
        return np.stack([t_logits.argmax(1)[0].cpu().numpy(), e_logits.argmax(1)[0].cpu().numpy()]).astype(np.uint8)
    def draw(logits):  # [1, C, 45, 45] -> sample a class per cell
        probs = logits.softmax(1)[0].permute(1, 2, 0).reshape(-1, logits.shape[1]).cpu()
        return torch.multinomial(probs, 1, generator=gen).reshape(45, 45).numpy()
    return np.stack([draw(t_logits), draw(e_logits)]).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--ticks", type=int, default=150)
    p.add_argument("--start", type=int, default=20)
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    c = ck["config"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CNN(c["k"], c["width"], c["depth"], copy_prior=c.get("copy_prior", False)).to(device).eval()
    model.load_state_dict(ck["state"])
    gen = torch.Generator().manual_seed(0)

    _, val = split_by_episode(find_episodes(*c["data"]), 0.1, seed=0)
    out = {"run": c.get("name"), "episodes": []}
    for ep in val[:args.episodes]:
        obs, acts = np.asarray(ep.obs), np.asarray(ep.actions)
        s, n = args.start, min(args.ticks, len(acts) - args.start)
        seed_frames = [obs[s - i] for i in range(model.k - 1, -1, -1)]  # same real starting history for all
        hist = {name: list(seed_frames) for name in ["argmax", "sample", "rule"]}
        rule_acts = list(acts[s - model.k + 1:s])  # the rule looks at the previous key too
        frames = {name: [enc(seed_frames[-1])] for name in ["real", "argmax", "sample", "rule"]}
        for t in range(s, s + n):
            key = int(acts[t])
            hist["argmax"].append(model_step(model, hist["argmax"], key, device, False, gen))
            hist["sample"].append(model_step(model, hist["sample"], key, device, True, gen))
            rule_acts.append(key)
            h = np.stack(hist["rule"][-2:])
            hist["rule"].append(rule_predict("shift_rhythm", h, np.array(rule_acts[-2:]), 1).copy())
            for name in ["argmax", "sample", "rule"]:
                frames[name].append(enc(hist[name][-1]))
            frames["real"].append(enc(obs[t + 1]))
        out["episodes"].append({"name": ep.path.name, "start": s,
                                "actions": [KEYS[a] for a in acts[s:s + n]], "frames": frames})
    OUT.write_text(json.dumps(out))
    print(f"wrote {len(out['episodes'])} rollouts x {n} ticks -> {OUT}")


if __name__ == "__main__":
    main()
