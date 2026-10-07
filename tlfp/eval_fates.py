"""One-step fate accuracy for a fate-head checkpoint (H13), on its validation split.

    uv run python -m tlfp.eval_fates checkpoints/<run>.pt
"""

from __future__ import annotations

import sys

import torch

from tlfp.data import find_episodes, split_by_episode
from tlfp.model import build_model
from tlfp.moves import kind
from tlfp.train import Frames

NAMES = ["stay", "up", "right", "down", "left", "die", "gone"]


@torch.no_grad()
def main(path: str):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(path, map_location=dev)
    model = build_model(ck["config"]).to(dev).eval()
    model.load_state_dict(ck["state"])
    _, val = split_by_episode(find_episodes(*ck["config"]["data"]), 0.1, seed=0)
    data = Frames(val, model.k, dev, fates=True)
    conf = torch.zeros(3, 7, 7, dtype=torch.long)  # [player/dino/skeleton, true, predicted]
    for i in range(0, len(data), 256):
        ctx, keys, _, fate = data.batch(torch.arange(i, min(i + 256, len(data)), device=dev))
        pred = model.forward_parts(ctx, keys)[2].argmax(1)
        k = torch.from_numpy(kind(ctx[:, -1, 1].cpu().numpy()))
        for j, kk in enumerate((1, 2, 3)):
            m = (k == kk) & (fate.cpu() >= 0)
            idx = fate.cpu()[m] * 7 + pred.cpu()[m]
            conf[j] += torch.bincount(idx, minlength=49).reshape(7, 7)
    for j, name in enumerate(["player", "dino", "skeleton"]):
        c = conf[j]
        tot = c.sum().item()
        if not tot:
            continue
        moved = c[1:5].sum().item()
        print(f"{name}: {tot} labelled, overall {c.diag().sum().item() / tot:.1%}, "
              f"moved ones right {c[1:5, 1:5].diag().sum().item() / max(moved, 1):.1%} of {moved}, "
              f"predicted to move {c[:, 1:5].sum().item() / tot:.1%} vs real {moved / tot:.1%}")
        print("   per true fate:", {NAMES[f]: f"{c[f, f].item() / c[f].sum().item():.0%} of {c[f].sum().item()}"
                                  for f in range(7) if c[f].sum()})


if __name__ == "__main__":
    main(sys.argv[1])
