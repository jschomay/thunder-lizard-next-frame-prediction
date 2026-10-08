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
    # chase/flee: for dinos, by distance band to the player, how often the real move and the
    # predicted fate head toward the player, away from it, or stay
    yy, xx = torch.meshgrid(torch.arange(45), torch.arange(45), indexing="ij")
    band = lambda d: torch.bucketize(d, torch.tensor([6, 13]), right=True)  # 0 near <=6, 1 mid, 2 far
    MV = torch.tensor([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])
    cf = torch.zeros(2, 3, 3, dtype=torch.long)  # [real/pred, band, toward/away/stay]
    for i in range(0, len(data), 256):
        ctx, keys, _, fate = data.batch(torch.arange(i, min(i + 256, len(data)), device=dev))
        pred = model.forward_parts(ctx, keys)[2].argmax(1)
        k = torch.from_numpy(kind(ctx[:, -1, 1].cpu().numpy()))
        ent = ctx[:, -1, 1].cpu()
        for b in range(len(ent)):
            pl = ((ent[b] >= 1) & (ent[b] <= 20)).nonzero()
            if not len(pl):
                continue
            py, px = pl[0]
            dm = (k[b] == 2) & (fate[b].cpu() >= 0) & (fate[b].cpu() <= 4)
            if not dm.any():
                continue
            d0 = (yy[dm] - py).abs() + (xx[dm] - px).abs()
            for r, f in enumerate([fate[b].cpu()[dm], pred[b].cpu()[dm].clamp(max=4)]):
                d1 = (yy[dm] + MV[f, 0] - py).abs() + (xx[dm] + MV[f, 1] - px).abs()
                cls = torch.where(f == 0, 2, torch.where(d1 < d0, 0, 1))
                cf[r].index_put_((band(d0), cls), torch.ones_like(cls), accumulate=True)
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
        if name == "dino":
            for i, bn in enumerate(["near (<=6)", "mid (7-13)", "far (>13)"]):
                r, q = cf[0, i].float(), cf[1, i].float()
                fmt = lambda v: f"toward {v[0] / v.sum():.0%} away {v[1] / v.sum():.0%} stay {v[2] / v.sum():.0%}"
                if r.sum():
                    print(f"   {bn:11s} n={int(r.sum()):5d}  real: {fmt(r)}  |  predicted: {fmt(q)}")
        print("   per true fate:", {NAMES[f]: f"{c[f, f].item() / c[f].sum().item():.0%} of {c[f].sum().item()}"
                                  for f in range(7) if c[f].sum()})


if __name__ == "__main__":
    main(sys.argv[1])
