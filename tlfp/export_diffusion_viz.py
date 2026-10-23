"""Real held-out moments for the whiteboard's Diffusion tab (H16).

    uv run python -m tlfp.export_diffusion_viz checkpoints/best-so-far.pt [checkpoints/s3-diffuse-last.pt]

For each moment: the last real frame, the real next frame, the one-shot model's terrain guess and how
sure it was per cell, and (with a diffusion checkpoint) the draft and confidence after every pass.
Confidence is written as one digit per cell, 0 = 0-10% ... 9 = 90-100%.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from tlfp.data import find_episodes, split_by_episode
from tlfp.model import build_model, diffuse_terrain
from tlfp.moves import find_scroll
from tlfp.rollout import enc

OUT = Path(".whiteboard/frame-prediction/diffusion.json")
digits = lambda c: "".join(str(min(9, int(v * 10))) for v in c.ravel())


def load(path):
    ck = torch.load(path, map_location="cpu")
    m = build_model(ck["config"]).eval()
    m.load_state_dict(ck["state"])
    return m, ck["config"]


@torch.no_grad()
def main():
    one, c = load(sys.argv[1])
    dif = load(sys.argv[2])[0] if len(sys.argv) > 2 else None
    _, val = split_by_episode(find_episodes(*c["data"]), 0.1, seed=0)
    out = {"oneshot": c.get("name"), "diffusion": sys.argv[2] if dif else None, "moments": []}
    for ep in val[:40]:
        obs, acts = np.asarray(ep.obs), np.asarray(ep.actions)
        # a moment where the view scrolls (new terrain comes in) or lava is moving
        for t in range(30, len(acts) - 1, 7):
            scroll = find_scroll(obs[t][0], obs[t + 1][0]) != (0, 0)
            lava = (obs[t][0] == 6).sum() > 20
            if scroll and (lava or len(out["moments"]) % 2 == 0):
                break
        else:
            continue
        ctx = torch.from_numpy(obs[t - 3:t + 1].astype(np.int64))[None]
        key = torch.tensor([int(acts[t])])
        p = one.forward_parts(ctx, key)[0].softmax(1)[0]
        m = {"episode": ep.path.name, "tick": t, "key": int(acts[t]), "last": enc(obs[t]), "real": enc(obs[t + 1]),
             "oneshot": "".join(map(str, p.argmax(0).numpy().ravel())), "oneshot_conf": digits(p.max(0).values.numpy())}
        if dif:
            _, drafts, _ = diffuse_terrain(dif, ctx, key)
            m["passes"] = [{"draft": "".join(map(str, d[0].numpy().ravel())), "conf": digits(cf[0].numpy())}
                           for d, cf in drafts]
        out["moments"].append(m)
        if len(out["moments"]) >= 8:
            break
    OUT.write_text(json.dumps(out))
    print(f"wrote {len(out['moments'])} moments -> {OUT}")


if __name__ == "__main__":
    main()
