"""Real backtracking moments for the whiteboard's Transformer tab (H18).

    uv run python -m tlfp.export_transformer_viz

For a few held-out backtrack games, a moment b where the player has come back to a place seen at a
earlier tick a: the frame at b plus the frames the memory branch would read (ages 0-7, then every 8th
back to 128). The page lets you click a patch and see which old patches look like it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tlfp.data import find_episodes, split_by_episode
from tlfp.rollout import enc
from tlfp.scorecard import BACKTRACK_DATA, return_pairs

OUT = Path(".whiteboard/frame-prediction/transformer.json")
AGES = sorted(set(range(8)) | set(range(8, 129, 8)))


def main():
    _, val = split_by_episode(find_episodes(*BACKTRACK_DATA), 0.1, seed=0)
    out = {"ages": AGES, "moments": []}
    for ep in val:
        obs = np.asarray(ep.obs)
        for a, b in return_pairs(ep):
            if b - a < 40 or b < 128:
                continue
            out["moments"].append({"episode": ep.path.name, "now": b, "seen_at": a,
                                   "frames": {str(g): enc(obs[b - g]) for g in AGES}, "next": enc(obs[b + 1])})
            break
        if len(out["moments"]) >= 4:
            break
    OUT.write_text(json.dumps(out))
    print(f"wrote {len(out['moments'])} moments -> {OUT}", [(m['now'], m['seen_at']) for m in out["moments"]])


if __name__ == "__main__":
    main()
