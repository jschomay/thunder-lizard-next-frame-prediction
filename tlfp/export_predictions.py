"""Run a trained checkpoint on validation episodes and export for the whiteboard Predictions tab.

    uv run python -m tlfp.export_predictions checkpoints/<run>.pt
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from tlfp.baselines import predict as baseline_predict
from tlfp.data import KEYS, find_episodes, split_by_episode
from tlfp.model import CNN
from tlfp.train import Frames

OUT = Path(".whiteboard/frame-prediction/predictions.json")
B36 = "0123456789abcdefghijklmnopqrstuvwxyz!@#$%^"
enc_t = lambda f: "".join(map(str, f[0].ravel()))
enc_e = lambda f: "".join(B36[v] for v in f[1].ravel())


def main(path: str, n_episodes: int = 2, max_ticks: int = 200):
    ck = torch.load(path, map_location="cpu")
    cfg = ck["config"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CNN(cfg["k"], cfg["width"], cfg["depth"], copy_prior=cfg.get("copy_prior", False)).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    _, val_eps = split_by_episode(find_episodes(*cfg["data"]), 0.1, seed=0)
    out = {"run": cfg.get("name"), "checkpoint_step": ck["step"], "val": ck["val"], "episodes": []}
    for ep in val_eps[:n_episodes]:
        data = Frames([ep], cfg["k"], device)
        n = min(len(data), max_ticks)
        context, keys, target = data.batch(torch.arange(n, device=device))
        pred = model.predict(context, keys).cpu().numpy()
        obs = np.asarray(ep.obs)
        out["episodes"].append({
            "name": ep.path.name,
            "actions": [KEYS[a] for a in ep.actions[:n]],
            "prev": [[enc_t(f), enc_e(f)] for f in obs[:n]],
            "truth": [[enc_t(f), enc_e(f)] for f in obs[1:n + 1]],
            "pred": [[enc_t(f), enc_e(f)] for f in pred],
            "rule": [[enc_t(f), enc_e(f)] for f in (baseline_predict("shift_rhythm", obs, ep.actions, t) for t in range(n))],
        })
    OUT.write_text(json.dumps(out))
    print(f"wrote {len(out['episodes'])} episodes -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
