"""Export a U-Net + split + fates checkpoint for the browser app (app/), plus starting states.

    uv run python -m tlfp.export_onnx checkpoints/s3-unet-fates-last.pt

Writes app/model.onnx and app/seeds.json. The browser does the one-hot encoding itself (cheap and
avoids ONNX one_hot quirks), so the model takes:
  x       float [1, k*50+5, 45, 45]  k one-hot frames (8 terrain + 42 entity each) + 5 key channels
  last_t  float [1, 8, 45, 45]       one-hot terrain of the last frame (copy prior)
  last_e  float [1, 42, 45, 45]      one-hot entities of the last frame (copy prior)
and returns terrain logits [1,8,45,45], entity log-probs [1,42,45,45], fate logits [1,7,45,45].
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tlfp.data import find_episodes, split_by_episode
from tlfp.model import build_model, combine, encode_input, group_entity

APP = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("app")  # optional output dir


class Exported(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, last_t, last_e, draft=None):
        m = self.m
        h = m.body(x)
        if m.diffuse:  # H16: draft = one-hot terrain + unknown (9) and the unknown share (1)
            h = h + m.draft_in(draft)
        if m.unet:
            g = F.interpolate(m.down(h), size=(45, 45), mode="nearest")
            h = h + m.up(torch.cat([h, g], 1))
        t = m.terrain_head(h) + m.copy_strength[0] * last_t
        e = combine(*group_entity(m.entity_head(h) + m.copy_strength[1] * last_e))
        return t, e, m.fate_head(h)


def main(path: str):
    ck = torch.load(path, map_location="cpu")
    c = ck["config"]
    model = build_model(c).eval()
    model.load_state_dict(ck["state"])
    assert model.split_entity and model.fates, "the app needs a split + fates model"
    wrap = Exported(model).eval()

    # starting states: 4 real frames from held-out games, at moments with dinos in view
    _, val = split_by_episode(find_episodes(*c["data"]), 0.1, seed=0)
    seeds = []
    for ep in val:
        obs = np.asarray(ep.obs)
        dinos = ((obs[:, 1] >= 21) & (obs[:, 1] <= 40)).sum((1, 2))
        for t in range(model.k + 10, len(obs) - 1, 40):
            if 2 <= dinos[t] <= 5:
                seeds.append({"episode": ep.path.name, "tick": t,
                              "frames": [[obs[s][0].ravel().tolist(), obs[s][1].ravel().tolist()]
                                         for s in range(t - model.k + 1, t + 1)]})
                break
        if len(seeds) >= 12:
            break

    ctx = torch.from_numpy(np.array([[[np.array(f[0]).reshape(45, 45), np.array(f[1]).reshape(45, 45)]
                                       for f in seeds[0]["frames"]]])).long()
    key = torch.tensor([2])
    x = encode_input(ctx, key)
    lt = F.one_hot(ctx[:, -1, 0], 8).permute(0, 3, 1, 2).float()
    le = F.one_hot(ctx[:, -1, 1], 42).permute(0, 3, 1, 2).float()
    dr = torch.cat([F.one_hot(torch.full((1, 45, 45), 8), 9).permute(0, 3, 1, 2).float(), torch.ones(1, 1, 45, 45)], 1)
    args = (x, lt, le, dr) if model.diffuse else (x, lt, le)
    names = ["x", "last_t", "last_e"] + (["draft"] if model.diffuse else [])
    with torch.no_grad():
        ours = wrap(*args)
        t, ep, f = model.forward_parts(ctx, key)
    assert all(torch.allclose(a, b, atol=1e-4) for a, b in zip(ours, (t, combine(*ep), f)))

    APP.mkdir(exist_ok=True)
    torch.onnx.export(wrap, args, APP / "model.onnx", input_names=names,
                      output_names=["terrain", "entity", "fate"], opset_version=17, dynamo=False)
    import onnxruntime as ort
    out = ort.InferenceSession(str(APP / "model.onnx")).run(None, dict(zip(names, (a.numpy() for a in args))))
    assert all(np.allclose(a, b.numpy(), atol=1e-3) for a, b in zip(out, ours)), "onnx output differs"
    (APP / "seeds.json").write_text(json.dumps({"run": c.get("name"), "k": model.k,
                                                "dino_weight": c.get("dino_weight", 1.0), "seeds": seeds}))
    print(f"wrote {APP / 'model.onnx'} ({(APP / 'model.onnx').stat().st_size / 1e6:.1f} MB), "
          f"{len(seeds)} seeds; onnx output matches torch")


if __name__ == "__main__":
    main(sys.argv[1])
