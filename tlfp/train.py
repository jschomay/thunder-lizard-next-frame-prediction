"""Train the CNN on recorded episodes and log to Weights & Biases.

    uv run python -m tlfp.train --data data/stage1 --k 2 --epochs 3
    uv run python -m tlfp.train --data data/stage1 --k 1 --no-wandb   # local only
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tlfp.data import find_episodes, split_by_episode
from tlfp.metrics import add, frame_metrics, summarize
from tlfp.model import build_model, split_ids


class Frames:
    """All episodes in one in-memory uint8 tensor, with window indices that never cross episodes."""

    def __init__(self, episodes, k: int, device):
        obs, acts, idx = [], [], []
        offset = 0
        for ep in episodes:
            o = np.asarray(ep.obs)
            obs.append(o)
            acts.append(np.asarray(ep.actions))
            for t in range(len(ep)):
                # frame indices for the k-window ending at t, padded with the episode's first frame
                idx.append([offset + max(0, s) for s in range(t - k + 1, t + 1)] + [offset + t + 1])
            offset += len(o)
        self.obs = torch.from_numpy(np.concatenate(obs)).to(device)              # [F, 2, 45, 45] uint8
        self.idx = torch.tensor(idx, device=device)                              # [N, k+1]
        a = np.concatenate([np.append(x, 0) for x in acts])                      # align actions with frames
        self.keys = torch.from_numpy(a.astype(np.int64)).to(device)
        self.k = k

    def __len__(self):
        return len(self.idx)

    def batch(self, rows: torch.Tensor):
        ix = self.idx[rows]
        context = self.obs[ix[:, :-1]].long()       # [B, k, 2, 45, 45]
        target = self.obs[ix[:, -1]].long()         # [B, 2, 45, 45]
        keys = self.keys[ix[:, -2]]                 # key at the last context frame
        return context, keys, target


def evaluate(model, data: Frames, batch_size=256) -> dict[str, float]:
    model.eval()
    totals: dict[str, int] = {}
    for i in range(0, len(data), batch_size):
        rows = torch.arange(i, min(i + batch_size, len(data)), device=data.obs.device)
        context, keys, target = data.batch(rows)
        pred = model.predict(context, keys)
        add(totals, frame_metrics(context[:, -1].cpu().numpy(), pred.cpu().numpy(), target.cpu().numpy()))
    model.train()
    return summarize(totals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=["data/stage1"])
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--evals-per-epoch", type=int, default=4)
    p.add_argument("--changed-weight", type=float, default=1.0, help="loss weight for cells that change (H9)")
    p.add_argument("--copy-prior", action="store_true", help="default to copying the last frame (H10)")
    p.add_argument("--warmup", type=float, default=0.05, help="fraction of steps for LR warmup")
    p.add_argument("--clip", type=float, default=1.0, help="gradient norm clip (0 = off)")
    p.add_argument("--max-steps", type=int, default=0, help="stop early (smoke tests)")
    p.add_argument("--split-entity", action="store_true", help="what's-here / facing / level heads (H11)")
    p.add_argument("--unet", action="store_true", help="add a zero-initialized shrink-and-grow branch")
    p.add_argument("--init", default=None, help="start from a checkpoint's weights (warm start, H4)")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_eps, val_eps = split_by_episode(find_episodes(*args.data), 0.1, seed=0)
    train, val = Frames(train_eps, args.k, device), Frames(val_eps, args.k, device)
    model = build_model(vars(args)).to(device)
    if args.init:
        # copy every weight whose name and shape match; new parts (split heads, U-Net branch) start fresh
        old, own = torch.load(args.init, map_location=device)["state"], model.state_dict()
        keep = {k: v for k, v in old.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        print(f"warm start from {args.init}: {len(keep)}/{len(own)} tensors copied, "
              f"new: {sorted({k.split('.')[0] for k in own if k not in keep})}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} train={len(train)} val={len(val)} params={n_params:,}")

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project="thunder-lizard-frame-prediction", name=args.name,
                         config={**vars(args), "params": n_params, "train_examples": len(train)})

    steps_per_epoch = len(train) // args.batch
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup))
    # linear warmup, then cosine decay to 5% of the peak learning rate
    lr_at = lambda s: args.lr * (s / warmup_steps if s < warmup_steps else
                                 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / (total_steps - warmup_steps))))
    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)
    ckpt = Path("checkpoints") / f"{args.name or 'run'}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    best = -1.0
    step, t0 = 0, time.time()
    if args.init:  # score the starting point, so a warm start that gets worse is obvious
        m = evaluate(model, val)
        print(f"start        val dynamic {m['dynamic_acc']:.2%}  changed {m['changed_acc']:.2%}  exact {m['exact_frame_acc']:.2%}")
        if run:
            run.log({f"val/{k}": v for k, v in m.items()}, step=0)
    for epoch in range(args.epochs):
        order = torch.randperm(len(train), device=device)
        for b in range(steps_per_epoch):
            context, keys, target = train.batch(order[b * args.batch:(b + 1) * args.batch])
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            t_logits, e_out = model.forward_parts(context, keys)
            # per-cell losses, with cells that change weighted up (H9)
            changed = context[:, -1] != target
            w = 1 + (args.changed_weight - 1) * changed.float()
            ce_t = F.cross_entropy(t_logits, target[:, 0], reduction="none")
            loss_t = (ce_t * w[:, 0]).sum() / w[:, 0].sum()
            if args.split_entity:
                # "what's here" is scored everywhere; facing and level only where there is a dino
                kind, dir_, level = split_ids(target[:, 1])
                dino = ((kind == 1) | (kind == 2)).float() * w[:, 1]
                ce_e = (F.cross_entropy(e_out[0], kind, reduction="none") * w[:, 1]).sum() / w[:, 1].sum()
                ce_e = ce_e + (F.cross_entropy(e_out[1], dir_, reduction="none") * dino).sum() / dino.sum().clamp(min=1)
                loss_e = ce_e + (F.cross_entropy(e_out[2], level, reduction="none") * dino).sum() / dino.sum().clamp(min=1)
            else:
                ce_e = F.cross_entropy(e_out, target[:, 1], reduction="none")
                loss_e = (ce_e * w[:, 1]).sum() / w[:, 1].sum()
            loss = loss_t + loss_e
            opt.zero_grad()
            loss.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
            if run and step % 20 == 0:
                run.log({"train/loss": loss.item(), "train/loss_terrain": loss_t.item(),
                         "train/loss_entity": loss_e.item(), "train/lr": lr_at(step),
                         "epoch": step / steps_per_epoch}, step=step)
            if step % eval_every == 0:
                m = evaluate(model, val)
                print(f"epoch {step / steps_per_epoch:5.2f}  loss {loss.item():.4f}  "
                      f"val dynamic {m['dynamic_acc']:.2%}  changed {m['changed_acc']:.2%}  "
                      f"exact {m['exact_frame_acc']:.2%}  ({time.time() - t0:.0f}s)")
                if m["dynamic_acc"] > best:
                    best = m["dynamic_acc"]
                    torch.save({"config": vars(args), "state": model.state_dict(), "step": step, "val": m}, ckpt)
                    print(f"  saved best -> {ckpt}")
                if run:
                    run.log({f"val/{k}": v for k, v in m.items()} | {"val/best_dynamic_acc": best}, step=step)
    if run:
        run.finish()


if __name__ == "__main__":
    main()
