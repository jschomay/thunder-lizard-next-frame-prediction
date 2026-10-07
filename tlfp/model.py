"""Approach A: a plain CNN that predicts the next state grid.

Input per cell: k frames x (8 terrain + 42 entity one-hot) + 5 key one-hot (same in every cell).
Output per cell: 8 terrain scores and 42 entity scores (logits) for the next frame.

Options for stage 3:
  split_entity (H11): instead of one 42-way guess, ask three questions per cell, "what's here"
      (empty / player / dino / skeleton), "which way does it face" (4) and "what level" (5).
      The three answers are recombined into the same 42 entity scores, so everything
      downstream (metrics, rollouts) is unchanged. Splitting stops the player's probability
      being spread over 20 look-alike classes, where "empty" can win even when P(player) > 50%.
  unet: a shrink-and-grow branch that sees the whole screen at 1/4 size, added to the
      full-size features. Its last layer starts at zero, so a warm-started model begins
      exactly as the plain CNN and learns to use the branch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tlfp.data import N_ENTITY, N_TERRAIN

N_KEYS = 5
N_KIND, N_DIR, N_LEVEL = 4, 4, 5  # kind: 0 empty, 1 player, 2 dino, 3 skeleton


def split_ids(e: torch.Tensor):
    """entity id -> (kind, dir, level); dir/level are 0 where there's no dino"""
    dino = (e >= 1) & (e <= 40)
    kind = torch.where(e == 0, 0, torch.where(e == 41, 3, torch.where(e <= 20, 1, 2)))
    rel = torch.where(e <= 20, e - 1, e - 21).clamp(min=0)
    return kind, torch.where(dino, rel % 4, 0), torch.where(dino, rel // 4, 0)


def combine(kind: torch.Tensor, dir_: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
    """log-probs [B,4|4|5,H,W] -> 42 entity log-probs [B,42,H,W] (id = base + level*4 + dir)"""
    k, d, l = kind.log_softmax(1), dir_.log_softmax(1), level.log_softmax(1)
    dl = (l[:, :, None] + d[:, None]).flatten(1, 2)   # [B, 20, H, W], index level*4 + dir
    return torch.cat([k[:, :1], k[:, 1:2] + dl, k[:, 2:3] + dl, k[:, 3:4]], 1)


def _group_matrix() -> torch.Tensor:
    """[42, 13] 0/1 matrix: which kind (4), facing (4) and level (5) group each entity id belongs to"""
    g = torch.zeros(N_ENTITY, N_KIND + N_DIR + N_LEVEL)
    k, d, l = split_ids(torch.arange(N_ENTITY))
    g[torch.arange(N_ENTITY), k] = 1
    dino = (k == 1) | (k == 2)
    g[torch.arange(N_ENTITY)[dino], N_KIND + d[dino]] = 1
    g[torch.arange(N_ENTITY)[dino], N_KIND + N_DIR + l[dino]] = 1
    return g


GROUPS = _group_matrix()


def group_entity(e: torch.Tensor):
    """42 scores -> (kind 4, dir 4, level 5) log-probabilities. Each group's probability is the sum
    of its members', e.g. "player" = all 20 player classes together. One matrix multiply (fast on MPS)."""
    p = torch.einsum("bchw,cg->bghw", e.softmax(1), GROUPS.to(e.device))
    lp = p.clamp(min=1e-12).log()
    return lp[:, :N_KIND], lp[:, N_KIND:N_KIND + N_DIR], lp[:, N_KIND + N_DIR:]


def build_model(c: dict) -> "CNN":
    return CNN(c["k"], c["width"], c["depth"], copy_prior=c.get("copy_prior", False),
               split_entity=c.get("split_entity", False), unet=c.get("unet", False),
               fates=c.get("fates", False))


def encode_input(context: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """context: int [B, k, 2, 45, 45], keys: int [B] -> float [B, k*50 + 5, 45, 45]"""
    B, k, _, H, W = context.shape
    terrain = F.one_hot(context[:, :, 0], N_TERRAIN)          # [B, k, H, W, 8]
    entity = F.one_hot(context[:, :, 1], N_ENTITY)            # [B, k, H, W, 42]
    frames = torch.cat([terrain, entity], dim=-1)             # [B, k, H, W, 50]
    frames = frames.permute(0, 1, 4, 2, 3).reshape(B, k * 50, H, W)
    key = F.one_hot(keys, N_KEYS)[:, :, None, None].expand(B, N_KEYS, H, W)  # copied into every cell
    return torch.cat([frames, key], dim=1).float()


class CNN(nn.Module):
    def __init__(self, k: int = 2, width: int = 64, depth: int = 6, copy_prior: bool = False,
                 split_entity: bool = False, unet: bool = False, fates: bool = False):
        super().__init__()
        self.k = k
        # copy prior (H10): add a learned bonus to the class each cell had in the last frame,
        # so "nothing changes" is the default and the network only has to learn changes
        self.copy_prior = copy_prior
        if copy_prior:
            self.copy_strength = nn.Parameter(torch.tensor([5.0, 5.0]))
        in_ch = k * (N_TERRAIN + N_ENTITY) + N_KEYS
        layers: list[nn.Module] = [nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU()]
        self.body = nn.Sequential(*layers)
        # 1x1 convs: per-cell readout of the features into class scores
        self.terrain_head = nn.Conv2d(width, N_TERRAIN, 1)
        self.split_entity = split_entity
        # split mode reuses the same 42 scores, but groups them into the three questions (see
        # group_entity), so a warm start from a flat-head model begins with the same predictions
        self.entity_head = nn.Conv2d(width, N_ENTITY, 1)
        # H13: per entity, "what happens to you?" (stay/up/right/down/left/die/gone), read at the
        # cells that hold an entity in the last frame (see tlfp/moves.py)
        self.fates = fates
        if fates:
            self.fate_head = nn.Conv2d(width, 7, 1)
            nn.init.zeros_(self.fate_head.weight); nn.init.zeros_(self.fate_head.bias)
        self.unet = unet
        if unet:  # 45 -> 23 -> 12 and back; each 3x3 at 12x12 sees ~1/4 of the screen
            self.down = nn.Sequential(
                nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
            self.up = nn.Sequential(nn.Conv2d(2 * width, width, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(width, width, 1))
            nn.init.zeros_(self.up[-1].weight); nn.init.zeros_(self.up[-1].bias)

    def forward(self, context: torch.Tensor, keys: torch.Tensor):
        t, e = self.forward_parts(context, keys)[:2]
        if self.split_entity:
            e = combine(*e)
        return t, e

    def forward_parts(self, context: torch.Tensor, keys: torch.Tensor):
        """terrain logits, and entity logits (42) or (kind, dir, level) when split"""
        h = self.body(encode_input(context, keys))
        if self.unet:
            g = F.interpolate(self.down(h), size=h.shape[-2:], mode="nearest")
            h = h + self.up(torch.cat([h, g], 1))
        last = context[:, -1]
        oh = lambda x, n: F.one_hot(x, n).permute(0, 3, 1, 2)
        t = self.terrain_head(h)
        if self.copy_prior:
            t = t + self.copy_strength[0] * oh(last[:, 0], N_TERRAIN)
        e = self.entity_head(h)
        if self.copy_prior:
            e = e + self.copy_strength[1] * oh(last[:, 1], N_ENTITY)
        e = group_entity(e) if self.split_entity else e
        return (t, e, self.fate_head(h)) if self.fates else (t, e)

    @torch.no_grad()
    def predict(self, context: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Most likely next frame: int [B, 2, 45, 45]"""
        t, e = self(context, keys)
        return torch.stack([t.argmax(1), e.argmax(1)], dim=1)
