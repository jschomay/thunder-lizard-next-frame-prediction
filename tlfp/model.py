"""Approach A: a plain CNN that predicts the next state grid.

Input per cell: k frames x (8 terrain + 42 entity one-hot) + 5 key one-hot (same in every cell).
Output per cell: 8 terrain scores and 42 entity scores (logits) for the next frame.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tlfp.data import N_ENTITY, N_TERRAIN

N_KEYS = 5


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
    def __init__(self, k: int = 2, width: int = 64, depth: int = 6, copy_prior: bool = False):
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
        self.entity_head = nn.Conv2d(width, N_ENTITY, 1)

    def forward(self, context: torch.Tensor, keys: torch.Tensor):
        h = self.body(encode_input(context, keys))
        t, e = self.terrain_head(h), self.entity_head(h)
        if self.copy_prior:
            last = context[:, -1]
            t = t + self.copy_strength[0] * F.one_hot(last[:, 0], N_TERRAIN).permute(0, 3, 1, 2)
            e = e + self.copy_strength[1] * F.one_hot(last[:, 1], N_ENTITY).permute(0, 3, 1, 2)
        return t, e

    @torch.no_grad()
    def predict(self, context: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Most likely next frame: int [B, 2, 45, 45]"""
        t, e = self(context, keys)
        return torch.stack([t.argmax(1), e.argmax(1)], dim=1)
