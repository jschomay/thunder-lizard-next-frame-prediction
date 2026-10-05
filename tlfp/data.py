"""Loading recorded episodes into training examples.

An episode on disk (written by harness/record.ts or the in-game recorder):
    obs.npy      uint8 [T+1, 2, 45, 45]   layer 0 = terrain class, layer 1 = entity class
    actions.npy  uint8 [T]                action[t] takes obs[t] -> obs[t+1]

A training example is a sliding window over an episode:
    context  [k, 2, 45, 45]   the k frames up to and including time t
    action   int              key held during tick t
    target   [2, 45, 45]      the frame at t+1 (what the model must predict)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

KEYS = ["NONE", "UP", "RIGHT", "DOWN", "LEFT"]
TERRAIN = ["OCEAN", "WATER", "DIRT", "GRASS", "SHRUB", "JUNGLE", "LAVA", "BURNT"]
N_TERRAIN = len(TERRAIN)
N_ENTITY = 42  # 0 empty, 1-20 player (level x dir), 21-40 npc dino, 41 skeleton
VIEW = 45


@dataclass
class Episode:
    path: Path
    obs: np.ndarray      # [T+1, 2, 45, 45] uint8 (memory-mapped, not loaded until used)
    actions: np.ndarray  # [T] uint8
    meta: dict

    @classmethod
    def load(cls, path: str | Path) -> "Episode":
        path = Path(path)
        return cls(
            path=path,
            obs=np.load(path / "obs.npy", mmap_mode="r"),
            actions=np.load(path / "actions.npy"),
            meta=json.loads((path / "meta.json").read_text()),
        )

    def __len__(self) -> int:
        return len(self.actions)


def find_episodes(*dirs: str | Path) -> list[Episode]:
    """All episodes under the given dirs, e.g. find_episodes("data/stage1", "data/human")."""
    eps = []
    for d in dirs:
        eps += [Episode.load(p.parent) for p in sorted(Path(d).glob("**/obs.npy"))]
    return eps


class FrameDataset(Dataset):
    """Every (k frames, action) -> next frame window across a list of episodes.

    The first k-1 ticks of an episode have too little history; they're padded by
    repeating the first frame, so short human sessions still contribute fully.
    """

    def __init__(self, episodes: list[Episode], k: int = 2):
        self.episodes = episodes
        self.k = k
        # flat index -> (episode, t)
        self.index = [(e, t) for e, ep in enumerate(episodes) for t in range(len(ep))]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        e, t = self.index[i]
        ep = self.episodes[e]
        frames = [ep.obs[max(0, s)] for s in range(t - self.k + 1, t + 1)]
        return (
            torch.from_numpy(np.stack(frames).astype(np.int64)),  # [k, 2, 45, 45]
            int(ep.actions[t]),
            torch.from_numpy(ep.obs[t + 1].astype(np.int64)),    # [2, 45, 45]
        )


def split_by_episode(episodes: list[Episode], val_fraction: float = 0.1, seed: int = 0):
    """Hold out whole episodes (not random ticks) so validation frames are truly unseen."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    n_val = max(1, int(len(episodes) * val_fraction))
    val = [episodes[i] for i in order[:n_val]]
    train = [episodes[i] for i in order[n_val:]]
    return train, val
