"""Hand-written predictors that any learned model has to beat.

    uv run python -m tlfp.baselines data/stage1 data/human

copy_last     next frame = current frame. Nothing ever moves.
shift_by_key  if a key is held, scroll the view one cell, every tick. Knows the controls
              but not the every-other-tick rhythm (uses 1 frame of history).
shift_rhythm  same, but only moves if the player did NOT move on the previous tick,
              and also moves on a NONE tick right after a press (the buffered key).
              Uses 2 frames of history: a hand-written preview of H3.
"""

from __future__ import annotations

import sys

import numpy as np

from tlfp.data import find_episodes, split_by_episode
from tlfp.metrics import add, frame_metrics, summarize

# key index -> (dy, dx) the player moves; the view content shifts the opposite way
DIRS = {1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1)}


def _player_mask(ent):
    return (ent >= 1) & (ent <= 20)


def scroll(frame: np.ndarray, key: int) -> np.ndarray:
    """Predict the frame after the player steps in direction `key` (player stays centered)."""
    dy, dx = DIRS[key]
    out = frame.copy()
    terrain, ent = frame[0], frame[1]
    pm = _player_mask(ent)
    # content moves opposite to the player; the newly revealed edge repeats the old edge (unknown)
    sh = lambda a: np.roll(a, (-dy, -dx), axis=(0, 1))
    t2 = sh(terrain)
    if dy == -1: t2[0] = terrain[0]
    if dy == 1: t2[-1] = terrain[-1]
    if dx == -1: t2[:, 0] = terrain[:, 0]
    if dx == 1: t2[:, -1] = terrain[:, -1]
    others = np.where(pm, 0, ent)
    e2 = sh(others)
    if pm.any():
        y, x = np.argwhere(pm)[0]
        level = (ent[y, x] - 1) // 4 + 1
        e2[y, x] = 1 + (level - 1) * 4 + (key - 1)  # player stays put, now facing the key direction
    out[0], out[1] = t2, e2
    return out


def moved(a: np.ndarray, b: np.ndarray) -> bool:
    return not np.array_equal(a[0], b[0])


def predict(name: str, obs: np.ndarray, actions: np.ndarray, t: int) -> np.ndarray:
    cur = obs[t]
    key = int(actions[t])
    if name == "copy_last":
        return cur
    if name == "shift_by_key":
        return scroll(cur, key) if key else cur
    if name == "shift_rhythm":
        prev_moved = t > 0 and moved(obs[t - 1], cur)
        pending = key or (t > 0 and int(actions[t - 1]) and not prev_moved and int(actions[t - 1]))
        if pending and not prev_moved:
            return scroll(cur, key or int(actions[t - 1]))
        return cur
    raise ValueError(name)


def evaluate(name: str, episodes) -> dict[str, float]:
    totals: dict[str, int] = {}
    for ep in episodes:
        obs = np.asarray(ep.obs)
        prev = obs[:-1]
        pred = np.stack([predict(name, obs, ep.actions, t) for t in range(len(ep))])
        add(totals, frame_metrics(prev, pred, obs[1:]))
    return summarize(totals)


def main(dirs: list[str]):
    names = ["copy_last", "shift_by_key", "shift_rhythm"]
    for d in dirs:
        eps = find_episodes(d)
        if d != "data/human":
            _, eps = split_by_episode(eps, 0.1)  # validation episodes only
        print(f"\n{d}  ({len(eps)} episodes, {sum(len(e) for e in eps)} ticks)")
        print(f"{'':14s} {'cell':>7s} {'dynamic':>8s} {'changed':>8s} {'terr.chg':>9s} {'ent.chg':>8s} {'exact frame':>12s}")
        for n in names:
            m = evaluate(n, eps)
            print(f"{n:14s} {m['cell_acc']:7.2%} {m['dynamic_acc']:8.2%} {m['changed_acc']:8.2%} {m['terrain_changed_acc']:9.2%} "
                  f"{m['entity_changed_acc']:8.2%} {m['exact_frame_acc']:12.2%}")
        print(f"  ({m['frac_frames_changed']:.0%} of ticks change anything)")


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/stage1"])
