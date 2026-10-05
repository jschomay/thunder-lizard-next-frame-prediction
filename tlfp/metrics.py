"""Scoring next-frame predictions.

Most cells don't change between ticks, so plain accuracy flatters any model that
just copies the last frame. The metrics that matter look at what *changed*.
"""

from __future__ import annotations

import numpy as np


def frame_metrics(prev: np.ndarray, pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compare predicted vs true next frames.

    All arrays are uint8 [N, 2, 45, 45]: N examples, layer 0 terrain, layer 1 entity.
    `prev` is the last context frame, used to find which cells changed.
    Returns sums and counts so results from many batches can be added up; use `summarize`.
    """
    correct = pred == target
    changed = prev != target
    # cells where either reality or the prediction changed: penalizes invented changes too
    dynamic = changed | (pred != prev)
    ent_t, ent_p = target[:, 1], pred[:, 1]
    occupied = ent_t != 0
    # exact frame: every cell of both layers right
    exact = correct.reshape(len(target), -1).all(axis=1)
    return {
        "cells": correct.size, "cells_correct": int(correct.sum()),
        "dynamic": int(dynamic.sum()), "dynamic_correct": int((correct & dynamic).sum()),
        "changed": int(changed.sum()), "changed_correct": int((correct & changed).sum()),
        "terrain_changed": int(changed[:, 0].sum()), "terrain_changed_correct": int((correct & changed)[:, 0].sum()),
        "entity_changed": int(changed[:, 1].sum()), "entity_changed_correct": int((correct & changed)[:, 1].sum()),
        "entity_occupied": int(occupied.sum()), "entity_occupied_correct": int(((ent_t == ent_p) & occupied).sum()),
        "frames": len(target), "frames_exact": int(exact.sum()),
        "frames_changed": int(changed.reshape(len(target), -1).any(axis=1).sum()),
    }


def summarize(totals: dict[str, int]) -> dict[str, float]:
    r = lambda a, b: totals[a] / totals[b] if totals[b] else float("nan")
    return {
        "cell_acc": r("cells_correct", "cells"),
        "dynamic_acc": r("dynamic_correct", "dynamic"),
        "changed_acc": r("changed_correct", "changed"),
        "terrain_changed_acc": r("terrain_changed_correct", "terrain_changed"),
        "entity_changed_acc": r("entity_changed_correct", "entity_changed"),
        "occupied_entity_acc": r("entity_occupied_correct", "entity_occupied"),
        "exact_frame_acc": r("frames_exact", "frames"),
        "frac_frames_changed": r("frames_changed", "frames"),
    }


def add(totals: dict[str, int], batch: dict[str, int]) -> dict[str, int]:
    for k, v in batch.items():
        totals[k] = totals.get(k, 0) + v
    return totals
