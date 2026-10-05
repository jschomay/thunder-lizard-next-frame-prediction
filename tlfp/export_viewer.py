"""Export a few recorded episodes as JSON for the whiteboard's Data tab.

    uv run python -m tlfp.export_viewer
"""

import json
from pathlib import Path

from tlfp.data import KEYS, find_episodes

OUT = Path(".whiteboard/frame-prediction/episodes.json")
MAX_TICKS = 150


def main():
    picks = []
    for source in ["data/human", "data/stage1", "data/stage2", "data/stage3", "data/stage4"]:
        eps = [e for e in find_episodes(source) if len(e) >= 20]
        # longest human sessions; first two bot episodes per stage
        eps = sorted(eps, key=len, reverse=True)[:3] if source == "data/human" else eps[:2]
        for ep in eps:
            n = min(len(ep), MAX_TICKS)
            picks.append({
                "name": f"{source.split('/')[-1]} / {ep.path.name}",
                "meta": ep.meta,
                "actions": [KEYS[a] for a in ep.actions[:n]],
                # one string per frame: 2025 terrain digits + 2025 entity values (base-36 to keep it 1 char)
                "terrain": ["".join(map(str, f[0].ravel())) for f in ep.obs[: n + 1]],
                "entity": ["".join(_b36(v) for v in f[1].ravel()) for f in ep.obs[: n + 1]],
            })
    OUT.write_text(json.dumps(picks))
    print(f"wrote {len(picks)} episodes -> {OUT} ({OUT.stat().st_size // 1024} KB)")


def _b36(v: int) -> str:
    return "0123456789abcdefghijklmnopqrstuvwxyz!@#$%^"[v]


if __name__ == "__main__":
    main()
