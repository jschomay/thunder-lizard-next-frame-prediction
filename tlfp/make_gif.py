"""Render now / real next / predicted next side by side as an animated GIF (and MP4 if ffmpeg exists).

    uv run python -m tlfp.make_gif --start 0 --ticks 120 --fps 6

Reads .whiteboard/frame-prediction/predictions.json (from tlfp.export_predictions).
White outlines = cells that really changed; red outlines = cells the model got wrong.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

B36 = "0123456789abcdefghijklmnopqrstuvwxyz!@#$%^"
TERRAIN_COLORS = ["#16206e", "#4a55e6", "#7a5a40", "#2aa52a", "#1a7a2a", "#0b4d0b", "#e3341f", "#3a2a22"]
ARROW = {"UP": "▲", "RIGHT": "▶", "DOWN": "▼", "LEFT": "◀", "NONE": "·"}
N = 45


def font(size):
    for f in ["/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial Unicode.ttf"]:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def draw_grid(d: ImageDraw.ImageDraw, ox, oy, frame, cell, white=(), red=()):
    terrain, entity = frame
    for i in range(N * N):
        x, y = ox + (i % N) * cell, oy + (i // N) * cell
        d.rectangle([x, y, x + cell - 1, y + cell - 1], fill=TERRAIN_COLORS[int(terrain[i])])
    for i in range(N * N):
        v = B36.index(entity[i])
        if not v:
            continue
        cx, cy = ox + (i % N) * cell + cell / 2, oy + (i // N) * cell + cell / 2
        color = "#dddddd" if v == 41 else "#ffe14d" if v <= 20 else "#ff4040"
        r = cell * 0.8
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline="#000")
    for cells, color, w in [(white, "#ffffff", 1), (red, "#ff1f2e", 2)]:
        for i in cells:
            x, y = ox + (i % N) * cell, oy + (i // N) * cell
            d.rectangle([x + 1, y + 1, x + cell - 2, y + cell - 2], outline=color, width=w)


def diff(a, b):
    return [i for i in range(N * N) if a[0][i] != b[0][i] or a[1][i] != b[1][i]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--ticks", type=int, default=120)
    p.add_argument("--fps", type=float, default=6)
    p.add_argument("--cell", type=int, default=8)
    p.add_argument("--out", default="media/predictions")
    args = p.parse_args()

    data = json.loads(Path(".whiteboard/frame-prediction/predictions.json").read_text())
    ep = data["episodes"][args.episode]
    c, gap, top, bottom = args.cell, 24, 64, 56
    W = 3 * N * c + 4 * gap
    H = top + N * c + bottom
    f_title, f_label, f_small = font(22), font(18), font(15)
    frames = []
    end = min(len(ep["actions"]), args.start + args.ticks)
    for t in range(args.start, end):
        img = Image.new("RGB", (W, H), "#111111")
        d = ImageDraw.Draw(img)
        now, truth, pred = ep["prev"][t], ep["truth"][t], ep["pred"][t]
        real_changed, wrong = diff(now, truth), diff(pred, truth)
        panels = [("now", now, (), ()), ("real next frame", truth, real_changed, ()),
                  ("model's prediction", pred, (), wrong)]
        for k, (label, fr, white, red) in enumerate(panels):
            ox = gap + k * (N * c + gap)
            d.text((ox, 22), label, fill="#f0f0f0", font=f_label)
            draw_grid(d, ox, top, fr, c, white, red)
        key = ep["actions"][t]
        y = top + N * c + 14
        d.text((gap, y), f"key held: {key} {ARROW[key]}", fill="#ffe14d", font=f_label)
        status = "✓ exactly right" if not wrong else f"{len(wrong)} wrong cells"
        d.text((gap + N * c + gap, y), f"□ white = cells that changed ({len(real_changed)})", fill="#ffffff", font=f_small)
        d.text((gap + 2 * (N * c + gap), y), f"□ red = model errors · {status}", fill="#ff6b74", font=f_small)
        d.text((W - 210, 22), f"tick {t}", fill="#888", font=f_small)
        frames.append(img)

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    frames[0].save(out.with_suffix(".gif"), save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {out.with_suffix('.gif')} ({len(frames)} frames, {out.with_suffix('.gif').stat().st_size // 1024} KB)")
    if shutil.which("ffmpeg"):  # MP4 is often smaller and sharper on LinkedIn
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out.with_suffix(".gif")),
                        "-movflags", "faststart", "-pix_fmt", "yuv420p",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(out.with_suffix(".mp4"))], check=True)
        print(f"wrote {out.with_suffix('.mp4')}")


if __name__ == "__main__":
    main()
