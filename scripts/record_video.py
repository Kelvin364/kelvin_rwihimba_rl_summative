"""Render a recorded episode to an animated GIF, headless, with no GUI and no ffmpeg.

Replays a trace's ACTION SEQUENCE through a freshly seeded environment. Because the
env is seeded and its dynamics are deterministic given the actions, the replay
reproduces the recorded episode exactly, so the video and the HTML viewer show the
same run.

Frames come from PyBullet's offscreen camera with a slow orbit, and each frame is
captioned with what the agent did, so the clip explains itself with the sound off.

    uv run python scripts/record_video.py --trace logs/traces/demo_ppo_seed9003.json
    uv run python scripts/record_video.py --trace ... --width 1000 --fps 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Must be set before environment.rendering imports pybullet.
os.environ.setdefault("AGRISCOUT_HEADLESS", "1")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

KIND_RGB = {
    "IRRIGATE": (58, 135, 229), "SPRAY": (233, 104, 52),
    "RETURN_TO_DEPOT": (27, 175, 122),
}


def _font(size: int):
    for path in _FONTS:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


def _caption(img: Image.Image, frame: dict, note: dict, meta: dict, i: int, n: int) -> None:
    """Burn a legible caption bar onto the frame."""
    w, h = img.size
    bar_h = int(h * 0.17)
    overlay = Image.new("RGBA", (w, bar_h), (12, 12, 11, 214))
    img.paste(overlay, (0, h - bar_h), overlay)
    d = ImageDraw.Draw(img)

    big, small = _font(int(h * 0.036)), _font(int(h * 0.026))
    x, y = int(w * 0.028), h - bar_h + int(bar_h * 0.16)

    dot = KIND_RGB.get(frame["action"], (150, 154, 160))
    r = int(h * 0.010)
    d.ellipse([x, y + r // 2, x + 2 * r, y + 2 * r + r // 2], fill=dot)
    d.text((x + 3 * r, y), note["reason"], font=big, fill=(255, 255, 255))
    d.text((x + 3 * r, y + int(bar_h * 0.36)), note["detail"], font=small,
           fill=(196, 195, 186))

    rw = f"{frame['reward']:+.2f}"
    tot = f"{frame['cum_reward']:+.1f}"
    right = (f"step {frame['t']}/{n}    reward {rw}    total {tot}    "
             f"health {np.mean(frame['health_grid']):.3f}")
    tw = d.textlength(right, font=small)
    d.text((w - tw - int(w * 0.028), y + int(bar_h * 0.36)), right, font=small,
           fill=(196, 195, 186))
    title = f"AgriScout: {meta.get('model', '?')} agent, seed {meta.get('seed', '?')}"
    tw2 = d.textlength(title, font=small)
    d.text((w - tw2 - int(w * 0.028), y), title, font=small, fill=(255, 255, 255))

    # Progress bar along the very bottom.
    d.rectangle([0, h - 4, int(w * (i + 1) / n), h], fill=(58, 135, 229))


def record(trace_path: Path, out: Path, width: int, fps: int, tween: int) -> Path:
    from environment.agriscout_env import AgriScoutEnv
    from environment.rendering import CELL, AgriScoutRenderer
    import pybullet as p

    from scripts.make_demo_html import annotate

    payload = json.loads(trace_path.read_text())
    frames, meta = payload["frames"], payload["meta"]
    notes = annotate(frames)

    env = AgriScoutEnv()
    env.reset(seed=int(meta["seed"]))
    renderer = AgriScoutRenderer(env, mode="direct")

    names = list(env.ACTION_NAMES)
    height = int(width * 9 / 16)
    target = [((env.n_cols - 1) / 2 + 1) * CELL, ((env.n_rows - 1) / 2 + 1) * CELL, 0.0]
    dist = max(env.n_rows, env.n_cols) * CELL * 1.05
    proj = p.computeProjectionMatrixFOV(52, width / height, 0.1, 100)

    images: list[Image.Image] = []
    n = len(frames)
    for i, (fr, note) in enumerate(zip(frames, notes)):
        env.step(names.index(fr["action"]))
        # Sub-frames interpolate the rover so motion reads as driving, not jumping.
        for k in range(1, tween + 1):
            renderer.render(k / tween)
            yaw = 38 + 30 * np.sin((i + k / tween) / n * 2 * np.pi)
            view = p.computeViewMatrixFromYawPitchRoll(target, dist, yaw, -38, 0, 2)
            raw = p.getCameraImage(width, height, view, proj,
                                   renderer=p.ER_TINY_RENDERER)
            rgb = np.reshape(raw[2], (height, width, 4))[:, :, :3].astype(np.uint8)
            img = Image.fromarray(rgb).convert("RGB")
            _caption(img, fr, note, meta, i, n)
            images.append(img)
        if (i + 1) % 40 == 0:
            print(f"  ...{i + 1}/{n} steps", flush=True)
    renderer.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    # Quantize to a shared adaptive palette, which keeps the GIF small without the
    # frame-to-frame colour churn that per-frame palettes cause.
    pal = images[0].quantize(colors=192, method=Image.MEDIANCUT)
    quant = [im.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for im in images]
    quant[0].save(out, save_all=True, append_images=quant[1:],
                  duration=int(1000 / fps), loop=0, optimize=True)
    mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({len(quant)} frames, {mb:.1f} MB, {len(quant)/fps:.1f}s)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--tween", type=int, default=2,
                    help="rendered sub-frames per env step (motion smoothness)")
    args = ap.parse_args()

    trace = Path(args.trace)
    out = Path(args.out) if args.out else Path("assets/demo") / f"{trace.stem}.gif"
    print(f"rendering {trace.name} -> {out}")
    record(trace, out, args.width, args.fps, args.tween)


if __name__ == "__main__":
    main()
