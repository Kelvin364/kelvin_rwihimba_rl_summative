"""Record the WebGL demo view to an animated GIF -- headless, no ffmpeg.

Captures the ACTUAL three.js scene the HTML viewer shows (lighting, shadows, the
treatment rings), rather than re-rendering the episode somewhere else, so the clip
and the interactive demo are the same picture.

How: headless Chrome loads the generated viewer with a capture harness injected.
The harness steps the episode, waits for the rover's motion lerp to settle, reads
each frame back with ``canvas.toDataURL`` and appends it to the DOM; Chrome is run
with ``--dump-dom``, and this script decodes the frames, burns in a caption and
writes the GIF. (``--screenshot`` cannot be used: it needs one browser launch per
frame, which is minutes of process startup for a 150-step episode.)

    uv run python scripts/record_video_web.py --episode ppo
    uv run python scripts/record_video_web.py --episode oracle --stride 2

Requires ``assets/demo/index.html`` to exist (build it with make_demo_html.py).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
KIND_RGB = {
    "irrigate": (58, 135, 229), "spray": (233, 104, 52),
    "depot": (27, 175, 122), "move": (150, 154, 160), "idle": (120, 122, 126),
}

_HARNESS = """
<script>
(function(){
  const EP = __EP__, STRIDE = __STRIDE__, SETTLE = __SETTLE__;
  // Everything here is SYNCHRONOUS apart from one setTimeout: headless Chrome under
  // --virtual-time-budget never fires requestAnimationFrame, so frames are produced
  // by calling Scene.renderNow() directly rather than awaiting the rAF loop.
  setTimeout(function(){
    // Recording layout: light theme reads far better than the dark surface once the
    // clip is embedded in a doc, and giving the field card the full column width
    // roughly doubles the captured canvas resolution.
    document.documentElement.setAttribute('data-theme', '__THEME__');
    document.querySelector('.grid2').style.gridTemplateColumns = '1fr';
    document.querySelectorAll('.grid2 > div').forEach(n => n.remove());
    document.querySelector('.controls').style.display = 'none';
    document.querySelector('#view3d').style.maxWidth = '__CW__px';
    ep = EP; snap3D = true; mountEpisode();
    Scene.setAuto(false);                    // steady camera for a clean recording
    Scene.setPixelRatio(1);                  // see Scene.setPixelRatio -- DPR 2 is 4x the work
    const cv = document.querySelector('#view3d canvas');
    const out = [];
    const E = episodes[EP];
    const t0 = Date.now();
    for (let i = 0; i < E.frames.length; i += STRIDE) {
      idx = i; snap3D = (i === 0); render();
      for (let k = 0; k < SETTLE; k++) Scene.renderNow();  // settle lerp + effects
      out.push(cv.toDataURL('image/jpeg', 0.88));
    }
    const d = document.createElement('div');
    d.id = 'capout'; d.textContent = out.join('|');
    document.body.appendChild(d);
    const s = document.createElement('div');
    s.id = 'capstat';
    s.textContent = cv.width + 'x' + cv.height + ' in ' + (Date.now() - t0) + 'ms';
    document.body.appendChild(s);
    document.title = 'CAPTURE_DONE';
  }, 300);
})();
</script>
"""


def _font(size: int):
    for path in _FONTS:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


def capture(html: Path, ep_index: int, stride: int, settle: int,
            theme: str = "light", win: str = "1240,840",
            canvas_w: int = 900) -> list[Image.Image]:
    page = html.read_text()
    harness = (_HARNESS.replace("__EP__", str(ep_index))
                       .replace("__STRIDE__", str(stride))
                       .replace("__SETTLE__", str(settle))
                       .replace("__THEME__", theme)
                       .replace("__CW__", str(canvas_w)))
    page = page.replace("</body>", harness + "</body>")

    with tempfile.TemporaryDirectory() as td:
        shot = Path(td) / "capture.html"
        shot.write_text(page)
        print("  driving headless Chrome (software GL, this takes a minute) ...",
              flush=True)
        proc = subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", "--use-gl=angle",
             "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
             "--hide-scrollbars", f"--window-size={win}",
             "--virtual-time-budget=600000", "--dump-dom", f"file://{shot}"],
            capture_output=True, text=True, timeout=780,
        )
    dom = proc.stdout
    m = re.search(r'<div id="capout">(.*?)</div>', dom, re.S)
    if not m:
        raise SystemExit("capture failed: no frames in the dumped DOM "
                         f"(chrome stderr tail: {proc.stderr[-400:]})")
    frames = [f for f in m.group(1).split("|") if f.startswith("data:image")]
    stat = re.search(r'<div id="capstat">(.*?)</div>', dom, re.S)
    print(f"  captured {len(frames)} frames"
          + (f" ({stat.group(1)})" if stat else ""))
    return [Image.open(io.BytesIO(base64.b64decode(f.split(",", 1)[1]))).convert("RGB")
            for f in frames]


def caption(img: Image.Image, f: dict, meta: dict, i: int, total: int) -> Image.Image:
    w, h = img.size
    bar = int(h * 0.19)
    out = Image.new("RGB", (w, h + bar), (16, 16, 15))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    big, small = _font(int(h * 0.045)), _font(int(h * 0.032))
    x, y = int(w * 0.03), h + int(bar * 0.16)

    r = max(3, int(h * 0.011))
    d.ellipse([x, y + r, x + 2 * r, y + 3 * r], fill=KIND_RGB.get(f["kind"], (150, 150, 150)))
    label = f["reason"] + ("   [wasted]" if f["wasted"] else "")
    d.text((x + 3 * r, y), label, font=big, fill=(255, 255, 255))
    d.text((x + 3 * r, y + int(bar * 0.40)), f["detail"], font=small, fill=(190, 189, 181))

    title = f"AgriScout — {meta.get('model','?')} agent · seed {meta.get('seed','?')}"
    stat = (f"step {f['t']}/{total}   reward {f['r']:+.2f}   total {f['cum']:+.1f}   "
            f"health {np.mean(f['h']):.3f}")
    for text, dy, col in ((title, 0, (255, 255, 255)),
                          (stat, int(bar * 0.40), (190, 189, 181))):
        tw = d.textlength(text, font=small)
        d.text((w - tw - int(w * 0.03), y + dy), text, font=small, fill=col)

    d.rectangle([0, h + bar - 5, int(w * (i + 1) / total), h + bar], fill=(58, 135, 229))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--html", default="assets/demo/index.html")
    ap.add_argument("--episode", default="ppo", help="agent name in the viewer")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=1, help="capture every Nth step")
    ap.add_argument("--settle", type=int, default=3,
                    help="synchronous render passes per captured frame")
    ap.add_argument("--win", default="1240,840", help="headless window size")
    ap.add_argument("--canvas-w", type=int, default=900,
                    help="max canvas CSS width (software GL cost scales with pixels)")
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--theme", choices=["light", "dark"], default="light")
    args = ap.parse_args()

    html = Path(args.html)
    if not html.exists():
        raise SystemExit(f"{html} not found -- build it with scripts/make_demo_html.py")
    if not Path(CHROME).exists():
        raise SystemExit(f"Chrome not found at {CHROME}")

    blob = re.search(r"const DATA = (\{.*?\});", html.read_text(), re.S)
    data = json.loads(blob.group(1))
    models = [e["meta"].get("model") for e in data["episodes"]]
    if args.episode not in models:
        raise SystemExit(f"episode {args.episode!r} not in {models}")
    ep_index = models.index(args.episode)
    E = data["episodes"][ep_index]

    print(f"recording '{args.episode}' ({len(E['frames'])} steps, stride {args.stride})")
    shots = capture(html, ep_index, args.stride, args.settle, args.theme, args.win, args.canvas_w)

    picked = E["frames"][::args.stride][:len(shots)]
    total = len(E["frames"])
    framed = [caption(im, f, E["meta"], i * args.stride, total)
              for i, (im, f) in enumerate(zip(shots, picked))]

    out = Path(args.out) if args.out else Path("assets/demo") / f"agriscout_{args.episode}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    pal = framed[0].quantize(colors=200, method=Image.MEDIANCUT)
    quant = [im.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for im in framed]
    quant[0].save(out, save_all=True, append_images=quant[1:],
                  duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {out}  ({len(quant)} frames, {out.stat().st_size/1e6:.1f} MB, "
          f"{len(quant)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
