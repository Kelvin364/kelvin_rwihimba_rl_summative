"""Render REPORT.md to a print-ready PDF.

No pandoc/LaTeX needed: Markdown → HTML (python-markdown) → PDF (Chrome's
``--print-to-pdf``). Figures are embedded as base64 data URIs so the intermediate
HTML is self-contained and Chrome needs no file access beyond the one page.

    uv run python scripts/build_report_pdf.py
    uv run python scripts/build_report_pdf.py --fig-height 62   # tune page count

I target 7-10 pages; ``--fig-height`` (max figure height in mm) is the lever I use to
land there, since the nine figures dominate the layout.
"""

from __future__ import annotations

import argparse
import base64
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

_CSS = """
@page { size: A4; margin: 16mm 15mm 14mm 15mm; }
* { box-sizing: border-box; }
body { font: 9.6pt/1.42 "Helvetica Neue", Helvetica, Arial, sans-serif; color:#111;
       margin:0; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
h1 { font-size: 19pt; margin: 0 0 2mm; letter-spacing:-.4pt; }
h2 { font-size: 12.5pt; margin: 5mm 0 1.6mm; padding-bottom: 1mm;
     border-bottom: 0.5pt solid #d8d6ce; break-after: avoid; }
h3 { font-size: 10.4pt; margin: 3.4mm 0 1.2mm; break-after: avoid; }
p  { margin: 0 0 1.9mm; orphans: 3; widows: 3; }
ul, ol { margin: 0 0 2mm; padding-left: 5mm; }
li { margin-bottom: 0.8mm; }
strong { font-weight: 650; }
code { font: 8.4pt ui-monospace, "SF Mono", Menlo, monospace;
       background:#f2f1ed; padding: 0.3mm 0.8mm; border-radius: 2px; }
pre { background:#f7f6f3; border:0.5pt solid #e2e0d8; border-radius: 3px;
      padding: 2mm 2.6mm; margin: 0 0 2.4mm; overflow: hidden; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8pt; line-height: 1.35; }
table { border-collapse: collapse; width: 100%; margin: 0 0 2.6mm;
        font-size: 8.5pt; break-inside: avoid; }
th, td { border: 0.5pt solid #dcdad2; padding: 1mm 1.6mm; text-align: left;
         vertical-align: top; }
th { background: #f2f1ed; font-weight: 650; }
tbody tr:nth-child(even) { background: #faf9f6; }
img { display:block; max-width: 100%; max-height: __FIGH__mm; margin: 1.5mm auto 1mm;
      break-inside: avoid; }
em { color:#4a4945; }
/* Figure captions are emphasised lines directly after an image. */
p > img + em, p > em:only-child { display:block; text-align:center; font-size:8.2pt;
      color:#5d5c57; margin-top:-0.5mm; }
hr { border:0; border-top:0.5pt solid #dcdad2; margin: 4mm 0; }
blockquote { margin:0 0 2mm; padding-left:3mm; border-left:1.5pt solid #dcdad2;
             color:#4a4945; }
a { color:#1c5cab; text-decoration:none; }
"""


def _embed_images(html: str) -> str:
    """Inline every local <img src> as a data URI."""
    def repl(m: re.Match) -> str:
        src = m.group(1)
        if src.startswith(("http:", "https:", "data:")):
            return m.group(0)
        path = (_REPO_ROOT / src).resolve()
        if not path.exists():
            print(f"  WARNING: missing image {src}", file=sys.stderr)
            return m.group(0)
        b64 = base64.b64encode(path.read_bytes()).decode()
        return m.group(0).replace(src, f"data:image/png;base64,{b64}")

    return re.sub(r'<img[^>]+src="([^"]+)"', repl, html)


def build(md_path: Path, out_pdf: Path, fig_height: int) -> Path:
    import markdown

    body = markdown.markdown(
        md_path.read_text(),
        extensions=["tables", "fenced_code", "attr_list", "sane_lists"],
    )
    body = _embed_images(body)
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{_CSS.replace('__FIGH__', str(fig_height))}</style></head>"
            f"<body>{body}</body></html>")

    tmp = out_pdf.with_suffix(".build.html")
    tmp.write_text(html)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={out_pdf}", f"file://{tmp}"],
        capture_output=True, timeout=180, check=False,
    )
    tmp.unlink(missing_ok=True)
    if not out_pdf.exists():
        raise SystemExit("Chrome did not produce a PDF")

    pages = out_pdf.read_bytes().count(b"/Type /Page") or \
        out_pdf.read_bytes().count(b"/Type/Page")
    print(f"wrote {out_pdf}  ({out_pdf.stat().st_size/1e6:.2f} MB, ~{pages} pages)")
    if not 7 <= pages <= 10:
        print(f"  NOTE: target is 7-10 pages; got ~{pages}. "
              f"Adjust with --fig-height (currently {fig_height}mm).")
    return out_pdf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--md", default="REPORT.md")
    ap.add_argument("--out", default="assets/AgriScout_Report.pdf")
    ap.add_argument("--fig-height", type=int, default=56,
                    help="max figure height in mm — the main page-count lever")
    args = ap.parse_args()
    build(_REPO_ROOT / args.md, _REPO_ROOT / args.out, args.fig_height)


if __name__ == "__main__":
    main()
