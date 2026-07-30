"""Turn a recorded episode trace into a self-contained, self-explanatory HTML demo.

The design goal is legibility without narration: a viewer should be able to open the
page and see WHAT the rover just did, WHERE, and WHETHER it helped, without anyone
talking them through it. Every step therefore carries a derived plain-English reason
(computed here, from the state delta between consecutive frames) rather than only an
action name.

    uv run python scripts/record_episode.py --all --seed 7
    uv run python scripts/make_demo_html.py --traces logs/traces/demo_*_seed7.json

Writes ``assets/demo/index.html`` -- one file, no external requests.

I chose the colours by measuring colour-blind separation rather than by eye, because
my first two attempts both failed that check:
  * Action identity uses blue / orange / aqua. My first pick was blue / orange /
    violet, but violet and blue are almost indistinguishable to a protan viewer on a
    dark background, so I replaced violet with aqua.
  * Crop health uses ONE ramp, dry-soil neutral -> deep green, so the signal is
    lightness and saturation rather than two opposed hues. I first tried an
    amber -> green ramp and dropped it: that is the classic red-green trap.
  * Since hue alone should never carry the meaning, health is encoded three ways at
    once -- ramp colour, canopy SIZE, and a dashed outline on any cell below the
    success threshold. The last two still read in greyscale.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from environment.agriscout_env import (  # noqa: E402
    NAV_PEST_THRESHOLD,
    SUCCESS_HEALTH,
    SUCCESS_PEST,
)

# Action -> semantic kind. Only the three kinds that CHANGE the world get an
# identity colour; movement and idling stay in neutral ink so the treatments pop.
ACTION_KIND = {
    "IRRIGATE": "irrigate",
    "SPRAY": "spray",
    "RETURN_TO_DEPOT": "depot",
    "MOVE_N": "move", "MOVE_S": "move", "MOVE_E": "move", "MOVE_W": "move",
    "SCAN": "idle", "WAIT": "idle",
}
DIRECTION = {"MOVE_N": "north", "MOVE_S": "south", "MOVE_E": "east", "MOVE_W": "west"}


def _cell_of(frame: dict) -> tuple[int, int]:
    return int(round(frame["rover"]["y"])), int(round(frame["rover"]["x"]))


def _nearest_hotspot(pest: list[list[float]], r: int, c: int):
    """Nearest cell above the pest threshold, by Manhattan distance."""
    best, best_d = None, None
    for i, row in enumerate(pest):
        for j, v in enumerate(row):
            if v > NAV_PEST_THRESHOLD:
                d = abs(i - r) + abs(j - c)
                if best_d is None or d < best_d:
                    best, best_d = (i, j), d
    return best, best_d


def _lowest_health(health: list[list[float]]):
    best, best_v = (0, 0), 2.0
    for i, row in enumerate(health):
        for j, v in enumerate(row):
            if v < best_v:
                best, best_v = (i, j), v
    return best


def annotate(frames: list[dict]) -> list[dict]:
    """Derive, per frame, what the agent did and whether it accomplished anything.

    A frame records state AFTER its step, so the "before" picture is the previous
    frame. Frame 0 has no predecessor and is reported without a delta.
    """
    out = []
    for i, f in enumerate(frames):
        prev = frames[i - 1] if i > 0 else None
        action = f["action"]
        kind = ACTION_KIND.get(action, "idle")
        r, c = _cell_of(f)
        reason, detail, wasted = action.replace("_", " ").title(), "", False

        if action == "IRRIGATE":
            before = prev["health_grid"][r][c] if prev else f["health_grid"][r][c]
            after = f["health_grid"][r][c]
            gain = after - before
            if prev and prev["water"] <= 0.001:
                reason, wasted = "Irrigated with an empty tank", True
                detail = "no water left - the action did nothing"
            elif gain > 0.001:
                reason = f"Watered cell ({r},{c})"
                detail = f"health {before:.2f} -> {after:.2f}  (+{gain:.2f})"
            else:
                reason, wasted = f"Watered cell ({r},{c}) - already healthy", True
                detail = f"health {before:.2f}, nothing to gain"

        elif action == "SPRAY":
            before = prev["pest_grid"][r][c] if prev else f["pest_grid"][r][c]
            after = f["pest_grid"][r][c]
            removed = before - after
            if prev and prev["pesticide"] <= 0.001:
                reason, wasted = "Sprayed with an empty tank", True
                detail = "no pesticide left - the action did nothing"
            elif removed > 0.01:
                reason = f"Cleared pests at ({r},{c})"
                detail = f"severity {before:.2f} -> {after:.2f}, neighbours halved"
            else:
                reason, wasted = f"Sprayed clean cell ({r},{c})", True
                detail = "no infestation here - pesticide wasted"

        elif action in DIRECTION:
            target, dist = _nearest_hotspot(f["pest_grid"], r, c)
            if target is not None:
                detail = f"nearest hotspot ({target[0]},{target[1]}), {dist} cells away"
            else:
                tr, tc = _lowest_health(f["health_grid"])
                detail = f"field clear - driest cell ({tr},{tc})"
            reason = f"Driving {DIRECTION[action]} to ({r},{c})"

        elif action == "RETURN_TO_DEPOT":
            if (r, c) == (0, 0):
                reason = "Refilled at the depot"
                detail = "water and pesticide back to full, battery topped up"
            else:
                reason, wasted = "Called for the depot while away from it", True
                detail = f"depot is at (0,0), rover is at ({r},{c})"

        elif action == "SCAN":
            reason, detail = f"Scanning from ({r},{c})", "costs battery, changes nothing"
        else:
            reason, detail = f"Holding at ({r},{c})", "no action taken this step"

        out.append({
            "kind": kind, "reason": reason, "detail": detail, "wasted": wasted,
            "cell": [r, c],
        })
    return out


def summarise(frames: list[dict], notes: list[dict]) -> dict:
    """Headline counts for the KPI row."""
    kinds = [n["kind"] for n in notes]
    return {
        "treatments": sum(1 for k in kinds if k in ("irrigate", "spray")),
        "wasted": sum(1 for n in notes if n["wasted"]),
        "moves": sum(1 for k in kinds if k == "move"),
        "idle": sum(1 for k in kinds if k == "idle"),
    }


def build_episode(path: Path) -> dict:
    payload = json.loads(Path(path).read_text())
    frames, meta = payload["frames"], payload["meta"]
    notes = annotate(frames)
    return {
        "meta": meta,
        "summary": summarise(frames, notes),
        "frames": [
            {
                "t": f["t"], "a": f["action"], "r": f["reward"], "cum": f["cum_reward"],
                "bat": f["battery"], "wat": f["water"], "pst": f["pesticide"],
                "h": f["health_grid"], "p": f["pest_grid"],
                "rover": [f["rover"]["y"], f["rover"]["x"], f["rover"]["heading"]],
                **notes[i],
            }
            for i, f in enumerate(frames)
        ],
    }


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
_VENDOR_THREE = _REPO_ROOT / "assets" / "demo" / "vendor" / "three.min.js"
_SCENE_JS = Path(__file__).resolve().parent / "demo_scene.js"


def render_html(episodes: list[dict]) -> str:
    data = json.dumps({
        "episodes": episodes,
        "successHealth": SUCCESS_HEALTH,
        "successPest": SUCCESS_PEST,
    }, separators=(",", ":"))
    # three.js and the scene module are INLINED so the page stays a single file that
    # works from file:// with no network and no CORS. (ES-module builds cannot be
    # imported from file://; the r160 UMD build sets window.THREE from a plain
    # <script>, which is why that build is the one vendored.)
    three = _VENDOR_THREE.read_text() if _VENDOR_THREE.exists() else ""
    if not three:
        print(f"WARNING: {_VENDOR_THREE} missing -- 3D view will be unavailable.")
    return (
        _TEMPLATE
        .replace("/*__DATA__*/null", data)
        .replace("/*__THREE__*/", three)
        .replace("/*__SCENE__*/", _SCENE_JS.read_text())
    )


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AgriScout — episode viewer</title>
<style>
:root{
  color-scheme: light;
  --surface-0:#f4f3ef; --surface-1:#fcfcfb; --surface-2:#eceae4;
  --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781; --line:#dedcd4;
  --irrigate:#2a78d6; --spray:#eb6834; --depot:#1baf7a;
  --warn:#eda100; --bad:#e34948; --good:#008300;
  --soil:#cbbfa8; --crop:#14671c;
  --shadow:0 1px 2px rgba(0,0,0,.06),0 8px 24px rgba(0,0,0,.06);
}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){
    color-scheme: dark;
    --surface-0:#111110; --surface-1:#1a1a19; --surface-2:#232322;
    --ink:#ffffff; --ink-2:#c3c2b7; --muted:#8f8e86; --line:#33332f;
    --irrigate:#3987e5; --spray:#d95926; --depot:#199e70;
    --warn:#c98500; --bad:#e66767; --good:#4caf50;
    --soil:#4a4436; --crop:#3fa845;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --surface-0:#111110; --surface-1:#1a1a19; --surface-2:#232322;
  --ink:#ffffff; --ink-2:#c3c2b7; --muted:#8f8e86; --line:#33332f;
  --irrigate:#3987e5; --spray:#d95926; --depot:#199e70;
  --warn:#c98500; --bad:#e66767; --good:#4caf50;
  --soil:#4a4436; --crop:#3fa845;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
/* The UA `[hidden]` rule is display:none, which an explicit display:grid on .field
   silently outranks -- both views rendered at once until this was added. */
[hidden]{display:none!important}
body{margin:0;background:var(--surface-0);color:var(--ink);
  font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:1180px;margin:0 auto;padding:24px 20px 56px}

/* ---------- header ---------- */
header{display:flex;flex-wrap:wrap;gap:16px;align-items:baseline;margin-bottom:6px}
h1{font-size:21px;margin:0;letter-spacing:-.01em}
.sub{color:var(--ink-2);font-size:13.5px}
.pill{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;
  font-size:12px;font-weight:650;border:1px solid var(--line);background:var(--surface-2)}
.pill.win{color:var(--good);border-color:color-mix(in srgb,var(--good) 40%,transparent)}
.pill.lose{color:var(--bad);border-color:color-mix(in srgb,var(--bad) 40%,transparent)}
select,button{font:inherit;color:var(--ink);background:var(--surface-1);
  border:1px solid var(--line);border-radius:8px;padding:6px 10px;cursor:pointer}
button:hover,select:hover{border-color:var(--muted)}

/* ---------- layout ---------- */
.grid2{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(280px,1fr);gap:18px;margin-top:16px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.card{background:var(--surface-1);border:1px solid var(--line);border-radius:14px;
  padding:16px;box-shadow:var(--shadow)}
.card h2{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);
  margin:0 0 12px;font-weight:700}

/* ---------- view tabs / 3D ---------- */
.cardhead{display:flex;align-items:center;justify-content:space-between;gap:12px;
  margin-bottom:12px}
.tabs{display:flex;gap:5px}
.tabs button{padding:4px 11px;font-size:12px;font-weight:600;border-radius:8px;
  background:var(--surface-2);border:1px solid transparent;color:var(--ink-2)}
.tabs button.on{background:var(--ink);color:var(--surface-1);border-color:var(--ink)}
.view3d{width:100%;border-radius:11px;overflow:hidden;background:
  linear-gradient(180deg,color-mix(in srgb,var(--irrigate) 12%,var(--surface-2)),
  var(--surface-2));cursor:grab;position:relative}
.view3d:active{cursor:grabbing}
.view3d canvas{display:block}
.hint{position:absolute;right:10px;bottom:9px;font-size:11px;color:var(--ink-2);
  background:color-mix(in srgb,var(--surface-1) 78%,transparent);padding:3px 8px;
  border-radius:6px;pointer-events:none}

/* ---------- field ---------- */
.field{display:grid;gap:5px;width:100%}
.cell{position:relative;aspect-ratio:1;border-radius:7px;background:var(--soil);
  display:grid;place-items:center;transition:background .12s}
.cell.below{outline:2px dashed var(--warn);outline-offset:-2px}
.canopy{border-radius:5px;transition:width .12s,height .12s,background .12s}
.pest{position:absolute;top:3px;right:3px;border-radius:50%;background:var(--bad);
  border:1.5px solid var(--surface-1)}
.drop{position:absolute;bottom:3px;left:4px;width:7px;height:7px;border-radius:50%;
  background:var(--irrigate);border:1.5px solid var(--surface-1)}
.depotpad{position:absolute;inset:0;border-radius:7px;border:2px solid var(--depot);
  opacity:.85}
.rover{position:absolute;inset:12%;border-radius:50%;background:var(--ink);
  border:2.5px solid var(--surface-1);display:grid;place-items:center;
  box-shadow:0 2px 8px rgba(0,0,0,.35);z-index:2}
.rover span{color:var(--surface-1);font-size:12px;line-height:1;font-weight:700}
.rover.act-irrigate{background:var(--irrigate)}
.rover.act-spray{background:var(--spray)}
.rover.act-depot{background:var(--depot)}
.ping{position:absolute;inset:-6%;border-radius:50%;z-index:1;
  animation:ping .6s ease-out}
@keyframes ping{from{transform:scale(.7);opacity:.55}to{transform:scale(1.35);opacity:0}}
@media(prefers-reduced-motion:reduce){.ping{animation:none;opacity:0}
  .cell,.canopy{transition:none}}

/* ---------- now-doing ---------- */
.now{border-left:4px solid var(--muted);padding:2px 0 2px 12px;min-height:74px}
.now.k-irrigate{border-color:var(--irrigate)} .now.k-spray{border-color:var(--spray)}
.now.k-depot{border-color:var(--depot)} .now.k-move{border-color:var(--muted)}
.now .act{font-size:11px;letter-spacing:.08em;font-weight:750;text-transform:uppercase;
  color:var(--muted);display:flex;align-items:center;gap:7px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--muted);flex:none}
.k-irrigate .dot{background:var(--irrigate)} .k-spray .dot{background:var(--spray)}
.k-depot .dot{background:var(--depot)}
.now .reason{font-size:17px;font-weight:640;margin:3px 0 2px;letter-spacing:-.01em}
.now .detail{font-size:13px;color:var(--ink-2)}
.tag{display:inline-block;margin-left:7px;padding:1px 7px;border-radius:5px;font-size:10.5px;
  font-weight:700;background:color-mix(in srgb,var(--bad) 15%,transparent);color:var(--bad);
  vertical-align:middle}

/* ---------- meters & stats ---------- */
.meter{margin:9px 0}
.meter .lab{display:flex;justify-content:space-between;font-size:12px;color:var(--ink-2);
  margin-bottom:4px}
.meter .lab b{color:var(--ink);font-variant-numeric:tabular-nums;font-weight:640}
.track{height:7px;border-radius:4px;background:var(--surface-2);overflow:hidden}
.fill{height:100%;border-radius:4px;transition:width .12s}
.kpis{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-top:4px}
.kpi{background:var(--surface-2);border-radius:10px;padding:9px 11px}
.kpi .v{font-size:19px;font-weight:680;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.kpi .k{font-size:11px;color:var(--muted);margin-top:1px}
.thr{font-size:11px;color:var(--muted);margin-top:2px}

/* ---------- timeline ---------- */
.controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:16px}
input[type=range]{flex:1;min-width:200px;accent-color:var(--irrigate)}
.step{font-variant-numeric:tabular-nums;font-size:13px;color:var(--ink-2);min-width:96px}
/* Container is transparent and idle steps get their own recessive fill: when both
   were surface-2 the idle tail rendered identically to the empty track, so an
   episode that finished its work early looked like a strip that stopped halfway. */
.track-strip{display:flex;gap:1px;margin-top:14px;height:24px;overflow:hidden;
  border-radius:4px;background:transparent}
.track-strip i{flex:1 1 0;min-width:0;height:100%;cursor:pointer;transition:opacity .1s;
  background:color-mix(in srgb,var(--muted) 20%,transparent)}
.track-strip i.move{background:color-mix(in srgb,var(--muted) 45%,transparent)}
.track-strip i.irrigate{background:var(--irrigate)}
.track-strip i.spray{background:var(--spray)}
.track-strip i.depot{background:var(--depot)}
/* Marker must not affect layout: outline+offset grew the flex item's box and blew
   the strip's height out into the legend below it. An inset shadow paints inside. */
.track-strip i.now{box-shadow:inset 0 0 0 2px var(--ink)}
.track-strip i.wasted{opacity:.35}
.striplab{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);
  margin-top:5px}
.spark{width:100%;height:52px;display:block;margin-top:6px}

/* ---------- legend & table ---------- */
.legend{display:flex;flex-wrap:wrap;gap:16px;margin-top:18px;font-size:12.5px;color:var(--ink-2)}
.legend i{display:inline-flex;align-items:center;gap:6px;font-style:normal}
.sw{width:12px;height:12px;border-radius:4px;flex:none;border:1px solid var(--line)}
.ramp{width:76px;height:12px;border-radius:4px;flex:none;
  background:linear-gradient(90deg,var(--soil),var(--crop))}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin-top:10px}
th,td{text-align:left;padding:5px 9px;border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums}
th{color:var(--muted);font-weight:650;font-size:11px;text-transform:uppercase;letter-spacing:.06em}
td.num{text-align:right}
details{margin-top:18px}
summary{cursor:pointer;color:var(--ink-2);font-size:13px}
.tablewrap{max-height:340px;overflow:auto;margin-top:8px;border:1px solid var(--line);
  border-radius:10px}
.note{font-size:12px;color:var(--muted);margin-top:10px;line-height:1.5}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>AgriScout — episode viewer</h1>
    <span class="sub" id="epMeta"></span>
    <span style="flex:1"></span>
    <select id="epPick" aria-label="Choose episode"></select>
    <button id="theme" aria-label="Toggle colour theme">◐ Theme</button>
  </header>
  <div class="sub" id="epHead"></div>

  <div class="grid2">
    <section class="card">
      <div class="cardhead">
        <h2 style="margin:0">Field — <span id="fieldDims"></span></h2>
        <div class="tabs" role="tablist">
          <button id="tab3d" class="on" role="tab" aria-selected="true">3D</button>
          <button id="tabGrid" role="tab" aria-selected="false">Grid</button>
          <button id="tabOrbit" title="Toggle the slow automatic camera orbit">◐ Orbit</button>
        </div>
      </div>
      <div class="view3d" id="view3d"></div>
      <div class="field" id="field" role="img" aria-label="Crop field state" hidden></div>
      <div class="controls">
        <button id="play">▶ Play</button>
        <button id="prev" aria-label="Previous step">◀</button>
        <button id="next" aria-label="Next step">▶</button>
        <input type="range" id="scrub" min="0" value="0" aria-label="Episode step">
        <span class="step" id="stepLab"></span>
      </div>
      <div class="track-strip" id="strip" role="img"
           aria-label="Action taken at each step of the episode"></div>
      <div class="striplab"><span>every step of the episode — click to jump</span>
        <span id="stripCounts"></span></div>
      <div class="legend">
        <i><span class="ramp"></span> crop health: bare soil → thriving</i>
        <i><span class="sw" style="background:var(--bad)"></span> pest severity (dot size)</i>
        <i><span class="sw" style="background:var(--irrigate)"></span> irrigate</i>
        <i><span class="sw" style="background:var(--spray)"></span> spray</i>
        <i><span class="sw" style="background:var(--depot)"></span> depot</i>
        <i><span class="sw" style="background:color-mix(in srgb,var(--muted) 45%,transparent)"></span>
           move</i>
        <i><span class="sw" style="background:color-mix(in srgb,var(--muted) 20%,transparent)"></span>
           idle / scan</i>
        <i><span class="sw" style="border:2px dashed var(--warn);background:none"></span>
           below success health</i>
      </div>
    </section>

    <div>
      <section class="card">
        <h2>What the agent is doing</h2>
        <div class="now" id="now">
          <div class="act"><span class="dot"></span><span id="actName"></span></div>
          <div class="reason" id="reason"></div>
          <div class="detail" id="detail"></div>
        </div>
        <div class="kpis" style="margin-top:14px">
          <div class="kpi"><div class="v" id="stepR"></div><div class="k">step reward</div></div>
          <div class="kpi"><div class="v" id="cumR"></div><div class="k">episode total</div></div>
        </div>
        <svg class="spark" id="spark" preserveAspectRatio="none" aria-hidden="true"></svg>
      </section>

      <section class="card" style="margin-top:16px">
        <h2>Rover resources</h2>
        <div id="meters"></div>
      </section>

      <section class="card" style="margin-top:16px">
        <h2>Field state</h2>
        <div id="fieldState"></div>
      </section>
    </div>
  </div>

  <details>
    <summary>Table view (every step)</summary>
    <div class="tablewrap"><table id="tbl">
      <thead><tr><th>Step</th><th>Action</th><th>What happened</th>
        <th class="num">Reward</th><th class="num">Total</th>
        <th class="num">Health</th><th class="num">Pest</th></tr></thead>
      <tbody></tbody>
    </table></div>
  </details>

  <p class="note" id="footNote"></p>
</div>

<script>/*__THREE__*/</script>
<script>/*__SCENE__*/</script>
<script>
const DATA = /*__DATA__*/null;
const $ = s => document.querySelector(s);
const clamp01 = v => Math.max(0, Math.min(1, v));
const mean = g => g.flat().reduce((a,b)=>a+b,0) / g.flat().length;

/* Health ramp: ONE sequential scale, dry soil -> deep crop green.
   Read live from CSS custom properties so it follows the active theme. */
function cssVar(n){ return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function hex2rgb(h){ h=h.replace('#',''); if(h.length===3) h=[...h].map(c=>c+c).join('');
  return [parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)]; }
function healthColor(v){
  const a=hex2rgb(cssVar('--soil')), b=hex2rgb(cssVar('--crop')), t=clamp01(v);
  return `rgb(${a.map((x,i)=>Math.round(x+(b[i]-x)*t)).join(',')})`;
}

let ep = 0, idx = 0, timer = null;
const episodes = DATA.episodes;
const has3D = typeof THREE !== 'undefined' && typeof Scene !== 'undefined';
/* `snap` suppresses rover interpolation and treatment effects: correct when the
   viewer jumps (scrub / episode switch), wrong during playback where the glide
   between cells is the point. */
let snap3D = true;

/* ---------- episode picker ---------- */
episodes.forEach((e,i)=>{
  const o=document.createElement('option');
  o.value=i; o.textContent=`${e.meta.model} — seed ${e.meta.seed}`;
  $('#epPick').appendChild(o);
});
$('#epPick').onchange = e => { ep=+e.target.value; idx=0; mountEpisode(); };

/* ---------- build the static grid once per episode ---------- */
let cells=[];
function mountEpisode(){
  const E=episodes[ep], m=E.meta, f0=E.frames[0];
  $('#epPick').value=ep;   // keep the picker in sync when ep is set programmatically
  const rows=f0.h.length, cols=f0.h[0].length;
  $('#fieldDims').textContent = `${rows}×${cols} cells`;
  const won = m.success;
  $('#epMeta').innerHTML =
    `<span class="pill ${won?'win':'lose'}">${won?'✓ SUCCESS':'✕ FAILED'}</span>`;
  $('#epHead').textContent =
    `${m.model} agent · seed ${m.seed} · ${E.frames.length} steps · `
    + `final reward ${m.total_reward>=0?'+':''}${m.total_reward} · `
    + `${E.summary.treatments} treatments (${E.summary.wasted} wasted), `
    + `${E.summary.moves} moves, ${E.summary.idle} idle`;

  const field=$('#field'); field.innerHTML=''; cells=[];
  field.style.gridTemplateColumns=`repeat(${cols},1fr)`;
  /* Row 0 is the depot end; render rows top-down so (0,0) reads top-left. */
  for(let r=0;r<rows;r++){ cells[r]=[];
    for(let c=0;c<cols;c++){
      const d=document.createElement('div'); d.className='cell';
      d.innerHTML=`<div class="canopy"></div>`;
      if(r===0&&c===0){ const p=document.createElement('div'); p.className='depotpad';
        p.title='Depot — RETURN_TO_DEPOT refills here'; d.appendChild(p); }
      field.appendChild(d); cells[r][c]=d;
    }
  }
  $('#scrub').max=E.frames.length-1;
  if(has3D){
    Scene.reset();
    Scene.mount($('#view3d'), rows, cols);
    if(!$('#view3d .hint')){
      const h=document.createElement('div'); h.className='hint';
      h.textContent='drag to orbit · scroll to zoom'; $('#view3d').appendChild(h);
    }
  }
  buildStrip(); buildTable(); drawSpark();
  $('#footNote').textContent =
    `Success requires mean health ≥ ${DATA.successHealth} AND mean pest ≤ ${DATA.successPest} `
    + `at the final step. Colour is never the only channel: health also drives canopy size, `
    + `and cells below the health threshold carry a dashed outline.`;
  render();
}

function buildStrip(){
  const E=episodes[ep], s=$('#strip'); s.innerHTML='';
  E.frames.forEach((f,i)=>{
    const b=document.createElement('i');
    b.className=f.kind+(f.wasted?' wasted':'');
    b.title=`step ${f.t}: ${f.reason}`;
    b.onclick=()=>{ idx=i; stop(); render(); };
    s.appendChild(b);
  });
  const S=E.summary;
  $('#stripCounts').textContent =
    `${S.treatments} treatments · ${S.wasted} wasted · ${S.moves} moves · ${S.idle} idle`;
}

function buildTable(){
  const tb=$('#tbl tbody'); tb.innerHTML='';
  episodes[ep].frames.forEach(f=>{
    const tr=document.createElement('tr');
    tr.innerHTML=`<td>${f.t}</td><td>${f.a}</td>`
      +`<td>${f.reason}${f.wasted?' <span class="tag">wasted</span>':''}</td>`
      +`<td class="num">${f.r>=0?'+':''}${f.r.toFixed(2)}</td>`
      +`<td class="num">${f.cum.toFixed(2)}</td>`
      +`<td class="num">${mean(f.h).toFixed(3)}</td>`
      +`<td class="num">${mean(f.p).toFixed(3)}</td>`;
    tb.appendChild(tr);
  });
}

function drawSpark(){
  const E=episodes[ep], sv=$('#spark');
  const W=300,H=52,vals=E.frames.map(f=>f.cum);
  const lo=Math.min(0,...vals), hi=Math.max(0,...vals), span=(hi-lo)||1;
  const pts=vals.map((v,i)=>[i/(vals.length-1)*W, H-4-((v-lo)/span)*(H-8)]);
  const zeroY=H-4-((0-lo)/span)*(H-8);
  sv.setAttribute('viewBox',`0 0 ${W} ${H}`);
  sv.innerHTML=
    `<line x1="0" y1="${zeroY}" x2="${W}" y2="${zeroY}" stroke="var(--line)" stroke-width="1"/>`
   +`<polyline fill="none" stroke="var(--ink-2)" stroke-width="2" vector-effect="non-scaling-stroke"
      points="${pts.map(p=>p.join(',')).join(' ')}"/>`
   +`<circle id="sparkDot" r="3.5" fill="var(--irrigate)" cx="${pts[0][0]}" cy="${pts[0][1]}"/>`;
  sv._pts=pts;
}

/* ---------- per-step render ---------- */
function render(){
  const E=episodes[ep], f=E.frames[idx];
  const rows=f.h.length, cols=f.h[0].length;

  for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){
    const d=cells[r][c], hv=f.h[r][c], pv=f.p[r][c];
    const can=d.querySelector('.canopy');
    /* redundant encoding: colour ramp AND canopy size both track health */
    const size=32+clamp01(hv)*56;
    can.style.width=size+'%'; can.style.height=size+'%';
    can.style.background=healthColor(hv);
    d.classList.toggle('below', hv < DATA.successHealth);
    let pd=d.querySelector('.pest');
    if(pv>0.02){
      if(!pd){ pd=document.createElement('div'); pd.className='pest'; d.appendChild(pd); }
      const s=5+clamp01(pv/0.6)*11; pd.style.width=s+'px'; pd.style.height=s+'px';
      pd.style.opacity=0.45+clamp01(pv/0.6)*0.55;
    } else if(pd) pd.remove();
    d.title=`(${r},${c})  health ${hv.toFixed(2)}  pest ${pv.toFixed(2)}`;
  }

  document.querySelectorAll('.rover,.ping').forEach(n=>n.remove());
  const [rr,rc]=f.cell;
  const rov=document.createElement('div');
  rov.className='rover act-'+f.kind;
  const ARROW={MOVE_N:'↑',MOVE_S:'↓',MOVE_E:'→',MOVE_W:'←',
               IRRIGATE:'💧',SPRAY:'✳',RETURN_TO_DEPOT:'⌂',SCAN:'◎',WAIT:'•'};
  rov.innerHTML=`<span>${ARROW[f.a]||'•'}</span>`;
  rov.title=`Rover at (${rr},${rc}) — ${f.reason}`;
  cells[rr][rc].appendChild(rov);
  if(f.kind==='irrigate'||f.kind==='spray'||f.kind==='depot'){
    const pg=document.createElement('div'); pg.className='ping';
    pg.style.background=`var(--${f.kind==='depot'?'depot':f.kind})`;
    cells[rr][rc].appendChild(pg);
  }

  $('#now').className='now k-'+f.kind;
  $('#actName').textContent=f.a.replace(/_/g,' ');
  $('#reason').innerHTML=f.reason+(f.wasted?' <span class="tag">wasted</span>':'');
  $('#detail').textContent=f.detail;
  $('#stepR').textContent=(f.r>=0?'+':'')+f.r.toFixed(2);
  $('#stepR').style.color=f.r>=0?'var(--good)':'var(--bad)';
  $('#cumR').textContent=(f.cum>=0?'+':'')+f.cum.toFixed(1);

  const mh=mean(f.h), mp=mean(f.p);
  $('#meters').innerHTML=[
    ['Battery',f.bat,'var(--ink-2)'],['Water',f.wat,'var(--irrigate)'],
    ['Pesticide',f.pst,'var(--spray)'],
  ].map(([n,v,col])=>`<div class="meter"><div class="lab"><span>${n}</span>
      <b>${(v*100).toFixed(0)}%</b></div>
    <div class="track"><div class="fill" style="width:${v*100}%;background:${col}"></div></div>
    </div>`).join('');

  $('#fieldState').innerHTML=
    `<div class="meter"><div class="lab"><span>Mean crop health</span>
      <b>${mh.toFixed(3)}</b></div>
     <div class="track"><div class="fill" style="width:${mh*100}%;
       background:${mh>=DATA.successHealth?'var(--good)':'var(--warn)'}"></div></div>
     <div class="thr">${mh>=DATA.successHealth?'✓ above':'✕ below'} the ${DATA.successHealth} success threshold</div></div>
     <div class="meter"><div class="lab"><span>Mean pest severity</span>
      <b>${mp.toFixed(3)}</b></div>
     <div class="track"><div class="fill" style="width:${Math.min(100,mp*100/0.6)}%;
       background:${mp<=DATA.successPest?'var(--good)':'var(--bad)'}"></div></div>
     <div class="thr">${mp<=DATA.successPest?'✓ under':'✕ over'} the ${DATA.successPest} limit</div></div>`;

  if(has3D&&Scene.isMounted()) Scene.update(f, DATA.successHealth, snap3D);
  snap3D=false;

  $('#scrub').value=idx;
  $('#stepLab').textContent=`step ${f.t} / ${E.frames.length}`;
  $('#strip').querySelectorAll('i').forEach((n,i)=>n.classList.toggle('now',i===idx));
  const pts=$('#spark')._pts, dot=$('#sparkDot');
  if(pts&&dot&&pts[idx]){ dot.setAttribute('cx',pts[idx][0]); dot.setAttribute('cy',pts[idx][1]); }
}

/* ---------- transport ---------- */
function stop(){ if(timer){clearInterval(timer);timer=null;} $('#play').textContent='▶ Play'; }
$('#play').onclick=()=>{
  if(timer){ stop(); return; }
  $('#play').textContent='❚❚ Pause';
  timer=setInterval(()=>{
    if(idx>=episodes[ep].frames.length-1){ stop(); return; }
    idx++; render();
  },110);
};
$('#prev').onclick=()=>{ stop(); idx=Math.max(0,idx-1); render(); };
$('#next').onclick=()=>{ stop(); idx=Math.min(episodes[ep].frames.length-1,idx+1); render(); };
$('#scrub').oninput=e=>{ stop(); snap3D=true; idx=+e.target.value; render(); };

/* ---------- view tabs ---------- */
function setView(three){
  $('#view3d').hidden=!three; $('#field').hidden=three;
  $('#tab3d').classList.toggle('on',three); $('#tabGrid').classList.toggle('on',!three);
  $('#tab3d').setAttribute('aria-selected',String(three));
  $('#tabGrid').setAttribute('aria-selected',String(!three));
  if(three&&has3D){ Scene.resize(); snap3D=true; render(); }
}
$('#tab3d').onclick=()=>setView(true);
$('#tabGrid').onclick=()=>setView(false);
$('#tabOrbit').onclick=()=>{
  if(!has3D) return;
  const on=!Scene.getAuto(); Scene.setAuto(on); $('#tabOrbit').classList.toggle('on',on);
};
addEventListener('resize',()=>{ if(has3D) Scene.resize(); });
if(!has3D){ $('#tab3d').disabled=true; $('#tabOrbit').disabled=true; }
document.addEventListener('keydown',e=>{
  if(e.key==='ArrowLeft'){$('#prev').click();} if(e.key==='ArrowRight'){$('#next').click();}
  if(e.key===' '){e.preventDefault();$('#play').click();}
});
$('#theme').onclick=()=>{
  const cur=document.documentElement.getAttribute('data-theme');
  const next=cur==='dark'?'light':cur==='light'?'dark':
    (matchMedia('(prefers-color-scheme: dark)').matches?'light':'dark');
  document.documentElement.setAttribute('data-theme',next);
  // The 3D scene bakes light/soil colours at mount time, so a theme flip has to
  // rebuild it rather than just re-render.
  drawSpark(); snap3D=true; mountEpisode();
};
mountEpisode();
setView(has3D);
if(has3D) $('#tabOrbit').classList.add('on');
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--traces", nargs="+", default=["logs/traces/demo_*.json"],
                    help="trace JSON paths or globs")
    ap.add_argument("--out", default="assets/demo/index.html")
    args = ap.parse_args()

    paths: list[Path] = []
    for pattern in args.traces:
        hits = sorted(glob.glob(pattern))
        if not hits and Path(pattern).exists():
            hits = [pattern]
        paths.extend(Path(h) for h in hits)
    if not paths:
        raise SystemExit(f"no traces matched {args.traces}")

    episodes = [build_episode(p) for p in paths]
    # Trained agents first, then the scripted references they are measured against.
    order = {"ppo": 0, "dqn": 1, "a2c": 2, "reinforce": 3, "oracle": 8, "random": 9}
    episodes.sort(key=lambda e: order.get(e["meta"].get("model", ""), 5))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(episodes))
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({kb:.0f} KiB, {len(episodes)} episode(s))")
    for e in episodes:
        m, s = e["meta"], e["summary"]
        print(f"  {m['model']:9s} seed={m['seed']} reward={m.get('total_reward'):+7.2f} "
              f"success={m.get('success')} treatments={s['treatments']} wasted={s['wasted']}")


if __name__ == "__main__":
    main()
