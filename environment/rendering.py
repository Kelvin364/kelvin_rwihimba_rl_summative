"""PyBullet GUI renderer + episode trace recorder for AgriScoutEnv.

IMPORTANT
---------
This module is imported ONLY when rendering is requested (e.g. ``main.py --render``
or the episode viewer). It imports ``pybullet`` at module load time. Training code
MUST NEVER import this module, so that training stays headless and Colab-safe.

The renderer is deliberately decoupled from any concrete environment class. It reads
a small, documented *render state contract* (see :class:`RenderStateProtocol`) off
whatever object it is given. The real ``AgriScoutEnv`` only needs to expose those
attributes; a lightweight mock is used by the smoke test.

Coordinate layout
------------------
* Grid cell ``(row, col)`` maps to world ``(x = (col + 1) * CELL, y = (row + 1) * CELL)``.
* The depot pad sits at the world origin ``(0, 0)``, just outside the planted field.
* Crop boxes have a fixed full height and are sunk into a thick ground slab so that
  the *visible* height above ``z = 0`` equals ``health * CROP_MAX_H`` -- this lets us
  animate height by moving a persistent body (cheap) instead of rebuilding geometry.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pybullet as p

logger = logging.getLogger(__name__)

# --- world / visual constants -------------------------------------------------
CELL = 1.0                 # metres between crop-cell centres
CROP_MAX_H = 0.9           # full height of a crop box (health == 1.0)
CROP_MIN_VIS = 0.06        # minimum visible sliver so dead cells are still placed
CROP_HALF_XY = 0.32        # crop box half-width in x/y
GROUND_THICK = 2.0         # thick slab so "buried" crop portions stay hidden
PEST_MAX_R = 0.55          # pest disc radius at severity == 1.0
PEST_DISC_H = 0.01
ROVER_HALF = (0.28, 0.20, 0.14)
ARROW_LEN = 0.45
BAR_MAX_H = 0.6            # rover status bar full height
BAR_HALF_XY = 0.05

# RGB anchors for the health colour ramp (green -> yellow -> red).
_C_GREEN = np.array([0.13, 0.72, 0.15])
_C_YELLOW = np.array([0.93, 0.86, 0.12])
_C_RED = np.array([0.82, 0.12, 0.10])
_C_IRRIGATED = np.array([0.20, 0.45, 1.0])   # blue tint for recently watered cells
_C_WATER_BAR = [0.20, 0.45, 1.0, 1.0]
_C_PEST_BAR = [1.0, 0.55, 0.05, 1.0]


@runtime_checkable
class RenderStateProtocol(Protocol):
    """Attributes the renderer/recorder read off an environment.

    The real ``AgriScoutEnv`` must expose these. Optional attributes have
    sensible fallbacks (see :func:`_getattr`).
    """

    n_rows: int
    n_cols: int
    health_grid: np.ndarray        # (n_rows, n_cols) in [0, 1]
    pest_grid: np.ndarray          # (n_rows, n_cols) severity in [0, 1]
    rover_row: float               # grid coords (may be fractional)
    rover_col: float
    rover_heading: float           # radians
    battery: float                 # in [0, 1]
    water: float                   # in [0, 1]
    pesticide: float               # in [0, 1]
    step_count: int
    last_action: int
    last_reward: float
    cum_reward: float
    ACTION_NAMES: list[str]
    # Optional: irrigation_grid (steps of blue tint remaining), ENV_VERSION


def _getattr(env: Any, name: str, default: Any) -> Any:
    val = getattr(env, name, None)
    return default if val is None else val


def _health_color(h: float) -> np.ndarray:
    """Interpolate green -> yellow -> red as health drops from 1 -> 0."""
    h = float(np.clip(h, 0.0, 1.0))
    if h >= 0.5:
        t = (h - 0.5) / 0.5           # 0 at yellow, 1 at green
        return _C_YELLOW + t * (_C_GREEN - _C_YELLOW)
    t = h / 0.5                       # 0 at red, 1 at yellow
    return _C_RED + t * (_C_YELLOW - _C_RED)


def _cell_world_xy(row: float, col: float) -> tuple[float, float]:
    return (col + 1) * CELL, (row + 1) * CELL


def _resolve_mode(mode: str | None) -> int:
    """Return the desired pybullet connection mode.

    GUI is only ever attempted when explicitly requested AND not headless.
    """
    if os.environ.get("AGRISCOUT_HEADLESS") == "1":
        return p.DIRECT
    if mode in ("human", "gui", "GUI"):
        return p.GUI
    return p.DIRECT


class AgriScoutRenderer:
    """Persistent PyBullet renderer for an AgriScout environment.

    Bodies are created once in :meth:`__init__` and merely re-posed / re-coloured
    on every :meth:`render` call, so it is cheap enough for real-time GUI and for
    headless DIRECT smoke tests alike.
    """

    def __init__(self, env: Any, mode: str | None = "human") -> None:
        self.env = env
        self.n_rows = int(_getattr(env, "n_rows", 8))
        self.n_cols = int(_getattr(env, "n_cols", 12))

        wanted = _resolve_mode(mode)
        self.connection_mode = self._connect(wanted)
        self.gui = self.connection_mode == p.GUI

        p.resetSimulation()
        p.setGravity(0, 0, 0)
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        self._crop_bodies: dict[tuple[int, int], int] = {}
        self._pest_bodies: list[int] = []
        self._debug_text_ids: dict[str, int] = {}

        self._build_static()
        self._build_crops()
        self._build_rover()
        self._setup_camera()

    # -- connection ------------------------------------------------------------
    def _connect(self, wanted: int) -> int:
        if wanted == p.GUI:
            try:
                cid = p.connect(p.GUI)
                if cid >= 0:
                    return p.GUI
                raise RuntimeError("p.connect(p.GUI) returned a negative id")
            except Exception as exc:  # noqa: BLE001 - fall back to headless
                logger.warning(
                    "Could not open a PyBullet GUI window (%s); "
                    "falling back to headless DIRECT mode.",
                    exc,
                )
        p.connect(p.DIRECT)
        return p.DIRECT

    # -- static geometry -------------------------------------------------------
    def _box(self, half, rgba, pos, mass=0.0):
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(half), rgbaColor=list(rgba))
        return p.createMultiBody(baseMass=mass, baseVisualShapeIndex=vis, basePosition=list(pos))

    def _build_static(self) -> None:
        field_cx = ((self.n_cols - 1) / 2 + 1) * CELL
        field_cy = ((self.n_rows - 1) / 2 + 1) * CELL
        half_x = (self.n_cols + 2) * CELL / 2
        half_y = (self.n_rows + 2) * CELL / 2
        # Thick brown ground slab with its top face exactly at z = 0.
        self.ground_id = self._box(
            (half_x, half_y, GROUND_THICK / 2),
            (0.45, 0.32, 0.18, 1.0),
            (field_cx / 2, field_cy / 2, -GROUND_THICK / 2),
        )
        # Grey depot pad at the world origin.
        self.depot_id = self._box(
            (0.6 * CELL, 0.6 * CELL, 0.02),
            (0.55, 0.55, 0.58, 1.0),
            (0.0, 0.0, 0.011),
        )

    def _build_crops(self) -> None:
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                x, y = _cell_world_xy(r, c)
                bid = self._box(
                    (CROP_HALF_XY, CROP_HALF_XY, CROP_MAX_H / 2),
                    (*_C_GREEN, 1.0),
                    (x, y, 0.0),
                )
                self._crop_bodies[(r, c)] = bid

    def _build_rover(self) -> None:
        self.rover_id = self._box(ROVER_HALF, (0.15, 0.15, 0.17, 1.0), (0, 0, ROVER_HALF[2]))
        self.arrow_id = self._box(
            (ARROW_LEN / 2, 0.04, 0.04),
            (0.95, 0.85, 0.1, 1.0),
            (0, 0, ROVER_HALF[2]),
        )
        self.water_bar_id = self._box(
            (BAR_HALF_XY, BAR_HALF_XY, BAR_MAX_H / 2), _C_WATER_BAR, (0, 0, -10)
        )
        self.pest_bar_id = self._box(
            (BAR_HALF_XY, BAR_HALF_XY, BAR_MAX_H / 2), _C_PEST_BAR, (0, 0, -10)
        )

    def _setup_camera(self) -> None:
        if not self.gui:
            return
        target = [
            ((self.n_cols - 1) / 2 + 1) * CELL,
            ((self.n_rows - 1) / 2 + 1) * CELL,
            0.0,
        ]
        dist = max(self.n_rows, self.n_cols) * CELL * 1.15
        p.resetDebugVisualizerCamera(
            cameraDistance=dist,
            cameraYaw=45.0,
            cameraPitch=-45.0,          # fixed 45 deg overhead
            cameraTargetPosition=target,
        )

    # -- per-frame update ------------------------------------------------------
    def render(self) -> None:
        env = self.env
        health = np.asarray(_getattr(env, "health_grid", np.ones((self.n_rows, self.n_cols))), dtype=float)
        pest = np.asarray(_getattr(env, "pest_grid", np.zeros((self.n_rows, self.n_cols))), dtype=float)
        irrig = _getattr(env, "irrigation_grid", np.zeros((self.n_rows, self.n_cols)))
        irrig = np.asarray(irrig, dtype=float)

        self._update_crops(health, irrig)
        self._update_pests(pest)
        self._update_rover()
        self._update_hud(health)

    def _update_crops(self, health: np.ndarray, irrig: np.ndarray) -> None:
        for (r, c), bid in self._crop_bodies.items():
            h = float(health[r, c])
            vis = CROP_MIN_VIS + h * (CROP_MAX_H - CROP_MIN_VIS)
            # Sink the fixed-height box so the visible part above z=0 equals `vis`.
            center_z = vis - CROP_MAX_H / 2
            x, y = _cell_world_xy(r, c)
            p.resetBasePositionAndOrientation(bid, [x, y, center_z], [0, 0, 0, 1])

            color = _health_color(h)
            timer = float(irrig[r, c]) if irrig.shape == health.shape else 0.0
            if timer > 0:
                blend = min(1.0, timer / 10.0) * 0.6
                color = color * (1 - blend) + _C_IRRIGATED * blend
            p.changeVisualShape(bid, -1, rgbaColor=[color[0], color[1], color[2], 1.0])

    def _update_pests(self, pest: np.ndarray) -> None:
        # Pest disc radius must scale with severity; pybullet can't resize a shape
        # in place, so we rebuild the (typically few) active discs each frame.
        for bid in self._pest_bodies:
            p.removeBody(bid)
        self._pest_bodies.clear()
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                sev = float(pest[r, c])
                if sev <= 0.02:
                    continue
                radius = PEST_MAX_R * min(1.0, sev)
                x, y = _cell_world_xy(r, c)
                vis = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=radius,
                    length=PEST_DISC_H,
                    rgbaColor=[0.9, 0.05, 0.05, 0.45],
                )
                bid = p.createMultiBody(
                    baseMass=0, baseVisualShapeIndex=vis, basePosition=[x, y, 0.02]
                )
                self._pest_bodies.append(bid)

    def _update_rover(self) -> None:
        env = self.env
        row = float(_getattr(env, "rover_row", 0.0))
        col = float(_getattr(env, "rover_col", 0.0))
        heading = float(_getattr(env, "rover_heading", 0.0))
        x, y = _cell_world_xy(row, col)
        quat = p.getQuaternionFromEuler([0, 0, heading])
        p.resetBasePositionAndOrientation(self.rover_id, [x, y, ROVER_HALF[2]], quat)

        # Heading arrow: offset forward from rover centre along the heading.
        ax = x + math.cos(heading) * (ROVER_HALF[0] + ARROW_LEN / 2)
        ay = y + math.sin(heading) * (ROVER_HALF[0] + ARROW_LEN / 2)
        p.resetBasePositionAndOrientation(self.arrow_id, [ax, ay, ROVER_HALF[2]], quat)

        # Status bars float above the rover; encode level via the buried-fraction
        # trick against an invisible baseline (bar base pinned at rover top).
        water = float(np.clip(_getattr(env, "water", 0.0), 0.0, 1.0))
        pest = float(np.clip(_getattr(env, "pesticide", 0.0), 0.0, 1.0))
        base_z = ROVER_HALF[2] * 2 + 0.02
        self._place_bar(self.water_bar_id, x - 0.12, y, base_z, water)
        self._place_bar(self.pest_bar_id, x + 0.12, y, base_z, pest)

    def _place_bar(self, bid: int, x: float, y: float, base_z: float, level: float) -> None:
        fill = max(0.02, level) * BAR_MAX_H
        # Box has fixed half-height BAR_MAX_H/2; raise it so `fill` shows above base_z
        # and the remainder is tucked below the rover top (hidden by the body).
        center_z = base_z + fill - BAR_MAX_H / 2
        p.resetBasePositionAndOrientation(bid, [x, y, center_z], [0, 0, 0, 1])

    def _update_hud(self, health: np.ndarray) -> None:
        if not self.gui:
            return
        env = self.env
        names = _getattr(env, "ACTION_NAMES", [])
        last_action = int(_getattr(env, "last_action", -1))
        action_name = names[last_action] if 0 <= last_action < len(names) else "-"
        lines = [
            f"step: {int(_getattr(env, 'step_count', 0))}",
            f"battery: {float(_getattr(env, 'battery', 0.0)):.2f}",
            f"water: {float(_getattr(env, 'water', 0.0)):.2f}",
            f"pesticide: {float(_getattr(env, 'pesticide', 0.0)):.2f}",
            f"mean health: {float(np.mean(health)):.3f}",
            f"last action: {action_name}",
            f"cum reward: {float(_getattr(env, 'cum_reward', 0.0)):.2f}",
        ]
        text = "   ".join(lines)
        prev = self._debug_text_ids.get("hud", -1)
        kwargs = dict(
            text=text,
            textPosition=[0.0, -1.0, 1.5],
            textColorRGB=[1, 1, 1],
            textSize=1.3,
        )
        if prev >= 0:
            self._debug_text_ids["hud"] = p.addUserDebugText(replaceItemUniqueId=prev, **kwargs)
        else:
            self._debug_text_ids["hud"] = p.addUserDebugText(**kwargs)

    def close(self) -> None:
        try:
            if p.isConnected():
                p.disconnect()
        except Exception:  # noqa: BLE001 - closing must never raise
            pass


# =============================================================================
# Episode trace recorder
# =============================================================================
def _results_dir() -> Path:
    return Path(os.environ.get("AGRISCOUT_RESULTS", "./logs"))


def _grid_to_list(grid: np.ndarray, ndigits: int = 2) -> list[list[float]]:
    """Downsample a 2D grid to nested lists of rounded floats to keep traces small."""
    arr = np.asarray(grid, dtype=float)
    return [[round(float(v), ndigits) for v in row] for row in arr]


@dataclass
class EpisodeRecorder:
    """Serialises an episode to ``<AGRISCOUT_RESULTS>/traces/<run_id>.json``.

    Schema::

        {
          "meta":   {"env_version": str, "model": str, "seed": int, ...},
          "frames": [
            {"t", "rover": {"x","y","heading"}, "battery", "water", "pesticide",
             "health_grid", "pest_grid", "action", "reward", "cum_reward"}
          ]
        }
    """

    run_id: str
    meta: dict[str, Any]

    def __post_init__(self) -> None:
        self.frames: list[dict[str, Any]] = []
        self._cum_reward: float = 0.0

    def record(self, env: Any, action: int, reward: float) -> None:
        self._cum_reward += float(reward)
        names = _getattr(env, "ACTION_NAMES", [])
        action = int(action)
        action_name = names[action] if 0 <= action < len(names) else str(action)
        frame = {
            "t": int(_getattr(env, "step_count", len(self.frames))),
            "rover": {
                "x": round(float(_getattr(env, "rover_col", 0.0)), 2),
                "y": round(float(_getattr(env, "rover_row", 0.0)), 2),
                "heading": round(float(_getattr(env, "rover_heading", 0.0)), 3),
            },
            "battery": round(float(_getattr(env, "battery", 0.0)), 3),
            "water": round(float(_getattr(env, "water", 0.0)), 3),
            "pesticide": round(float(_getattr(env, "pesticide", 0.0)), 3),
            "health_grid": _grid_to_list(_getattr(env, "health_grid", np.zeros((0, 0)))),
            "pest_grid": _grid_to_list(_getattr(env, "pest_grid", np.zeros((0, 0)))),
            "action": action_name,
            "reward": round(float(reward), 3),
            "cum_reward": round(self._cum_reward, 3),
        }
        self.frames.append(frame)

    def save(self, results_dir: str | os.PathLike | None = None) -> Path:
        base = Path(results_dir) if results_dir is not None else _results_dir()
        traces = base / "traces"
        traces.mkdir(parents=True, exist_ok=True)
        path = traces / f"{self.run_id}.json"
        payload = {"meta": self.meta, "frames": self.frames}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")))
        tmp.replace(path)  # atomic + idempotent
        return path
