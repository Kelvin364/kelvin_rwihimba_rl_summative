"""My PyBullet renderer for AgriScoutEnv (used by ``main.py`` in play mode).

IMPORTANT
---------
I import this module ONLY when rendering is actually requested, because it loads
``pybullet`` at import time. Training code must never touch it, which is what keeps
my training runs headless and portable.

I deliberately decoupled the renderer from any concrete environment class: it reads a
small documented *render state contract* (see :class:`RenderStateProtocol`) off
whatever object it is handed. ``AgriScoutEnv`` only has to expose those attributes,
and my smoke test passes in a lightweight mock instead.

Coordinate layout
------------------
* Grid cell ``(row, col)`` maps to world ``(x = (col + 1) * CELL, y = (row + 1) * CELL)``.
* The depot pad is drawn on grid cell ``(0, 0)``, the same square where
  RETURN_TO_DEPOT actually refills.
* Crop stalks have a fixed full height and sit sunk into a thick ground slab, so the
  *visible* height above ``z = 0`` tracks health. That lets me animate growth by
  moving a persistent body, which is far cheaper than rebuilding geometry.
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
CROP_MAX_H = 0.9           # full height of a crop stalk (health == 1.0)
CROP_MIN_VIS = 0.06        # minimum visible sliver so dead cells are still placed
CROP_HALF_XY = 0.32        # canopy half-width in x/y
STALK_HALF_XY = 0.07       # stalk half-width, thin so the canopy reads as foliage
CANOPY_H = 0.22            # canopy slab thickness
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
# Pest severity ramp: pale amber (light) -> deep red (severe).
_C_PEST_LIGHT = np.array([0.98, 0.72, 0.18])
_C_PEST_HEAVY = np.array([0.78, 0.06, 0.06])


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
            # Shadows give the crops readable depth. Without them the field renders as
            # flat coloured squares and the height differences are invisible.
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        self._crop_bodies: dict[tuple[int, int], int] = {}
        self._canopy_bodies: dict[tuple[int, int], int] = {}
        self._pest_bodies: dict[tuple[int, int], int] = {}
        self._debug_text_ids: dict[str, int] = {}

        # Interpolation state (see _interpolate_pose).
        self._prev_row = float(_getattr(env, "rover_row", 0.0))
        self._prev_col = float(_getattr(env, "rover_col", 0.0))
        self._prev_heading = float(_getattr(env, "rover_heading", 0.0))
        self._render_row, self._render_col = self._prev_row, self._prev_col
        self._render_heading = self._prev_heading

        self._build_static()
        self._build_crops()
        self._build_pests()
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
        # Note to self: centre on (field_cx, field_cy), NOT half of it. I had this
        # halved at first, which put the slab centre a quarter of the way in and left
        # the far row and column of crops hanging off the edge in mid-air.
        self.ground_id = self._box(
            (half_x, half_y, GROUND_THICK / 2),
            (0.42, 0.30, 0.17, 1.0),
            (field_cx, field_cy, -GROUND_THICK / 2),
        )
        # Depot pad. RETURN_TO_DEPOT only refills at GRID cell (0, 0), so I draw the
        # pad there. I originally drew it at the world origin, which put the marker one
        # cell diagonally away from the square that actually refills.
        dx, dy = _cell_world_xy(0, 0)
        self.depot_id = self._box(
            (0.5 * CELL, 0.5 * CELL, 0.02),
            (0.62, 0.63, 0.66, 1.0),
            (dx, dy, 0.012),
        )
        # Hatched border so the depot square reads as a marked pad, not a grey tile.
        for sx, sy in ((0.46, 0.0), (-0.46, 0.0), (0.0, 0.46), (0.0, -0.46)):
            self._box(
                (0.5 * CELL if sy else 0.04, 0.04 if sy else 0.5 * CELL, 0.03),
                (0.95, 0.78, 0.15, 1.0),
                (dx + sx, dy + sy, 0.02),
            )

    def _build_crops(self) -> None:
        """One stalk + one canopy per cell.

        Both are fixed-height bodies that get *re-posed* rather than rebuilt, so
        animating growth costs two resetBasePositionAndOrientation calls per cell.
        """
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                x, y = _cell_world_xy(r, c)
                self._crop_bodies[(r, c)] = self._box(
                    (STALK_HALF_XY, STALK_HALF_XY, CROP_MAX_H / 2),
                    (0.36, 0.26, 0.12, 1.0),
                    (x, y, 0.0),
                )
                self._canopy_bodies[(r, c)] = self._box(
                    (CROP_HALF_XY, CROP_HALF_XY, CANOPY_H / 2),
                    (*_C_GREEN, 1.0),
                    (x, y, 0.0),
                )

    def _build_pests(self) -> None:
        """One permanent disc per cell; see :meth:`_update_pests`."""
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                vis = p.createVisualShape(
                    p.GEOM_CYLINDER, radius=PEST_MAX_R, length=PEST_DISC_H,
                    rgbaColor=[*_C_PEST_LIGHT, 0.4],
                )
                self._pest_bodies[(r, c)] = p.createMultiBody(
                    baseMass=0, baseVisualShapeIndex=vis, basePosition=[0, 0, -5.0]
                )

    def _cyl(self, radius, length, rgba, pos, orn=(0, 0, 0, 1)):
        vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=list(rgba)
        )
        return p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=vis,
            basePosition=list(pos), baseOrientation=list(orn),
        )

    def _build_rover(self) -> None:
        """A recognisable field machine, not a cube.

        The rover is assembled from several bodies held in ``self._rover_parts`` as
        (body_id, local offset, local orientation). Every part is re-posed together
        in :meth:`_update_rover`, so the whole machine translates and rotates as one
        without any physics or parenting.
        """
        self._rover_parts: list[tuple[int, tuple[float, float, float], tuple]] = []

        def part(bid, offset, orn=(0, 0, 0, 1)):
            self._rover_parts.append((bid, offset, orn))
            return bid

        # Chassis + dark instrument deck + solar panel.
        self.rover_id = part(
            self._box(ROVER_HALF, (0.94, 0.94, 0.92, 1.0), (0, 0, -10)),
            (0.0, 0.0, 0.15),
        )
        part(self._box((0.13, 0.11, 0.05), (0.18, 0.20, 0.25, 1.0), (0, 0, -10)),
             (-0.02, 0.0, 0.30))
        part(self._box((0.22, 0.16, 0.012), (0.08, 0.13, 0.24, 1.0), (0, 0, -10)),
             (-0.01, 0.0, 0.36))

        # Four wheels (cylinders laid on their side).
        wheel_orn = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
        for dx, dy in ((0.19, 0.21), (0.19, -0.21), (-0.19, 0.21), (-0.19, -0.21)):
            part(self._cyl(0.10, 0.06, (0.10, 0.10, 0.11, 1.0), (0, 0, -10), wheel_orn),
                 (dx, dy, 0.10), wheel_orn)

        # Forward sensor boom + cone: an unmistakable heading indicator.
        boom_orn = p.getQuaternionFromEuler([0, math.pi / 2, 0])
        part(self._cyl(0.02, 0.26, (0.62, 0.65, 0.70, 1.0), (0, 0, -10), boom_orn),
             (0.34, 0.0, 0.22), boom_orn)
        self.arrow_id = part(
            self._box((0.09, 0.05, 0.05), (0.98, 0.78, 0.10, 1.0), (0, 0, -10)),
            (0.50, 0.0, 0.22),
        )
        # Beacon: recoloured per action so the machine announces what it just did.
        self.beacon_id = part(
            self._box((0.05, 0.05, 0.05), (1.0, 1.0, 1.0, 1.0), (0, 0, -10)),
            (-0.16, 0.0, 0.42),
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

    # -- pose interpolation ----------------------------------------------------
    def _interpolate_pose(self, alpha: float) -> None:
        """Blend the drawn pose from the previous committed pose to the current one."""
        env = self.env
        row = float(_getattr(env, "rover_row", 0.0))
        col = float(_getattr(env, "rover_col", 0.0))
        heading = float(_getattr(env, "rover_heading", 0.0))

        # Take the shortest way round so a W->E turn does not spin the long way.
        delta = (heading - self._prev_heading + math.pi) % (2 * math.pi) - math.pi
        self._render_row = self._prev_row + (row - self._prev_row) * alpha
        self._render_col = self._prev_col + (col - self._prev_col) * alpha
        self._render_heading = self._prev_heading + delta * alpha

        if alpha >= 1.0:  # commit, so the next step interpolates from here
            self._prev_row, self._prev_col = row, col
            self._prev_heading = self._prev_heading + delta

    # -- per-frame update ------------------------------------------------------
    def render(self, alpha: float = 1.0) -> None:
        """Draw the current env state.

        ``alpha`` in [0, 1] interpolates the rover between its previous and current
        cell: call ``render(a)`` a handful of times per env step with a sweeping
        from 0 to 1 and the rover glides instead of jumping. ``alpha=1`` (the
        default) simply snaps to the current pose and commits it.
        """
        env = self.env
        health = np.asarray(_getattr(env, "health_grid", np.ones((self.n_rows, self.n_cols))), dtype=float)
        pest = np.asarray(_getattr(env, "pest_grid", np.zeros((self.n_rows, self.n_cols))), dtype=float)
        irrig = _getattr(env, "irrigation_grid", np.zeros((self.n_rows, self.n_cols)))
        irrig = np.asarray(irrig, dtype=float)

        self._interpolate_pose(float(np.clip(alpha, 0.0, 1.0)))
        self._update_crops(health, irrig)
        self._update_pests(pest)
        self._update_rover()
        self._update_hud(health)

    def _update_crops(self, health: np.ndarray, irrig: np.ndarray) -> None:
        for (r, c), bid in self._crop_bodies.items():
            h = float(health[r, c])
            vis = CROP_MIN_VIS + h * (CROP_MAX_H - CROP_MIN_VIS)
            x, y = _cell_world_xy(r, c)
            # Sink the fixed-height stalk so the visible part above z=0 equals `vis`.
            p.resetBasePositionAndOrientation(
                bid, [x, y, vis - CROP_MAX_H / 2], [0, 0, 0, 1]
            )
            # Canopy rides on top of the stalk, so healthy crops stand visibly taller.
            p.resetBasePositionAndOrientation(
                self._canopy_bodies[(r, c)], [x, y, vis], [0, 0, 0, 1]
            )

            color = _health_color(h)
            timer = float(irrig[r, c]) if irrig.shape == health.shape else 0.0
            if timer > 0:
                blend = min(1.0, timer / 10.0) * 0.6
                color = color * (1 - blend) + _C_IRRIGATED * blend
            p.changeVisualShape(
                self._canopy_bodies[(r, c)], -1,
                rgbaColor=[color[0], color[1], color[2], 1.0],
            )

    def _update_pests(self, pest: np.ndarray) -> None:
        """Re-colour the persistent pest discs (one per cell).

        I first destroyed and re-created these on every frame so the radius could track
        severity, since pybullet cannot resize a primitive in place. That churned up to
        n_rows*n_cols bodies per frame and visibly flickered. Now each cell owns one
        permanent disc and I encode severity in colour and opacity instead: pale
        amber for a light infestation, deep red for a severe one. It is flicker-free
        and readable at a glance. Cells below the threshold I park under the slab.
        """
        for (r, c), bid in self._pest_bodies.items():
            sev = float(pest[r, c])
            x, y = _cell_world_xy(r, c)
            if sev <= 0.02:
                p.resetBasePositionAndOrientation(bid, [x, y, -5.0], [0, 0, 0, 1])
                continue
            t = min(1.0, sev / 0.6)                    # 0 = faint, 1 = severe
            color = _C_PEST_LIGHT + t * (_C_PEST_HEAVY - _C_PEST_LIGHT)
            alpha = 0.30 + 0.55 * t
            p.changeVisualShape(bid, -1, rgbaColor=[*color, alpha])
            # Lift slightly with severity so overlapping discs stay distinguishable.
            p.resetBasePositionAndOrientation(bid, [x, y, 0.02 + 0.01 * t], [0, 0, 0, 1])

    def _update_rover(self) -> None:
        env = self.env
        # Interpolated pose: `render()` may be called several times per env step
        # (see AgriScoutRenderer.render's `alpha`), so the rover glides between
        # cells instead of teleporting.
        row, col = self._render_row, self._render_col
        heading = self._render_heading
        x, y = _cell_world_xy(row, col)
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        quat = p.getQuaternionFromEuler([0, 0, heading])

        for bid, (ox, oy, oz), local_orn in self._rover_parts:
            # Rotate the part's local offset into world space about the rover's yaw.
            wx = x + ox * cos_h - oy * sin_h
            wy = y + ox * sin_h + oy * cos_h
            orn = p.multiplyTransforms([0, 0, 0], quat, [0, 0, 0], local_orn)[1]
            p.resetBasePositionAndOrientation(bid, [wx, wy, oz], orn)

        # Beacon colour follows the last action, matching the HTML demo's encoding.
        names = _getattr(env, "ACTION_NAMES", [])
        last = int(_getattr(env, "last_action", -1))
        action = names[last] if 0 <= last < len(names) else ""
        colour = {
            "IRRIGATE": [0.16, 0.47, 0.84, 1.0],
            "SPRAY": [0.92, 0.41, 0.20, 1.0],
            "RETURN_TO_DEPOT": [0.10, 0.69, 0.48, 1.0],
        }.get(action, [0.60, 0.63, 0.67, 1.0])
        p.changeVisualShape(self.beacon_id, -1, rgbaColor=colour)

        # Status bars float above the rover; encode level via the buried-fraction
        # trick against an invisible baseline (bar base pinned at rover top).
        water = float(np.clip(_getattr(env, "water", 0.0), 0.0, 1.0))
        pest = float(np.clip(_getattr(env, "pesticide", 0.0), 0.0, 1.0))
        base_z = ROVER_HALF[2] * 2 + 0.22
        self._place_bar(self.water_bar_id, x - 0.14, y, base_z, water)
        self._place_bar(self.pest_bar_id, x + 0.14, y, base_z, pest)

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
# Lives in environment.trace (numpy-only) so episodes can be recorded headlessly
# without importing pybullet. Re-exported here for backwards compatibility.
from environment.trace import (  # noqa: E402,F401
    EpisodeRecorder,
    _grid_to_list,
    _results_dir,
)
