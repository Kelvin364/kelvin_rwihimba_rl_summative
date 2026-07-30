"""Episode trace recording: serialise an episode to JSON.

Deliberately dependency-light: numpy only. Trace recording has nothing to do with
3D rendering, but it used to live in ``environment.rendering``, which imports
pybullet at module load. That made it impossible to record an episode headlessly
(for the HTML viewer, or on a machine with no GL) without dragging in pybullet.

``environment.rendering`` re-exports these names, so existing imports keep working.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _getattr(env: Any, name: str, default: Any) -> Any:
    val = getattr(env, name, None)
    return default if val is None else val


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

    NB: a frame records the state AFTER its step, and there is no frame for the
    reset state, so the "before" picture for frame ``i`` is frame ``i - 1``.
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
