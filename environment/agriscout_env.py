"""AgriScoutEnv -- a Gymnasium environment for autonomous crop scouting.

A rover drives over an 8x12 field of crop cells. Each cell has a *health* level
and a *pest severity*. The rover carries finite battery, water and pesticide and
must keep the field healthy (irrigate stressed cells, spray infested ones) while
managing its resources and returning to the depot to refill.

This module is training-safe: it imports ONLY gymnasium + numpy. It must never
import pybullet / rendering. The renderer reads the attributes exposed here (see
``environment.rendering.RenderStateProtocol``).
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ENV_VERSION = "agriscout-v0"

ACTION_NAMES = [
    "MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W",
    "SCAN", "IRRIGATE", "SPRAY", "RETURN_TO_DEPOT", "WAIT",
]

# Success thresholds (episode counts as a success if the field ends healthy and
# pests are controlled).
SUCCESS_HEALTH = 0.70
SUCCESS_PEST = 0.30


class AgriScoutEnv(gym.Env):
    """Discrete-action, fully-observable crop-scouting environment.

    Observation (Box, float32 in [0, 1], length ``3*R*C + 6``):
        health grid | pest grid | irrigation timers/10 | rover row,col (norm)
        | battery | water | pesticide | step fraction.

    Actions (Discrete 9): see :data:`ACTION_NAMES`.

    Reward (per step): net field improvement minus time/waste costs::

        r = 10*(mean_health_after - mean_health_before)
            + 5*(mean_pest_before - mean_pest_after)
            - 0.05 (time) - waste/bump penalties
            - 2.0 terminal penalty if the battery dies.
    """

    metadata = {"render_modes": []}
    ENV_VERSION = ENV_VERSION
    ACTION_NAMES = ACTION_NAMES

    def __init__(self, n_rows: int = 8, n_cols: int = 12, max_steps: int = 200) -> None:
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        obs_dim = 3 * n_rows * n_cols + 6
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # Populated in reset().
        self.health_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        self.pest_grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        self.irrigation_grid = np.zeros((n_rows, n_cols), dtype=np.int32)
        self.reset(seed=0)

    # -- gym API ---------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        self.health_grid = rng.uniform(0.5, 1.0, (self.n_rows, self.n_cols)).astype(np.float32)
        self.pest_grid = rng.uniform(0.0, 0.25, (self.n_rows, self.n_cols)).astype(np.float32)
        self.irrigation_grid = np.zeros((self.n_rows, self.n_cols), dtype=np.int32)

        self.rover_row = 0.0
        self.rover_col = 0.0
        self.rover_heading = 0.0
        self.battery = 1.0
        self.water = 1.0
        self.pesticide = 1.0

        self.step_count = 0
        self.last_action = -1
        self.last_reward = 0.0
        self.cum_reward = 0.0
        self.water_used = 0.0
        return self._obs(), self._info()

    def step(self, action: int):
        action = int(action)
        self.last_action = action
        name = ACTION_NAMES[action]

        mean_h_before = float(self.health_grid.mean())
        mean_p_before = float(self.pest_grid.mean())

        penalty = self._apply_action(name)
        self._advance_world(name)

        mean_h_after = float(self.health_grid.mean())
        mean_p_after = float(self.pest_grid.mean())

        reward = (
            10.0 * (mean_h_after - mean_h_before)
            + 5.0 * (mean_p_before - mean_p_after)
            - 0.05
            + penalty
        )

        self.step_count += 1
        terminated = self.battery <= 0.0
        truncated = self.step_count >= self.max_steps
        if terminated:
            reward -= 2.0

        self.last_reward = float(reward)
        self.cum_reward += float(reward)

        info = self._info()
        if terminated or truncated:
            info["is_success"] = bool(self.is_success)
            info["mean_final_health"] = mean_h_after
            info["water_used"] = self.water_used
        return self._obs(), float(reward), terminated, truncated, info

    # -- mechanics -------------------------------------------------------------
    def _cell(self) -> tuple[int, int]:
        r = int(np.clip(round(self.rover_row), 0, self.n_rows - 1))
        c = int(np.clip(round(self.rover_col), 0, self.n_cols - 1))
        return r, c

    def _apply_action(self, name: str) -> float:
        """Apply the discrete action; return an extra reward penalty (<= 0)."""
        penalty = 0.0
        r, c = self._cell()

        if name in ("MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W"):
            dr, dc, heading = {
                "MOVE_N": (1, 0, math.pi / 2),
                "MOVE_S": (-1, 0, -math.pi / 2),
                "MOVE_E": (0, 1, 0.0),
                "MOVE_W": (0, -1, math.pi),
            }[name]
            nr, nc = self.rover_row + dr, self.rover_col + dc
            if 0 <= nr <= self.n_rows - 1 and 0 <= nc <= self.n_cols - 1:
                self.rover_row, self.rover_col = nr, nc
            else:
                penalty -= 0.05  # bumped a boundary
            self.rover_heading = heading
            self.battery -= 0.004

        elif name == "SCAN":
            self.battery -= 0.003

        elif name == "IRRIGATE":
            if self.water > 0:
                self.water_used += 0.05
                self.water = max(0.0, self.water - 0.05)
                self.irrigation_grid[r, c] = 10
                if self.health_grid[r, c] > 0.9:
                    penalty -= 0.05  # watering an already-healthy cell
                self.health_grid[r, c] = min(1.0, self.health_grid[r, c] + 0.15)
                self.battery -= 0.003
            else:
                penalty -= 0.10  # no water to give

        elif name == "SPRAY":
            if self.pesticide > 0:
                self.pesticide = max(0.0, self.pesticide - 0.05)
                if self.pest_grid[r, c] < 0.05:
                    penalty -= 0.05  # spraying a clean cell
                self.pest_grid[r, c] = max(0.0, self.pest_grid[r, c] - 0.5)
                self.battery -= 0.003
            else:
                penalty -= 0.10  # no pesticide left

        elif name == "RETURN_TO_DEPOT":
            if (r, c) == (0, 0):
                self.water = 1.0
                self.pesticide = 1.0
                self.battery = min(1.0, self.battery + 0.30)
            else:
                penalty -= 0.02  # only useful at the depot

        # WAIT: no-op besides the passive drains below.
        return penalty

    def _advance_world(self, name: str) -> None:
        rng = self.np_random
        # Pests creep upward everywhere.
        self.pest_grid = np.clip(
            self.pest_grid + rng.uniform(0.0, 0.015, self.pest_grid.shape), 0.0, 1.0
        ).astype(np.float32)
        # Health decays; pests accelerate decay; irrigation slows it.
        decay = 0.003 + 0.02 * self.pest_grid
        irrigated = self.irrigation_grid > 0
        decay = np.where(irrigated, decay * 0.3, decay)
        self.health_grid = np.clip(self.health_grid - decay, 0.0, 1.0).astype(np.float32)
        # Irrigated + low-pest cells recover a little.
        recover = np.where(irrigated & (self.pest_grid < 0.2), 0.006, 0.0)
        self.health_grid = np.clip(self.health_grid + recover, 0.0, 1.0).astype(np.float32)

        self.irrigation_grid = np.maximum(0, self.irrigation_grid - 1)
        self.battery = max(0.0, self.battery - 0.002)  # passive idle drain

    # -- observation / info ----------------------------------------------------
    def _obs(self) -> np.ndarray:
        parts = [
            self.health_grid.ravel(),
            self.pest_grid.ravel(),
            np.clip(self.irrigation_grid.ravel() / 10.0, 0.0, 1.0),
            np.array([
                self.rover_row / max(1, self.n_rows - 1),
                self.rover_col / max(1, self.n_cols - 1),
                self.battery,
                self.water,
                self.pesticide,
                self.step_count / self.max_steps,
            ], dtype=np.float32),
        ]
        obs = np.concatenate(parts).astype(np.float32)
        return np.clip(obs, 0.0, 1.0)

    def _info(self) -> dict[str, Any]:
        return {
            "mean_health": float(self.health_grid.mean()),
            "mean_pest": float(self.pest_grid.mean()),
            "battery": float(self.battery),
            "water_used": float(self.water_used),
        }

    # -- convenience -----------------------------------------------------------
    @property
    def mean_health(self) -> float:
        return float(self.health_grid.mean())

    @property
    def mean_pest(self) -> float:
        return float(self.pest_grid.mean())

    @property
    def is_success(self) -> bool:
        return self.mean_health >= SUCCESS_HEALTH and self.mean_pest <= SUCCESS_PEST

    def render(self):  # rendering is handled out-of-band by environment.rendering
        raise NotImplementedError("Use environment.rendering.AgriScoutRenderer for visuals.")


def make_env(seed: int | None = None, **kwargs) -> AgriScoutEnv:
    """Factory for a single AgriScoutEnv (importable for SubprocVecEnv workers)."""
    env = AgriScoutEnv(**kwargs)
    if seed is not None:
        env.reset(seed=seed)
    return env


# Register with gymnasium (idempotent across re-imports / subprocesses).
if "AgriScout-v0" not in gym.registry:
    gym.register(id="AgriScout-v0", entry_point="environment.agriscout_env:AgriScoutEnv")
