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
SUCCESS_HEALTH = 0.60
SUCCESS_PEST = 0.15

# Pest dynamics (localized; moderate so success stays discriminative -- tuned so
# the oracle gate passes with oracle-random >= 15 AND random success <= 0.30).
N_HOTSPOTS = (3, 5)          # rng.integers(low, high) -> 3..4 hotspots
HOTSPOT_SEVERITY = (0.3, 0.6)
INFEST_THRESHOLD = 0.1       # a cell is "infested" above this
HOTSPOT_GROWTH = 0.025       # per-step growth of infested cells
SPREAD_COEF = 0.028          # P(spread) = SPREAD_COEF * min(severity, SPREAD_SEV_CAP)
SPREAD_SEV_CAP = 0.5         # spread probability saturates -> growth is bounded,
#                              never exponential (a missed hotspot costs linearly)
SPREAD_SEED = 0.20           # severity a freshly-infected neighbour receives
#                              (> INFEST_THRESHOLD so infections escalate)

# Health dynamics (as in the oracle-passing version: a cleaned, irrigated field
# can reach SUCCESS_HEALTH).
BASE_DECAY = 0.0005          # passive per-step health loss on a clean cell
PEST_DECAY_COEF = 0.02       # extra decay proportional to local pest severity
IRRIGATED_DECAY_FACTOR = 0.3  # irrigated cells decay slower
RECOVER_RATE = 0.012         # per-step recovery for irrigated, low-pest cells
BASE_RECOVER = 0.004         # passive recovery for ANY low-pest cell (gated on
#                              LOCAL pest, not field mean): collapse is never
#                              absorbing -- a cleaned cell always heals back.
LOW_PEST_RECOVER = 0.2       # a cell recovers when its local pest is below this
IRRIGATE_BOOST = 0.2         # instant health added by an IRRIGATE action

# --- ACTION-ATTRIBUTED REWARD ------------------------------------------------
# The per-step reward is 100% attributable to the agent's OWN actions: natural
# pest spread/growth and passive health decay do NOT enter the reward (they only
# change the world state / success). This removes the uncontrollable per-step
# noise that previously collapsed PPO to a single action.
#
#   r = REWARD_IRRIGATE_COEF * (health added at the treated cell by THIS irrigate)
#     + REWARD_SPRAY_COEF    * (pest severity removed by THIS spray, incl. halving)
#     - TIME_COST
#     - FIELD_PRESSURE_COEF * mean_pest   (tiny smooth term: "farming" pest is
#                                          never profitable)
#     - waste penalties (clean-spray / overwater / empty-tank / wall-bump / depot)
#   Terminal: + SUCCESS_BONUS on success, - DEATH_PENALTY on battery death.
REWARD_IRRIGATE_COEF = 1.0
REWARD_SPRAY_COEF = 0.6
TIME_COST = 0.05
FIELD_PRESSURE_COEF = 0.05
SUCCESS_BONUS = 20.0
DEATH_PENALTY = 2.0
# Waste penalties.
CLEAN_SPRAY_PENALTY = 0.05   # spraying a cell that is already clean
OVERWATER_PENALTY = 0.05     # irrigating an already-healthy cell
EMPTY_TANK_PENALTY = 0.10    # spray/irrigate with an empty tank
WALL_BUMP_PENALTY = 0.05     # moving into a boundary
DEPOT_PENALTY = 0.02         # RETURN_TO_DEPOT while not at the depot

# --- Potential-based reward shaping (Ng, Harada & Russell, 1999) --------------
# F(s, a, s') = gamma * Phi(s') - Phi(s) is *policy-invariant*: it provably does
# NOT change the set of optimal policies, so the oracle gate and reward semantics
# remain valid. Phi rewards being close to the highest-priority work (nearest
# active pest hotspot, else the lowest-health cell), giving a per-step gradient
# TOWARD work that solves the navigation credit-assignment problem.
NAV_SHAPING_COEF = 0.5       # ~+0.5 total per full approach (>> 0.05 time cost)
NAV_SHAPING_GAMMA = 0.99
NAV_PEST_THRESHOLD = 0.15    # cells above this are "work to do"


class AgriScoutEnv(gym.Env):
    """Discrete-action, fully-observable crop-scouting environment.

    Observation (Box, float32 in [0, 1], length ``3*R*C + 6``):
        health grid | pest grid | irrigation timers/10 | rover row,col (norm)
        | battery | water | pesticide | step fraction.

    Actions (Discrete 9): see :data:`ACTION_NAMES`.

    Reward (per step): ACTION-ATTRIBUTED -- 100% determined by the agent's own
    actions; natural pest spread/growth and passive health decay do NOT enter the
    reward, only the world state::

        r = 1.0 * (health added at the treated cell by THIS irrigate)
            + 0.6 * (pest severity removed by THIS spray, incl. neighbour halving)
            - 0.05 (time) - 0.05 * mean_pest (field pressure) - waste penalties
            + 0.5 * (0.99 * Phi(s') - Phi(s))   policy-invariant nav shaping
            + 20.0 terminal bonus if the episode ends in success
            - 2.0 terminal penalty if the battery dies.
    """

    metadata = {"render_modes": []}
    ENV_VERSION = ENV_VERSION
    ACTION_NAMES = ACTION_NAMES

    def __init__(self, n_rows: int = 6, n_cols: int = 9, max_steps: int = 150) -> None:
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
        # Localized pests: a few infested hotspots, rest clean (treatable).
        self.pest_grid = np.zeros((self.n_rows, self.n_cols), dtype=np.float32)
        n_hot = int(rng.integers(*N_HOTSPOTS))
        flat_idx = rng.choice(self.n_rows * self.n_cols, size=n_hot, replace=False)
        sev = rng.uniform(*HOTSPOT_SEVERITY, size=n_hot).astype(np.float32)
        self.pest_grid.flat[flat_idx] = sev
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

        # Reward comes ONLY from the agent's action (attribution) + time/pressure;
        # the world dynamics below change state but never the reward directly.
        phi_before = self._potential()
        action_reward = self._apply_action(name)
        self._advance_world(name)
        # Policy-invariant potential-based shaping (Ng et al. 1999).
        shaping = NAV_SHAPING_COEF * (NAV_SHAPING_GAMMA * self._potential() - phi_before)

        reward = (
            action_reward
            - TIME_COST
            - FIELD_PRESSURE_COEF * float(self.pest_grid.mean())
            + shaping
        )

        self.step_count += 1
        terminated = self.battery <= 0.0
        truncated = self.step_count >= self.max_steps
        if terminated:
            reward -= DEATH_PENALTY
        if (terminated or truncated) and self.is_success:
            reward += SUCCESS_BONUS

        self.last_reward = float(reward)
        self.cum_reward += float(reward)

        info = self._info()
        if terminated or truncated:
            info["is_success"] = bool(self.is_success)
            info["mean_final_health"] = float(self.health_grid.mean())
            info["water_used"] = self.water_used
        return self._obs(), float(reward), terminated, truncated, info

    # -- mechanics -------------------------------------------------------------
    def _cell(self) -> tuple[int, int]:
        r = int(np.clip(round(self.rover_row), 0, self.n_rows - 1))
        c = int(np.clip(round(self.rover_col), 0, self.n_cols - 1))
        return r, c

    def _potential(self) -> float:
        """Shaping potential Phi(s): negative normalized distance to the nearest
        high-priority cell (nearest active pest hotspot, else the lowest-health
        cell). Higher (closer to 0) = rover is nearer to useful work."""
        r, c = self._cell()
        pest_cells = np.argwhere(self.pest_grid > NAV_PEST_THRESHOLD)
        if pest_cells.shape[0] > 0:
            d = int((np.abs(pest_cells[:, 0] - r) + np.abs(pest_cells[:, 1] - c)).min())
        else:
            tr, tc = np.unravel_index(int(np.argmin(self.health_grid)), self.health_grid.shape)
            d = abs(int(tr) - r) + abs(int(tc) - c)
        return -float(d) / (self.n_rows + self.n_cols)

    def _apply_action(self, name: str) -> float:
        """Apply the discrete action; return the ACTION-ATTRIBUTED reward.

        Positive terms are the agent's direct effect (health added by irrigate,
        pest removed by spray); negative terms are waste penalties. Passive world
        dynamics are handled separately in ``_advance_world`` and never rewarded.
        """
        reward = 0.0
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
                reward -= WALL_BUMP_PENALTY  # bumped a boundary
            self.rover_heading = heading
            self.battery -= 0.004

        elif name == "SCAN":
            self.battery -= 0.003

        elif name == "IRRIGATE":
            if self.water > 0:
                self.water_used += 0.05
                self.water = max(0.0, self.water - 0.05)
                self.irrigation_grid[r, c] = 10
                before = float(self.health_grid[r, c])
                if before > 0.9:
                    reward -= OVERWATER_PENALTY  # watering an already-healthy cell
                after = min(1.0, before + IRRIGATE_BOOST)
                self.health_grid[r, c] = after
                reward += REWARD_IRRIGATE_COEF * (after - before)  # attributed health
                self.battery -= 0.003
            else:
                reward -= EMPTY_TANK_PENALTY  # no water to give

        elif name == "SPRAY":
            if self.pesticide > 0:
                self.pesticide = max(0.0, self.pesticide - 0.05)
                if float(self.pest_grid[r, c]) < 0.05:
                    reward -= CLEAN_SPRAY_PENALTY  # spraying an already-clean cell
                removed = float(self.pest_grid[r, c])  # current cell cleared to 0
                self.pest_grid[r, c] = 0.0
                for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols:
                        removed += 0.5 * float(self.pest_grid[nr, nc])  # halved away
                        self.pest_grid[nr, nc] *= 0.5
                reward += REWARD_SPRAY_COEF * removed  # attributed pest removed
                self.battery -= 0.003
            else:
                reward -= EMPTY_TANK_PENALTY  # no pesticide left

        elif name == "RETURN_TO_DEPOT":
            if (r, c) == (0, 0):
                self.water = 1.0
                self.pesticide = 1.0
                self.battery = min(1.0, self.battery + 0.30)
            else:
                reward -= DEPOT_PENALTY  # only useful at the depot

        # WAIT: no-op besides the passive drains below.
        return reward

    @staticmethod
    def _shift(a: np.ndarray, dr: int, dc: int) -> np.ndarray:
        """Return ``out`` where ``out[r, c] = a[r-dr, c-dc]`` (0 past the edges)."""
        out = np.roll(a, (dr, dc), axis=(0, 1))
        if dr > 0:
            out[:dr, :] = 0
        elif dr < 0:
            out[dr:, :] = 0
        if dc > 0:
            out[:, :dc] = 0
        elif dc < 0:
            out[:, dc:] = 0
        return out

    def _advance_world(self, name: str) -> None:
        rng = self.np_random
        pest_pre = self.pest_grid
        infested = pest_pre > INFEST_THRESHOLD

        # Hotspots grow; then infested cells spread to 4-neighbours (one pass).
        pest = np.where(infested, pest_pre + HOTSPOT_GROWTH, pest_pre)
        new_infection = np.zeros_like(pest)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            src = self._shift(pest_pre, dr, dc)          # infested neighbour severity
            # Spread probability SATURATES in severity -> bounded, never exponential.
            prob = SPREAD_COEF * np.minimum(src, SPREAD_SEV_CAP) * (src > INFEST_THRESHOLD)
            hit = rng.random(pest.shape) < prob
            new_infection = np.maximum(new_infection, hit * SPREAD_SEED)
        pest = np.maximum(pest, new_infection)
        self.pest_grid = np.clip(pest, 0.0, 1.0).astype(np.float32)

        # Health decays (pests accelerate it); irrigated cells decay slower.
        decay = BASE_DECAY + PEST_DECAY_COEF * self.pest_grid
        irrigated = self.irrigation_grid > 0
        decay = np.where(irrigated, decay * IRRIGATED_DECAY_FACTOR, decay)
        health = self.health_grid - decay
        # Recovery is gated on LOCAL cell pest only (never the field mean), so a
        # cleaned cell always heals back -> collapse is non-absorbing / graceful.
        low_pest = self.pest_grid < LOW_PEST_RECOVER
        recover = np.where(low_pest, BASE_RECOVER, 0.0)
        recover = np.where(irrigated & low_pest, RECOVER_RATE, recover)
        self.health_grid = np.clip(health + recover, 0.0, 1.0).astype(np.float32)

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
