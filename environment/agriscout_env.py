"""AgriScoutEnv -- my Gymnasium environment for autonomous crop scouting.

A rover drives over a 6x9 field of crop cells. Each cell carries a *health* level
and a *pest severity*. The rover has finite battery, water and pesticide, and has to
keep the field healthy -- irrigating stressed cells, spraying infested ones -- while
managing those resources and returning to the depot to refill.

I keep this module training-safe: it imports ONLY gymnasium and numpy, and never
pybullet or my renderer, so training stays headless. The renderer reads the
attributes I expose here (see ``environment.rendering.RenderStateProtocol``).
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ENV_VERSION = "agriscout-v1"

ACTION_NAMES = [
    "MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W",
    "SCAN", "IRRIGATE", "SPRAY", "RETURN_TO_DEPOT", "WAIT",
]

# Success thresholds (episode counts as a success if the field ends healthy and
# pests are controlled).
SUCCESS_HEALTH = 0.60
SUCCESS_PEST = 0.15

# Pest dynamics. I keep hotspots localized and their growth moderate so that success
# stays discriminative: I tuned these until my scripted oracle clears a random policy
# by a wide margin while a random policy still almost never succeeds.
N_HOTSPOTS = (3, 5)          # rng.integers(low, high) -> 3..4 hotspots
HOTSPOT_SEVERITY = (0.3, 0.6)
INFEST_THRESHOLD = 0.1       # a cell is "infested" above this
HOTSPOT_GROWTH = 0.025       # per-step growth of infested cells
SPREAD_COEF = 0.028          # P(spread) = SPREAD_COEF * min(severity, SPREAD_SEV_CAP)
SPREAD_SEV_CAP = 0.5         # spread probability saturates -> growth is bounded,
#                              never exponential (a missed hotspot costs linearly)
SPREAD_SEED = 0.20           # severity a freshly-infected neighbour receives
#                              (> INFEST_THRESHOLD so infections escalate)

# Health dynamics. I tuned these so a cleaned, irrigated field can actually reach
# SUCCESS_HEALTH -- otherwise the task would be unwinnable by construction.
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
# I make the per-step reward attributable entirely to the agent's OWN actions:
# natural pest spread and passive health decay change the world state and the success
# check, but never the reward. Before I did this, that uncontrollable per-step noise
# was collapsing PPO onto a single action.
#
#   r = REWARD_IRRIGATE_COEF * (health added at the treated cell by THIS irrigate)
#     + REWARD_SPRAY_COEF    * (pest severity removed by THIS spray, incl. halving)
#     - TIME_COST
#     - waste penalties (clean-spray / overwater / empty-tank / wall-bump / depot)
#     + potential-based shaping (see below)
#   Terminal: + SUCCESS_BONUS on success, - DEATH_PENALTY on battery death.
REWARD_IRRIGATE_COEF = 1.0
REWARD_SPRAY_COEF = 0.6
# I keep TIME_COST small on purpose. The horizon is fixed at max_steps (the battery
# cannot realistically die inside 150 steps), so a big time cost just adds a constant
# ~-7.5 offset with no gradient attached to it -- all it does is drown the real signal.
TIME_COST = 0.01
SUCCESS_BONUS = 5.0
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
# remain valid. Phi has two parts:
#
#   FIELD term  -- Phi_field = POT_HEALTH_COEF * mean_health
#                            - POT_PEST_COEF   * mean_pest
#     This is my success condition itself, made dense. In my first version I left it
#     out, so a policy only learned whether it had succeeded once, at the very last
#     step, from a single all-or-nothing bonus. When I measured it, the oracle-vs-
#     random gap in the dense reward was 2.16 over 150 steps (0.014/step) against
#     per-episode noise of sigma ~ 2-5 -- signal-to-noise below 1 -- and none of my
#     agents ever learned anything. Because the shaping telescopes, the total field
#     shaping an episode can earn is exactly Phi_field(s_T) - Phi_field(s_0): the
#     agent is paid for the NET improvement it produces, and cycling Phi up and down
#     sums to exactly zero (see SHAPING_GAMMA), so it cannot be farmed.
#
#   NAV term    -- rewards being close to the highest-priority work (nearest
#     active pest hotspot, else the lowest-health cell), giving a per-step
#     gradient TOWARD work that solves navigation credit assignment.
#
# I set the coefficients from the trajectory spread I measured on seeds 9000-9019:
# health delta oracle-vs-random = +0.487, pest delta = -0.570. At 25/20 that gives a
# ~24-point dense separation, which ``test_dense_reward_is_learnable`` holds me to.
POT_HEALTH_COEF = 25.0
POT_PEST_COEF = 20.0
POT_NAV_COEF = 1.0           # per approach step: 1.0/15 = +0.067 >> 0.01 time cost
# I use gamma = 1.0 so the shaping sum telescopes exactly over the episode. At
# gamma = 0.99 a large positive Phi bleeds -0.01*Phi every step -- about -34 per
# episode here, which would have re-drowned the signal I had just fixed. At 1.0 any
# cycle in Phi also sums to exactly zero, so the agent cannot farm it.
SHAPING_GAMMA = 1.0
NAV_PEST_THRESHOLD = 0.15    # cells above this are "work to do"

# --- Egocentric target features ----------------------------------------------
# I flatten the grid into the observation, which destroys spatial adjacency: to
# navigate from that alone, an MLP policy has to internally learn an argmax over
# R*C cells AND a Manhattan distance, from an unstructured vector. I could see the
# symptom in the action counts -- a PPO policy that had clearly started learning was
# still spending MOVE_E 0.22 / MOVE_W 0.26, oscillating in place because it could
# not work out which way to go.
#
# These 8 features state the navigation target directly. They are a pure function of
# state I already expose in full (my scripted oracle computes the same argmax), so I
# am not handing the agent privileged information -- I am only removing a
# representation burden that has nothing to do with the control problem.
# Signed offsets are mapped into [0, 1] with 0.5 == aligned, so the observation
# space stays Box(0, 1).
N_EGO_FEATURES = 8


class AgriScoutEnv(gym.Env):
    """Discrete-action, fully-observable crop-scouting environment.

    Observation (Box, float32 in [0, 1], length ``3*R*C + 6 + N_EGO_FEATURES``,
    which is 176 at the default 6x9 size):
        health grid | pest grid | irrigation timers/10 | rover row,col (norm)
        | battery | water | pesticide | step fraction | egocentric target features.

    Actions (Discrete 9): see :data:`ACTION_NAMES`.

    Reward (per step): action-attributed. I make it depend only on what the agent
    itself does; natural pest spread and passive health decay change the world but
    never the reward, so the agent is not blamed for what it cannot control::

        r = 1.0 * (health added at the treated cell by THIS irrigate)
            + 0.6 * (pest severity removed by THIS spray, incl. neighbour halving)
            - 0.01 (time) - waste penalties
            + (Phi(s') - Phi(s))    policy-invariant shaping, where
              Phi = 25 * mean_health - 20 * mean_pest + 1.0 * (-dist to work)
            + 5.0 terminal bonus if the episode ends in success
            - 2.0 terminal penalty if the battery dies.

    The shaping term is what makes the task learnable: it pays the agent
    continuously for net progress toward the success condition instead of only
    once, at the end, via a single all-or-nothing bonus.
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
        obs_dim = 3 * n_rows * n_cols + 6 + N_EGO_FEATURES
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
        shaping = SHAPING_GAMMA * self._potential() - phi_before

        reward = action_reward - TIME_COST + shaping

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

    def _nav_potential(self) -> float:
        """Negative normalized distance to the nearest high-priority cell (nearest
        active pest hotspot, else the lowest-health cell). Higher (closer to 0)
        = rover is nearer to useful work."""
        r, c = self._cell()
        pest_cells = np.argwhere(self.pest_grid > NAV_PEST_THRESHOLD)
        if pest_cells.shape[0] > 0:
            d = int((np.abs(pest_cells[:, 0] - r) + np.abs(pest_cells[:, 1] - c)).min())
        else:
            tr, tc = np.unravel_index(int(np.argmin(self.health_grid)), self.health_grid.shape)
            d = abs(int(tr) - r) + abs(int(tc) - c)
        return -float(d) / (self.n_rows + self.n_cols)

    def _potential(self) -> float:
        """Shaping potential Phi(s) = field-state term + navigation term.

        The field term is the success condition made dense; the nav term supplies
        the per-step gradient toward work. See the constants block for the full
        rationale and the measured coefficient derivation.
        """
        field = (
            POT_HEALTH_COEF * float(self.health_grid.mean())
            - POT_PEST_COEF * float(self.pest_grid.mean())
        )
        return field + POT_NAV_COEF * self._nav_potential()

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
    def _ego_features(self) -> np.ndarray:
        """Relative offsets to the two navigation targets (see N_EGO_FEATURES).

        Layout: [has_hotspot, pest_dr, pest_dc, max_pest,
                 health_dr, health_dc, health_here, pest_here]
        Offsets are ``(delta / span + 1) / 2`` so 0.5 means "already aligned".
        """
        r, c = self._cell()

        def _rel(tr: int, tc: int) -> tuple[float, float]:
            dr = (tr - r) / max(1, self.n_rows - 1)
            dc = (tc - c) / max(1, self.n_cols - 1)
            return (dr + 1.0) / 2.0, (dc + 1.0) / 2.0

        pest_cells = np.argwhere(self.pest_grid > NAV_PEST_THRESHOLD)
        if pest_cells.shape[0] > 0:
            # Nearest hotspot by Manhattan distance -- the same target the nav
            # potential rewards approaching, so the two signals agree.
            d = np.abs(pest_cells[:, 0] - r) + np.abs(pest_cells[:, 1] - c)
            tr, tc = pest_cells[int(np.argmin(d))]
            pest_dr, pest_dc = _rel(int(tr), int(tc))
            has_hotspot = 1.0
        else:
            pest_dr = pest_dc = 0.5      # aligned == "nothing to go to"
            has_hotspot = 0.0

        hr, hc = np.unravel_index(int(np.argmin(self.health_grid)), self.health_grid.shape)
        health_dr, health_dc = _rel(int(hr), int(hc))

        return np.array([
            has_hotspot, pest_dr, pest_dc, float(self.pest_grid.max()),
            health_dr, health_dc,
            float(self.health_grid[r, c]), float(self.pest_grid[r, c]),
        ], dtype=np.float32)

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
            self._ego_features(),
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
