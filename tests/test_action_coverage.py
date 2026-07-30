"""Every action is legal, has a defined effect, and its edge case is handled.

The report claims the action space is exhaustive and that degenerate uses are
explicitly penalised rather than silently ignored. These tests are the evidence for
that claim, so it cannot quietly stop being true.

    uv run pytest tests/test_action_coverage.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.agriscout_env import (  # noqa: E402
    ACTION_NAMES,
    CLEAN_SPRAY_PENALTY,
    DEPOT_PENALTY,
    EMPTY_TANK_PENALTY,
    OVERWATER_PENALTY,
    WALL_BUMP_PENALTY,
    AgriScoutEnv,
)

A = {name: i for i, name in enumerate(ACTION_NAMES)}


def _env(seed: int = 0) -> AgriScoutEnv:
    env = AgriScoutEnv()
    env.reset(seed=seed)
    return env


def test_every_action_is_legal_and_steps_the_env():
    """No action may raise, and each must advance the environment."""
    for name, idx in A.items():
        env = _env()
        obs, reward, term, trunc, info = env.step(idx)
        assert env.step_count == 1, f"{name} did not advance the step counter"
        assert np.isfinite(reward), f"{name} produced a non-finite reward"
        assert env.observation_space.contains(obs), f"{name} produced an invalid obs"
        assert not (term or trunc), f"{name} ended the episode on step 1"


def test_all_actions_reachable_by_a_random_policy():
    """A random policy must actually exercise all 9 actions (no dead entries)."""
    env = _env()
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(3000):
        a = int(rng.integers(0, env.action_space.n))
        seen.add(a)
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            env.reset()
    assert seen == set(range(len(ACTION_NAMES))), (
        f"unreached actions: {sorted(set(range(len(ACTION_NAMES))) - seen)}"
    )


def test_movement_changes_position_and_respects_bounds():
    env = _env()
    env.rover_row, env.rover_col = 2.0, 3.0
    env.step(A["MOVE_N"])
    assert (env.rover_row, env.rover_col) == (3.0, 3.0)
    env.step(A["MOVE_E"])
    assert (env.rover_row, env.rover_col) == (3.0, 4.0)


# The waste penalties are defined on the ACTION-ATTRIBUTED reward, so they are
# asserted against `_apply_action` rather than the step reward. The step reward also
# carries potential-based shaping, which can legitimately swamp a -0.05 penalty: with
# the pest grid zeroed, every cell starts recovering and the health term alone pays
# about +0.03. Asserting on the total would test the shaping, not the penalty.
@pytest.mark.parametrize("action,row,col", [
    ("MOVE_S", 0.0, 0.0),                       # off the south edge
    ("MOVE_W", 0.0, 0.0),                       # off the west edge
])
def test_wall_bump_is_penalised_not_silent(action, row, col):
    """Driving into a boundary must cost, and must not move the rover."""
    env = _env()
    env.rover_row, env.rover_col = row, col
    before = (env.rover_row, env.rover_col)
    attributed = env._apply_action(action)
    assert (env.rover_row, env.rover_col) == before, "rover left the field"
    assert attributed == pytest.approx(-WALL_BUMP_PENALTY), "wall bump was free"


def test_spraying_a_clean_cell_is_penalised():
    env = _env()
    env.pest_grid[:] = 0.0                       # nothing to kill anywhere
    assert env._apply_action("SPRAY") == pytest.approx(-CLEAN_SPRAY_PENALTY)


def test_irrigating_a_healthy_cell_is_penalised():
    env = _env()
    env.health_grid[:] = 1.0                     # nothing to gain anywhere
    assert env._apply_action("IRRIGATE") == pytest.approx(-OVERWATER_PENALTY)


@pytest.mark.parametrize("action,tank", [("IRRIGATE", "water"), ("SPRAY", "pesticide")])
def test_empty_tank_is_penalised(action, tank):
    env = _env()
    setattr(env, tank, 0.0)
    assert env._apply_action(action) == pytest.approx(-EMPTY_TANK_PENALTY)


def test_depot_refills_only_at_the_depot_cell():
    env = _env()
    env.rover_row, env.rover_col = 0.0, 0.0      # on the pad
    env.water, env.pesticide = 0.2, 0.2
    env.step(A["RETURN_TO_DEPOT"])
    assert env.water == 1.0 and env.pesticide == 1.0, "depot did not refill"

    env = _env()
    env.rover_row, env.rover_col = 3.0, 4.0      # away from the pad
    env.water = 0.2
    attributed = env._apply_action("RETURN_TO_DEPOT")
    assert env.water < 1.0, "refilled while away from the depot"
    assert attributed == pytest.approx(-DEPOT_PENALTY), "off-depot call was free"


def test_treatments_actually_change_the_world():
    """IRRIGATE and SPRAY must have a measurable effect at the rover's cell."""
    env = _env()
    r, c = 2, 3
    env.rover_row, env.rover_col = float(r), float(c)
    env.health_grid[r, c] = 0.4
    env.step(A["IRRIGATE"])
    assert env.health_grid[r, c] > 0.4, "IRRIGATE did not raise cell health"

    env = _env()
    env.rover_row, env.rover_col = float(r), float(c)
    env.pest_grid[r, c] = 0.8
    env.step(A["SPRAY"])
    assert env.pest_grid[r, c] < 0.8, "SPRAY did not reduce cell pest severity"


def test_resources_deplete_and_are_bounded():
    env = _env()
    for _ in range(120):
        env.step(A["MOVE_N"])
    assert env.battery < 1.0, "battery never depleted"
    for attr in ("battery", "water", "pesticide"):
        assert 0.0 <= getattr(env, attr) <= 1.0, f"{attr} left [0, 1]"


def test_episode_truncates_at_the_horizon():
    env = _env()
    trunc = False
    for _ in range(env.max_steps):
        _, _, term, trunc, info = env.step(A["WAIT"])
        if term or trunc:
            break
    assert trunc, "episode did not truncate at max_steps"
    assert "is_success" in info and "mean_final_health" in info


def test_seeding_is_reproducible():
    """Same seed + same actions must give byte-identical trajectories."""
    def rollout(seed: int) -> list[float]:
        env = _env(seed)
        rng = np.random.default_rng(123)
        return [float(env.step(int(rng.integers(0, 9)))[1]) for _ in range(80)]

    assert rollout(42) == rollout(42), "identical seeds diverged"
    assert rollout(42) != rollout(43), "different seeds produced identical rollouts"
