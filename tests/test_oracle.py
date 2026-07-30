"""Winnability proof for AgriScoutEnv.

A scripted greedy *oracle* must clearly beat both a random policy and a do-nothing
(WAIT) policy, and must actually succeed on most episodes. This test IS the
definition of a winnable environment -- if it fails, the dynamics constants in
``environment/agriscout_env.py`` need tuning.

    uv run pytest tests/test_oracle.py -s -q
    uv run python tests/test_oracle.py         # just print the benchmark
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.agriscout_env import SUCCESS_BONUS, AgriScoutEnv  # noqa: E402

EVAL_SEEDS = list(range(9000, 9020))

# I require the dense (non-terminal) reward to separate a good policy from a bad one
# by at least this much. My first reward function scored 2.16 here -- below the
# per-episode noise floor (sigma ~ 2-5) -- and nothing I trained learned anything in
# ~8M env steps. This assertion exists so that cannot happen to me twice.
MIN_DENSE_GAP = 10.0


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
def _idx(env) -> dict[str, int]:
    return {name: i for i, name in enumerate(env.ACTION_NAMES)}


def _move_toward(r: int, c: int, tr: int, tc: int, A: dict[str, int]) -> int:
    if r < tr:
        return A["MOVE_N"]
    if r > tr:
        return A["MOVE_S"]
    if c < tc:
        return A["MOVE_E"]
    if c > tc:
        return A["MOVE_W"]
    return A["WAIT"]


def oracle_policy(env) -> int:
    """Greedy privileged policy: treat the worst cell, refill when low."""
    A = _idx(env)
    r, c = int(round(env.rover_row)), int(round(env.rover_col))

    # Resource management: head to the depot to refill when running low.
    if env.battery < 0.2 or env.water < 0.1 or env.pesticide < 0.1:
        if (r, c) == (0, 0):
            return A["RETURN_TO_DEPOT"]
        return _move_toward(r, c, 0, 0, A)

    pest, health = env.pest_grid, env.health_grid

    # 1) Knock down the worst pest hotspot.
    if pest.max() > 0.15:
        tr, tc = np.unravel_index(int(np.argmax(pest)), pest.shape)
        return A["SPRAY"] if (r, c) == (tr, tc) else _move_toward(r, c, tr, tc, A)

    # 2) Otherwise irrigate the lowest-health cell that isn't already watered.
    mask = (health < 0.7) & (env.irrigation_grid == 0)
    if mask.any():
        masked = np.where(mask, health, np.inf)
        tr, tc = np.unravel_index(int(np.argmin(masked)), health.shape)
        return A["IRRIGATE"] if (r, c) == (tr, tc) else _move_toward(r, c, tr, tc, A)

    return A["WAIT"]


def wait_policy(env) -> int:
    return _idx(env)["WAIT"]


def make_random_policy(seed: int = 1234):
    rng = np.random.default_rng(seed)

    def policy(env) -> int:
        return int(rng.integers(0, env.action_space.n))

    return policy


# --------------------------------------------------------------------------- #
# Rollouts
# --------------------------------------------------------------------------- #
def run_policy(policy_fn, seeds=EVAL_SEEDS) -> dict[str, float]:
    env = AgriScoutEnv()
    rewards, dense, successes, healths = [], [], [], []
    for seed in seeds:
        env.reset(seed=int(seed))
        done = False
        ep_r = 0.0
        while not done:
            obs_reward = env.step(policy_fn(env))
            _, r, term, trunc, _ = obs_reward
            ep_r += r
            done = term or trunc
        won = env.is_success
        rewards.append(ep_r)
        # Dense reward = everything the agent earns BEFORE the terminal bonus.
        # This is the part that actually carries a learning gradient.
        dense.append(ep_r - (SUCCESS_BONUS if won else 0.0))
        successes.append(1.0 if won else 0.0)
        healths.append(env.mean_health)
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_dense_reward": float(np.mean(dense)),
        "success_rate": float(np.mean(successes)),
        "mean_final_health": float(np.mean(healths)),
    }


def benchmark() -> dict[str, dict[str, float]]:
    return {
        "random": run_policy(make_random_policy()),
        "wait": run_policy(wait_policy),
        "oracle": run_policy(oracle_policy),
    }


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_check_env():
    from stable_baselines3.common.env_checker import check_env

    check_env(AgriScoutEnv(), warn=True)


def test_env_is_winnable():
    b = benchmark()
    _print_table(b)
    rnd, wait, orc = b["random"], b["wait"], b["oracle"]

    assert orc["mean_reward"] > rnd["mean_reward"] + 15, (
        f"oracle {orc['mean_reward']:.2f} not > random {rnd['mean_reward']:.2f} + 15"
    )
    assert orc["success_rate"] >= 0.6, f"oracle success_rate {orc['success_rate']:.2f} < 0.60"
    # Success must be discriminative: a random policy should mostly FAIL.
    assert rnd["success_rate"] <= 0.30, (
        f"random success_rate {rnd['success_rate']:.2f} > 0.30 (task too easy / not discriminative)"
    )
    assert orc["mean_reward"] - wait["mean_reward"] > 5, (
        f"oracle {orc['mean_reward']:.2f} not clearly above WAIT {wait['mean_reward']:.2f}"
    )


def test_dense_reward_is_learnable():
    """My reward has to carry signal BEFORE the terminal success bonus.

    An agent cannot learn from a bonus it has never once received. If nearly all of
    the oracle's advantage sits in a terminal all-or-nothing bonus, the task is
    unlearnable no matter how good the algorithm or the hyperparameters -- which is
    exactly what my first reward did (dense gap 2.16, 0 successes in 44 runs).
    """
    b = benchmark()
    _print_table(b)
    orc, rnd, wait = b["oracle"], b["random"], b["wait"]

    gap_rnd = orc["mean_dense_reward"] - rnd["mean_dense_reward"]
    gap_wait = orc["mean_dense_reward"] - wait["mean_dense_reward"]
    print(f"\ndense gap: oracle-random={gap_rnd:.2f}  oracle-wait={gap_wait:.2f} "
          f"(need >= {MIN_DENSE_GAP})")

    assert gap_rnd >= MIN_DENSE_GAP, (
        f"dense oracle-random gap {gap_rnd:.2f} < {MIN_DENSE_GAP}: the reward is "
        f"dominated by the terminal bonus and will not be learnable"
    )
    assert gap_wait >= MIN_DENSE_GAP, (
        f"dense oracle-wait gap {gap_wait:.2f} < {MIN_DENSE_GAP}: doing nothing is "
        f"nearly as good as working -- no gradient toward the task"
    )
    # The bonus must be a tiebreaker on top of dense signal, not the whole signal.
    frac = SUCCESS_BONUS / (orc["mean_reward"] - rnd["mean_reward"])
    print(f"terminal bonus is {frac:.0%} of the oracle-vs-random gap (need < 50%)")
    assert frac < 0.5, f"terminal bonus is {frac:.0%} of the total gap -- too sparse"


def test_shaping_beats_time_cost():
    """Moving one step toward work has to be net-POSITIVE.

    In my first version I scaled the navigation shaping against the per-EPISODE time
    cost instead of the per-STEP one, so approaching work actually scored
    -0.015/step -- the term was penalising the exact behaviour I had added it to
    encourage. This test pins the sign down.
    """
    from environment.agriscout_env import (
        POT_NAV_COEF, SHAPING_GAMMA, TIME_COST,
    )

    env = AgriScoutEnv()
    # One step closer shortens the normalized distance by 1/(n_rows + n_cols).
    per_step = POT_NAV_COEF * SHAPING_GAMMA / (env.n_rows + env.n_cols)
    net = per_step - TIME_COST
    print(f"\napproach step: shaping={per_step:+.4f} time={-TIME_COST:+.4f} net={net:+.4f}")
    assert net > 0, (
        f"approaching work nets {net:+.4f}/step -- the agent is punished for "
        f"navigating toward the task"
    )


def _print_table(b: dict[str, dict[str, float]]) -> None:
    print("\n=== throughput benchmark (20 episodes, seeds 9000-9019) ===")
    print(f"{'policy':<10}{'mean_reward':>13}{'dense':>10}{'success':>10}"
          f"{'final_health':>15}")
    for name in ("random", "wait", "oracle"):
        m = b[name]
        print(f"{name:<10}{m['mean_reward']:>13.2f}{m['mean_dense_reward']:>10.2f}"
              f"{m['success_rate']:>10.2f}{m['mean_final_health']:>15.3f}")


if __name__ == "__main__":
    _print_table(benchmark())
