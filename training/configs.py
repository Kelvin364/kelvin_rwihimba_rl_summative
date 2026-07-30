"""Hyper-parameter sweep configs: 40 runs (10 per algorithm).

Each config is ``{"algo", "run_id", "hparams", "total_steps"}``. The sweep budget
is fixed (``SWEEP_STEPS``) so all algorithms are compared under the same
environment-step budget; the smoke test overrides ``total_steps`` directly.
"""

from __future__ import annotations

import itertools
from typing import Any

SWEEP_STEPS = 100_000  # sweeps only need to RANK configs, not fully converge
N_ENVS = 4  # for PPO / A2C (SubprocVecEnv)

# The sweep's job is to rank configurations WITHIN one algorithm; the fair
# cross-algorithm comparison happens in the finals, where every algorithm gets the
# same FINAL_STEPS budget. Vanilla REINFORCE performs ~500 gradient updates per
# 150k env steps against PPO's ~23,000, so at 100k steps its runs are still
# indistinguishable noise and ranking them would be meaningless. It gets a longer
# sweep budget so the ranking measures the hyper-parameters rather than the seed.
SWEEP_STEPS_BY_ALGO = {"reinforce": 250_000}


def sweep_steps(algo: str) -> int:
    return SWEEP_STEPS_BY_ALGO.get(algo, SWEEP_STEPS)


def _take(prod, n):
    return list(itertools.islice(prod, n))


def _dqn_configs() -> list[dict[str, Any]]:
    # I vary learning rate, discount and REPLAY BUFFER SIZE -- the three levers I
    # found matter most for a value-based learner's stability here. Buffer size took
    # the slot I first gave to exploration_fraction: when I swept that, it made no
    # measurable difference (0.1 -> -5.75 vs 0.2 -> -5.61 mean reward). My shaped
    # reward already gives a dense gradient, so the epsilon schedule is not the
    # constraint on this agent.
    combos = _take(itertools.product(
        [1e-3, 5e-4, 3e-4], [0.98, 0.99], [50_000, 200_000]), 10)
    out = []
    for i, (lr, gamma, buf) in enumerate(combos):
        out.append({
            "algo": "dqn",
            "run_id": f"dqn_{i:02d}",
            "total_steps": SWEEP_STEPS,
            "hparams": {
                "learning_rate": lr,
                "gamma": gamma,
                "buffer_size": buf,
                "learning_starts": 1_000,
                "batch_size": 64,
                "train_freq": 4,
                "target_update_interval": 1_000,
                "exploration_fraction": 0.2,
                "exploration_final_eps": 0.05,
                "seed": i,
            },
        })
    return out


def _ppo_configs() -> list[dict[str, Any]]:
    # entropy grid spans {0.0, 0.01, 0.05}: ent=0.0 demonstrably collapses to a
    # single action (a report finding); 0.01/0.05 provide the exploration fix.
    combos = _take(itertools.product([3e-4, 1e-3], [256, 512], [0.01, 0.05, 0.0]), 10)
    out = []
    for i, (lr, n_steps, ent) in enumerate(combos):
        out.append({
            "algo": "ppo",
            "run_id": f"ppo_{i:02d}",
            "total_steps": SWEEP_STEPS,
            "hparams": {
                "learning_rate": lr,
                "n_steps": n_steps,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": ent,
                "n_envs": N_ENVS,
                "seed": i,
            },
        })
    return out


def _a2c_configs() -> list[dict[str, Any]]:
    # entropy grid spans {0.0, 0.01, 0.05} (see PPO note); ent=0.01 is the baseline.
    combos = _take(itertools.product([7e-4, 3e-4], [5, 16], [0.01, 0.05, 0.0]), 10)
    out = []
    for i, (lr, n_steps, ent) in enumerate(combos):
        out.append({
            "algo": "a2c",
            "run_id": f"a2c_{i:02d}",
            "total_steps": SWEEP_STEPS,
            "hparams": {
                "learning_rate": lr,
                "n_steps": n_steps,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "ent_coef": ent,
                "n_envs": N_ENVS,
                "seed": i,
            },
        })
    return out


def _reinforce_configs() -> list[dict[str, Any]]:
    # Grid spans the three variables that measurably mattered: the baseline (the
    # variance-reduction question this algorithm exists to demonstrate), the
    # learning rate, and how many episodes are averaged into one gradient step.
    # `episodes_per_batch` trades gradient variance against update count and was
    # decisive -- at 1 episode/update the policy sat at entropy 1.95 of a 2.20
    # maximum after 150k steps, i.e. still essentially uniform.
    combos = _take(itertools.product(
        [1e-3, 5e-4], ["value", "mean", "none"], [2, 4]), 10)
    out = []
    for i, (lr, baseline, epb) in enumerate(combos):
        out.append({
            "algo": "reinforce",
            "run_id": f"reinforce_{i:02d}",
            "total_steps": sweep_steps("reinforce"),
            "hparams": {
                "learning_rate": lr,
                "gamma": 0.99,
                "baseline": baseline,
                "ent_coef": 0.01,
                "hidden_sizes": [128, 128],
                "episodes_per_batch": epb,
                # Scale-only advantage rescaling: divides by std, never re-centres,
                # so `baseline` stays the real experimental variable.
                "normalize_advantage": True,
                # Without clipping every lr >= 3e-3 drove entropy to 0 within a few
                # updates and froze the policy on a single action.
                "max_grad_norm": 0.5,
                # The critic needs its own optimizer and several steps per batch;
                # fitted jointly at the policy's lr its loss never dropped below
                # ~15, so `returns - V` was still effectively raw returns.
                "value_lr": 3e-3,
                "value_epochs": 5,
                "seed": i,
            },
        })
    return out


def get_sweep_configs() -> list[dict[str, Any]]:
    configs = _dqn_configs() + _ppo_configs() + _a2c_configs() + _reinforce_configs()
    assert len(configs) == 40, f"expected 40 configs, got {len(configs)}"
    return configs


ALGOS = ["dqn", "ppo", "a2c", "reinforce"]
FINAL_STEPS = 400_000  # finals produce the report curves and the demo agent


def core_cost(algo: str) -> int:
    """Cost-weight for the dispatcher: PPO/A2C use a SubprocVecEnv (heavier)."""
    return 5 if algo in ("ppo", "a2c") else 1
