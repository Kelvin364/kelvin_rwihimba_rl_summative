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


def _take(prod, n):
    return list(itertools.islice(prod, n))


def _dqn_configs() -> list[dict[str, Any]]:
    combos = _take(itertools.product([1e-3, 5e-4, 3e-4], [0.98, 0.99], [0.1, 0.2]), 10)
    out = []
    for i, (lr, gamma, expf) in enumerate(combos):
        out.append({
            "algo": "dqn",
            "run_id": f"dqn_{i:02d}",
            "total_steps": SWEEP_STEPS,
            "hparams": {
                "learning_rate": lr,
                "gamma": gamma,
                "buffer_size": 50_000,
                "learning_starts": 1_000,
                "batch_size": 64,
                "train_freq": 4,
                "target_update_interval": 1_000,
                "exploration_fraction": expf,
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
    combos = _take(itertools.product([1e-3, 5e-4], [0.99, 0.995], ["none", "mean", "value"]), 10)
    out = []
    for i, (lr, gamma, baseline) in enumerate(combos):
        out.append({
            "algo": "reinforce",
            "run_id": f"reinforce_{i:02d}",
            "total_steps": SWEEP_STEPS,
            "hparams": {
                "learning_rate": lr,
                "gamma": gamma,
                "baseline": baseline,
                "ent_coef": 0.01,
                "hidden_sizes": [128, 128],
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
