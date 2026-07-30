"""DQN (value-based) training entry point.

The shared machinery (vectorised envs, checkpointing, resume, evaluation and
`DONE.json`) lives in :mod:`training.sweep`, so DQN and the policy-gradient methods
are trained and scored by *identical* code and the comparison stays fair. This module
holds only what is DQN-specific: its hyperparameter grid and its CLI.

    uv run python -m training.dqn_training --sweep          # all 10 configurations
    uv run python -m training.dqn_training --final          # best config @ 400k steps
    uv run python -m training.dqn_training --run dqn_04     # one configuration
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.configs import FINAL_STEPS, get_sweep_configs
from training.sweep import results_root, run_experiment

ALGO = "dqn"


def configs() -> list[dict]:
    """The 10 DQN sweep configurations (see training/configs.py for the grid)."""
    return [c for c in get_sweep_configs() if c["algo"] == ALGO]


def run_sweep() -> None:
    for cfg in configs():
        res = run_experiment(ALGO, cfg["run_id"], cfg["hparams"], cfg["total_steps"])
        print(f"[dqn] {cfg['run_id']}: det={res['mean_reward']:.2f} "
              f"stoch={res['mean_reward_stochastic']:.2f}", flush=True)


def run_final() -> None:
    from training.run_all import _best_run_for_algo

    best = _best_run_for_algo(ALGO)
    if best is None:
        raise SystemExit("no completed DQN sweep runs; run --sweep first")
    out = results_root() / "finals" / f"{ALGO}_best"
    print(f"[dqn] finals from {best['run_id']} -> {FINAL_STEPS} steps", flush=True)
    res = run_experiment(ALGO, f"{ALGO}_best", best["hparams"], FINAL_STEPS, out_dir=out)
    print(f"[dqn] final: det={res['mean_reward']:.2f} "
          f"stoch={res['mean_reward_stochastic']:.2f} "
          f"success={res['success_rate_stochastic']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sweep", action="store_true", help="run all 10 configurations")
    g.add_argument("--final", action="store_true", help="retrain the best at 400k steps")
    g.add_argument("--run", metavar="RUN_ID", help="run a single configuration")
    args = ap.parse_args()

    if args.sweep:
        run_sweep()
    elif args.final:
        run_final()
    else:
        cfg = next((c for c in configs() if c["run_id"] == args.run), None)
        if cfg is None:
            raise SystemExit(f"unknown run {args.run!r}; "
                             f"choose from {[c['run_id'] for c in configs()]}")
        run_experiment(ALGO, cfg["run_id"], cfg["hparams"], cfg["total_steps"])


if __name__ == "__main__":
    main()
