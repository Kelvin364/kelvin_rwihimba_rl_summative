"""Policy-gradient training entry point for REINFORCE, PPO and A2C.

Shares the exact training/evaluation machinery used by DQN (:mod:`training.sweep`),
so value-based and policy-gradient results are directly comparable: same environment,
same step budget accounting, same held-out eval seeds, same metrics.

PPO and A2C come from Stable-Baselines3; REINFORCE is implemented from scratch in
:mod:`training.reinforce` (Monte-Carlo returns, three baseline modes, episode
batching, a separately-optimised critic and gradient clipping).

    uv run python -m training.pg_training --sweep              # all 30 PG configs
    uv run python -m training.pg_training --sweep --algo ppo   # just PPO's 10
    uv run python -m training.pg_training --final              # best of each @ 400k
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

PG_ALGOS = ("reinforce", "ppo", "a2c")


def configs(algo: str | None = None) -> list[dict]:
    wanted = (algo,) if algo else PG_ALGOS
    return [c for c in get_sweep_configs() if c["algo"] in wanted]


def run_sweep(algo: str | None = None) -> None:
    for cfg in configs(algo):
        res = run_experiment(cfg["algo"], cfg["run_id"], cfg["hparams"], cfg["total_steps"])
        print(f"[pg] {cfg['run_id']}: det={res['mean_reward']:.2f} "
              f"stoch={res['mean_reward_stochastic']:.2f}", flush=True)


def run_final(algo: str | None = None) -> None:
    from training.run_all import _best_run_for_algo

    for name in ((algo,) if algo else PG_ALGOS):
        best = _best_run_for_algo(name)
        if best is None:
            print(f"[pg] {name}: no completed sweep runs, skipping", flush=True)
            continue
        out = results_root() / "finals" / f"{name}_best"
        print(f"[pg] {name}: finals from {best['run_id']} -> {FINAL_STEPS} steps",
              flush=True)
        res = run_experiment(name, f"{name}_best", best["hparams"], FINAL_STEPS,
                             out_dir=out)
        print(f"[pg] {name} final: det={res['mean_reward']:.2f} "
              f"stoch={res['mean_reward_stochastic']:.2f} "
              f"success={res['success_rate_stochastic']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--algo", choices=PG_ALGOS, help="restrict to one algorithm")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sweep", action="store_true", help="run the sweep configurations")
    g.add_argument("--final", action="store_true", help="retrain the best at 400k steps")
    args = ap.parse_args()

    run_sweep(args.algo) if args.sweep else run_final(args.algo)


if __name__ == "__main__":
    main()
