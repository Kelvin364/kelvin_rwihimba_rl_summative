"""My learnability gate: the cheap check I run BEFORE spending compute on a sweep.

``tests/test_oracle.py`` proves a *scripted* policy can win. I learned the hard way
that this is necessary but not sufficient: my first reward function passed the oracle
gate and was still unlearnable, because ~90% of the oracle's advantage sat in a single
terminal all-or-nothing bonus. I only found out after 40 sweep configs and four 400k
finals, roughly 8M environment steps, in which not one run recorded a single
success.

So I added this gate to ask the question that actually matters: does a real LEARNING
agent improve? It trains a short run and requires the trained policy to clearly beat a
random policy on held-out seeds.

Run it before committing compute to ``training.run_all``::

    uv run python scripts/learnability_gate.py                 # PPO, 150k steps
    uv run python scripts/learnability_gate.py --algo a2c
    uv run python scripts/learnability_gate.py --steps 60000 --algo dqn
    uv run python scripts/learnability_gate.py --all           # all four algorithms

Exit code is non-zero if the gate fails, so it is CI/automation friendly.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Keep gate artefacts out of the real results tree.
_GATE_ROOT = _REPO_ROOT / "logs" / "gate"
os.environ.setdefault("AGRISCOUT_RESULTS", str(_GATE_ROOT))

GATE_STEPS = 150_000
GATE_SEEDS = list(range(9000, 9020))

# I require the trained policy to beat random by at least this much. Random scores
# about -13.9 and my oracle about +14.6, so the full span is ~28.5. I set the bar at
# 8.0, a little under a third of the way, because I want it to mean "this is
# unmistakably learning", not "this has solved the task".
MIN_IMPROVEMENT = 8.0

# Reasonable defaults per algorithm (not tuned, since tuning is the sweep's job).
GATE_HPARAMS = {
    "ppo": {
        "learning_rate": 3e-4, "n_steps": 256, "batch_size": 64, "n_epochs": 10,
        "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01, "n_envs": 4, "seed": 0,
    },
    "a2c": {
        "learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0,
        "ent_coef": 0.01, "n_envs": 4, "seed": 0,
    },
    "dqn": {
        "learning_rate": 5e-4, "gamma": 0.99, "buffer_size": 50_000,
        "learning_starts": 1_000, "batch_size": 64, "train_freq": 4,
        "target_update_interval": 1_000, "exploration_fraction": 0.2,
        "exploration_final_eps": 0.05, "seed": 0,
    },
    "reinforce": {
        "learning_rate": 1e-3, "gamma": 0.99, "baseline": "value",
        "ent_coef": 0.01, "hidden_sizes": [128, 128], "seed": 0,
        "episodes_per_batch": 2, "normalize_advantage": True,
        "max_grad_norm": 0.5, "value_lr": 3e-3, "value_epochs": 5,
    },
}

# I give vanilla REINFORCE a bigger budget to clear the same bar, because its
# slowness is a real property of the algorithm and not something I should tune away.
# Per 150k env steps PPO performs ~23,000 gradient updates (n_steps 256 x 4 envs,
# 10 epochs, minibatch 64); REINFORCE performs one update per batch of complete
# episodes, about 500. When I measured it, its 400k-step curve was still climbing
# (-15.7 -> -3.2) at the point PPO had long since plateaued. Pretending 150k is a fair
# comparison would just be measuring the budget, so I give it the finals budget.
GATE_STEPS_BY_ALGO = {"reinforce": 400_000}


def random_baseline() -> dict[str, float]:
    """Score a uniform-random policy on the gate seeds (the bar to beat)."""
    from tests.test_oracle import make_random_policy, run_policy

    return run_policy(make_random_policy(), GATE_SEEDS)


def run_gate(algo: str, steps: int, fresh: bool = True) -> dict[str, object]:
    """Train `algo` for `steps` env steps and compare it against random."""
    from training.sweep import evaluate, results_root, run_experiment

    run_id = f"{algo}_gate"
    folder = results_root() / "sweeps" / algo / run_id
    if fresh and folder.exists():
        shutil.rmtree(folder)  # a stale DONE.json would silently skip training

    print(f"\n{'=' * 68}\n{algo.upper()} learnability gate, {steps:,} env steps\n{'=' * 68}",
          flush=True)
    t0 = time.time()
    run_experiment(algo, run_id, GATE_HPARAMS[algo], steps)
    train_s = time.time() - t0

    if algo == "reinforce":
        from training.reinforce import load_agent

        predictor = load_agent(folder, GATE_HPARAMS[algo])
    else:
        from stable_baselines3 import A2C, DQN, PPO

        predictor = {"ppo": PPO, "a2c": A2C, "dqn": DQN}[algo].load(folder / "model.zip")

    det = evaluate(predictor, GATE_SEEDS, deterministic=True)
    sto = evaluate(predictor, GATE_SEEDS, deterministic=False)
    # Judge on whichever action mode the policy is actually better in; a policy
    # that is only good when sampled is still a policy that learned something.
    best_mode = "deterministic" if det["mean_reward"] >= sto["mean_reward"] else "stochastic"
    best = det if best_mode == "deterministic" else sto

    return {
        "algo": algo, "steps": steps, "train_seconds": train_s,
        "det": det, "stoch": sto, "best_mode": best_mode, "best": best,
    }


def report(result: dict, rnd: dict[str, float]) -> bool:
    """Print a verdict for one algorithm; return True if it passed."""
    algo, det, sto = result["algo"], result["det"], result["stoch"]
    best, mode = result["best"], result["best_mode"]
    improvement = best["mean_reward"] - rnd["mean_reward"]
    passed = improvement >= MIN_IMPROVEMENT

    print(f"\n--- {algo.upper()} result ({result['train_seconds']:.0f}s training) ---")
    print(f"{'policy':<22}{'reward':>10}{'success':>10}{'health':>10}")
    print(f"{'random (baseline)':<22}{rnd['mean_reward']:>10.2f}"
          f"{rnd['success_rate']:>10.2f}{rnd['mean_final_health']:>10.3f}")
    print(f"{algo + ' deterministic':<22}{det['mean_reward']:>10.2f}"
          f"{det['success_rate']:>10.2f}{det['mean_final_health']:>10.3f}")
    print(f"{algo + ' stochastic':<22}{sto['mean_reward']:>10.2f}"
          f"{sto['success_rate']:>10.2f}{sto['mean_final_health']:>10.3f}")
    print(f"\nimprovement over random ({mode}): {improvement:+.2f} "
          f"(need >= {MIN_IMPROVEMENT})")
    print(f"VERDICT: {'PASS: the agent is learning' if passed else 'FAIL: no learning signal'}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--algo", choices=sorted(GATE_HPARAMS), default="ppo")
    parser.add_argument("--steps", type=int, default=None,
                        help="override the per-algorithm step budget")
    parser.add_argument("--all", action="store_true", help="run every algorithm")
    args = parser.parse_args()

    algos = sorted(GATE_HPARAMS) if args.all else [args.algo]

    def budget(algo: str) -> int:
        if args.steps is not None:
            return args.steps
        return GATE_STEPS_BY_ALGO.get(algo, GATE_STEPS)

    print(f"AGRISCOUT_RESULTS = {os.environ['AGRISCOUT_RESULTS']}")
    print("scoring the random baseline on seeds 9000-9019 ...", flush=True)
    rnd = random_baseline()
    print(f"random: reward={rnd['mean_reward']:.2f} success={rnd['success_rate']:.2f}")

    verdicts = {a: report(run_gate(a, budget(a)), rnd) for a in algos}

    print(f"\n{'=' * 68}\nGATE SUMMARY\n{'=' * 68}")
    for algo, ok in verdicts.items():
        print(f"  {algo:<12}{'PASS' if ok else 'FAIL'}")
    all_ok = all(verdicts.values())
    print(f"\n{'ALL GATES PASSED: safe to run the full sweep.' if all_ok else 'GATE FAILED: fix the reward before spending compute on the sweep.'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
