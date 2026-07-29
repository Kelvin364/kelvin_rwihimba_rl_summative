"""Training-layer smoke test.

Runs every algorithm for 5,000 env steps, confirms each produces progress.csv +
DONE.json + a summary row, then proves skip/resume by deleting DONE.json files and
re-running. Uses a small checkpoint frequency so REINFORCE resumes from a genuine
mid-training checkpoint.

    uv run python scripts/smoke_train.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Keep smoke outputs isolated from real sweeps.
os.environ.setdefault("AGRISCOUT_RESULTS", str(_REPO_ROOT / "logs" / "smoke"))

STEPS = 5_000

SMOKE = {
    "dqn": {
        "learning_rate": 1e-3, "buffer_size": 10_000, "learning_starts": 500,
        "batch_size": 64, "gamma": 0.99, "train_freq": 4,
        "target_update_interval": 500, "exploration_fraction": 0.3, "seed": 0,
    },
    "ppo": {
        "learning_rate": 3e-4, "n_steps": 256, "batch_size": 64, "n_epochs": 4,
        "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.0, "n_envs": 4, "seed": 0,
    },
    "a2c": {
        "learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0,
        "ent_coef": 0.0, "n_envs": 4, "seed": 0,
    },
    "reinforce": {
        "learning_rate": 1e-3, "gamma": 0.99, "baseline": "value",
        "ent_coef": 0.01, "hidden_sizes": [128, 128], "seed": 0,
    },
}


def _show_outputs(sweep, algo: str) -> None:
    folder = sweep.results_root() / "sweeps" / algo / f"{algo}_smoke"
    files = sorted(p.name for p in folder.glob("*")) + [
        "checkpoints/" + p.name for p in (folder / "checkpoints").glob("*")
    ]
    has_prog = (folder / "progress.csv").exists()
    has_done = (folder / "DONE.json").exists()
    n_prog = sum(1 for _ in open(folder / "progress.csv")) - 1 if has_prog else 0
    print(f"  {algo:10s} progress.csv={has_prog}({n_prog} rows) DONE.json={has_done}")
    print(f"             files: {files}")


def main() -> None:
    import training.sweep as sweep
    from training.sweep import results_root, run_experiment

    # Small checkpoint freq so 5k-step runs write real mid-training checkpoints.
    sweep.CHECKPOINT_FREQ = 2_000

    print(f"AGRISCOUT_RESULTS = {os.environ['AGRISCOUT_RESULTS']}")
    print(f"\n===== PASS 1: train each algorithm for {STEPS} steps =====")
    for algo, hp in SMOKE.items():
        print(f"\n--- {algo} ---")
        res = run_experiment(algo, f"{algo}_smoke", hp, STEPS)
        print(
            f"  -> mean_reward={res['mean_reward']:.2f} "
            f"success_rate={res['success_rate']:.2f} "
            f"final_health={res['mean_final_health']:.3f} "
            f"episodes_to_converge={res['episodes_to_converge']}"
        )

    print("\n===== artifacts produced =====")
    for algo in SMOKE:
        _show_outputs(sweep, algo)

    print("\n===== summary.csv rows =====")
    for algo in SMOKE:
        sp = results_root() / "sweeps" / algo / "summary.csv"
        print(f"--- {sp} ---")
        print("   " + sp.read_text().strip().replace("\n", "\n   "))

    print(f"\n===== PASS 2: re-run all (expect every run SKIPPED) =====")
    for algo, hp in SMOKE.items():
        run_experiment(algo, f"{algo}_smoke", hp, STEPS)

    print("\n===== PASS 3: prove RESUME =====")
    # (a) SB3 (dqn): delete DONE.json, keep model.zip -> resume at 5000, remaining=0.
    dqn_dir = results_root() / "sweeps" / "dqn" / "dqn_smoke"
    (dqn_dir / "DONE.json").unlink()
    print("deleted dqn DONE.json (model.zip kept) -> re-run:")
    run_experiment("dqn", "dqn_smoke", SMOKE["dqn"], STEPS)

    # (b) REINFORCE: delete DONE.json + final model, keep a mid checkpoint -> resume + train remainder.
    r_dir = results_root() / "sweeps" / "reinforce" / "reinforce_smoke"
    (r_dir / "DONE.json").unlink()
    (r_dir / "model.pt").unlink()
    ckpts = sorted((r_dir / "checkpoints").glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    # Drop all but the earliest checkpoint so training must continue from it.
    for extra in ckpts[1:]:
        extra.unlink()
    kept = ckpts[0].name if ckpts else "(none)"
    print(f"deleted reinforce DONE.json + model.pt, kept only checkpoint {kept} -> re-run:")
    res = run_experiment("reinforce", "reinforce_smoke", SMOKE["reinforce"], STEPS)
    print(f"  -> reinforce re-completed, mean_reward={res['mean_reward']:.2f}")

    print("\n===== smoke complete =====")


if __name__ == "__main__":
    main()
