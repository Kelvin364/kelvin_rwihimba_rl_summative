"""Generalization evaluation of the four final models.

Each final model (``finals/<algo>_best/``) is evaluated on:
  * held-out seeds 10000-10049 (NEVER used in training or tuning), and
  * 20 training-distribution seeds (0-19),
and the aggregate metrics per model per seed-set are written to
``<AGRISCOUT_RESULTS>/generalization/results.csv``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from training.configs import ALGOS
from training.sweep import evaluate, results_root

HELDOUT_SEEDS = list(range(10_000, 10_050))
TRAIN_SEEDS = list(range(0, 20))

SEED_SETS = {
    "heldout_10000-10049": HELDOUT_SEEDS,
    "train_dist_0-19": TRAIN_SEEDS,
}


def _load_predictor(algo: str, folder: Path):
    """Load a final model as a `.predict`-compatible predictor."""
    if algo in ("dqn", "ppo", "a2c"):
        from stable_baselines3 import A2C, DQN, PPO

        cls = {"dqn": DQN, "ppo": PPO, "a2c": A2C}[algo]
        return cls.load(folder / "model.zip")
    if algo == "reinforce":
        from training.reinforce import load_agent

        hparams = json.loads((folder / "DONE.json").read_text()).get("hparams", {})
        return load_agent(folder, hparams)
    raise ValueError(f"Unknown algo: {algo!r}")


def run_generalization() -> Path:
    root = results_root()
    out_dir = root / "generalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.csv"

    # Both action-selection modes are reported. Scoring generalization on the
    # deterministic policy alone understates every agent here -- their greedy argmax
    # is much worse than sampling (PPO: -10.6 vs +5.5 on the same weights) -- so a
    # deterministic-only table would describe a different policy than the one the
    # demo actually runs.
    fields = [
        "algo", "seed_set", "n_seeds",
        "mean_reward", "std_reward", "success_rate", "mean_final_health", "water_used",
        "mean_reward_stochastic", "std_reward_stochastic", "success_rate_stochastic",
        "mean_final_health_stochastic",
    ]
    rows = []
    for algo in ALGOS:
        folder = root / "finals" / f"{algo}_best"
        if not (folder / "DONE.json").exists():
            print(f"[generalization] skipping {algo}: no final model at {folder}", flush=True)
            continue
        predictor = _load_predictor(algo, folder)
        for set_name, seeds in SEED_SETS.items():
            m = evaluate(predictor, seeds, deterministic=True)
            s = evaluate(predictor, seeds, deterministic=False)
            rows.append({
                "algo": algo,
                "seed_set": set_name,
                "n_seeds": len(seeds),
                "mean_reward": round(m["mean_reward"], 4),
                "std_reward": round(m["std_reward"], 4),
                "success_rate": round(m["success_rate"], 4),
                "mean_final_health": round(m["mean_final_health"], 4),
                "water_used": round(m["water_used"], 4),
                "mean_reward_stochastic": round(s["mean_reward"], 4),
                "std_reward_stochastic": round(s["std_reward"], 4),
                "success_rate_stochastic": round(s["success_rate"], 4),
                "mean_final_health_stochastic": round(s["mean_final_health"], 4),
            })
            print(
                f"[generalization] {algo:10s} {set_name:22s} "
                f"det={m['mean_reward']:7.2f}+-{m['std_reward']:5.2f} succ={m['success_rate']:.2f} | "
                f"stoch={s['mean_reward']:7.2f}+-{s['std_reward']:5.2f} succ={s['success_rate']:.2f} "
                f"health={s['mean_final_health']:.3f}",
                flush=True,
            )

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[generalization] wrote {out_path} ({len(rows)} rows)", flush=True)
    return out_path


def main() -> None:
    run_generalization()


if __name__ == "__main__":
    main()
