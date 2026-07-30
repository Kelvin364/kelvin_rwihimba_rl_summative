"""Single PPO validation run on the (fixed) env before the full sweep.

Trains PPO for 60k steps in an isolated results dir, then prints the learning
curve (rollout/ep_rew_mean over timesteps) and the held-out eval so we can confirm
episode reward climbs above the random baseline and eval success_rate > 0.

    uv run python scripts/validate_ppo.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("AGRISCOUT_RESULTS", str(_REPO_ROOT / "logs" / "validation"))

PPO_BASELINE = {
    "learning_rate": 3e-4, "n_steps": 512, "batch_size": 64, "n_epochs": 10,
    "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01, "n_envs": 4, "seed": 0,
}


def action_distribution_and_entropy(model, seeds=range(9000, 9005)):
    """Deterministic action histogram + mean policy entropy over visited states."""
    import collections

    import torch
    from environment.agriscout_env import AgriScoutEnv

    env = AgriScoutEnv()
    hist = collections.Counter()
    entropies = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        done = False
        while not done:
            obs_t = model.policy.obs_to_tensor(obs)[0]
            with torch.no_grad():
                entropies.append(float(model.policy.get_distribution(obs_t).entropy().mean()))
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            hist[env.ACTION_NAMES[action]] += 1
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
    total = sum(hist.values())
    dist = {k: round(v / total, 3) for k, v in hist.most_common()}
    return dist, (sum(entropies) / len(entropies) if entropies else 0.0)


def main() -> None:
    from training.sweep import results_root, run_experiment

    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 60_000
    run_id = sys.argv[2] if len(sys.argv) > 2 else "ppo_validation"

    # Compute the random baseline fresh (reward scale may change between runs).
    from tests.test_oracle import make_random_policy, run_policy

    random_baseline = run_policy(make_random_policy())["mean_reward"]

    print(f"AGRISCOUT_RESULTS = {os.environ['AGRISCOUT_RESULTS']}")
    print(f"random baseline (eval seeds 9000-9019): {random_baseline:.2f}")
    print(f"Training PPO baseline for {steps:,} steps ...\n")
    payload = run_experiment("ppo", run_id, PPO_BASELINE, steps)

    prog = results_root() / "sweeps" / "ppo" / run_id / "progress.csv"
    rows = []
    ent_curve = []  # (timesteps, policy entropy) from train/entropy_loss
    with open(prog) as fh:
        for row in csv.DictReader(fh):
            ts = row.get("time/total_timesteps")
            rew = row.get("rollout/ep_rew_mean")
            if ts and rew not in (None, ""):
                rows.append((int(ts), float(rew)))
            eloss = row.get("train/entropy_loss")
            if ts and eloss not in (None, ""):
                # SB3 PPO logs entropy_loss = -mean(entropy)  ->  entropy = -eloss
                ent_curve.append((int(ts), -float(eloss)))

    print("=== learning curve: rollout/ep_rew_mean vs timesteps ===")
    print(f"{'timesteps':>12}{'ep_rew_mean':>14}")
    # Print ~12 evenly-spaced samples.
    if rows:
        step = max(1, len(rows) // 12)
        for ts, rew in rows[::step]:
            print(f"{ts:>12}{rew:>14.2f}")
        if rows[-1] not in rows[::step]:
            ts, rew = rows[-1]
            print(f"{ts:>12}{rew:>14.2f}")
        first_rew = rows[0][1]
        last_rew = rows[-1][1]
        print(f"\nfirst ep_rew_mean : {first_rew:.2f}")
        print(f"last  ep_rew_mean : {last_rew:.2f}")
        print(f"improvement       : {last_rew - first_rew:+.2f}")
    print(f"\nrandom baseline   : {random_baseline:.2f}")

    print("\n=== held-out eval (seeds 9000-9019) ===")
    print(f"{'':18}{'deterministic':>15}{'stochastic':>14}")
    print(f"{'mean_reward':18}{payload['mean_reward']:>15.2f}"
          f"{payload['mean_reward_stochastic']:>14.2f}")
    print(f"{'success_rate':18}{payload['success_rate']:>15.2f}"
          f"{payload['success_rate_stochastic']:>14.2f}")
    print(f"{'mean_final_health':18}{payload['mean_final_health']:>15.3f}"
          f"{payload['mean_final_health_stochastic']:>14.3f}")

    from stable_baselines3 import PPO

    model = PPO.load(results_root() / "sweeps" / "ppo" / run_id / "model.zip")
    dist, mean_entropy = action_distribution_and_entropy(model)
    n_actions_used = sum(1 for v in dist.values() if v >= 0.02)
    print("\n=== learned policy behaviour (greedy) ===")
    print(f"action distribution : {dist}")
    print(f"distinct actions used (>=2%): {n_actions_used}")
    print(f"mean policy entropy : {mean_entropy:.3f}  (max = ln 9 = {2.197:.3f})")

    # Entropy still declining? Compare last quartile mean vs the prior quartile.
    entropy_declining = False
    if len(ent_curve) >= 4:
        q = len(ent_curve) // 4
        prev_q = sum(e for _, e in ent_curve[-2 * q:-q]) / max(1, q)
        last_q = sum(e for _, e in ent_curve[-q:]) / max(1, q)
        entropy_declining = last_q < prev_q
        print(f"training entropy    : first {ent_curve[0][1]:.3f} -> last {ent_curve[-1][1]:.3f} "
              f"(declining={entropy_declining})")

    # PASS = det eval >= random+10 AND det success >= 0.3 AND entropy < 1.5 and declining.
    det_reward = payload["mean_reward"]
    det_success = payload["success_rate"]
    c_reward = det_reward >= random_baseline + 10
    c_success = det_success >= 0.3
    c_entropy = mean_entropy < 1.5 and entropy_declining
    ok = c_reward and c_success and c_entropy
    print("\n=== GATE ===")
    print(f"(a) det eval reward >= random+10 ({random_baseline + 10:.2f}) : "
          f"{det_reward:.2f}  {'PASS' if c_reward else 'FAIL'}")
    print(f"(b) det success_rate >= 0.30                : "
          f"{det_success:.2f}  {'PASS' if c_success else 'FAIL'}")
    print(f"(c) entropy < 1.5 and still declining       : "
          f"{mean_entropy:.3f}/{entropy_declining}  {'PASS' if c_entropy else 'FAIL'}")
    print("\n=== VERDICT:", "PASS ===" if ok else "FAIL ===")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
