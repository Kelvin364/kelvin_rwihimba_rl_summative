"""Record an episode to a JSON trace, headless, with no pybullet and no display.

Feeds ``scripts/make_demo_html.py``. Works for any trained agent plus the two
scripted reference policies, so the demo can show a trained agent next to the
oracle it is being measured against.

    uv run python scripts/record_episode.py --agent ppo --seed 7
    uv run python scripts/record_episode.py --agent oracle --seed 7
    uv run python scripts/record_episode.py --agent all --seed 7   # every agent
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from environment.agriscout_env import AgriScoutEnv
from environment.trace import EpisodeRecorder

TRAINED = ("ppo", "dqn", "a2c", "reinforce")
SCRIPTED = ("oracle", "random")


def _policy(agent: str, seed: int):
    """Return a ``fn(env, obs) -> int`` action selector for `agent`."""
    if agent == "oracle":
        from tests.test_oracle import oracle_policy

        return lambda env, obs: oracle_policy(env)
    if agent == "random":
        rng = np.random.default_rng(seed)
        return lambda env, obs: int(rng.integers(0, env.action_space.n))

    from main import load_predictor

    predictor = load_predictor(agent)

    def act(env, obs):
        action, _ = predictor.predict(obs, deterministic=False)
        return int(np.asarray(action).reshape(-1)[0])

    return act


def record(agent: str, seed: int, out_dir: Path | None = None) -> Path:
    env = AgriScoutEnv()
    policy = _policy(agent, seed)
    obs, _ = env.reset(seed=seed)
    recorder = EpisodeRecorder(
        run_id=f"demo_{agent}_seed{seed}",
        meta={
            "env_version": env.ENV_VERSION,
            "model": agent,
            "seed": seed,
            "n_rows": env.n_rows,
            "n_cols": env.n_cols,
        },
    )
    done = False
    while not done:
        action = policy(env, obs)
        obs, reward, term, trunc, _ = env.step(action)
        recorder.record(env, action, reward)
        done = term or trunc

    recorder.meta.update({
        "success": bool(env.is_success),
        "final_health": round(env.mean_health, 4),
        "final_pest": round(env.mean_pest, 4),
        "total_reward": round(env.cum_reward, 3),
        "steps": int(env.step_count),
    })
    path = recorder.save(results_dir=out_dir)
    print(
        f"{agent:10s} seed={seed}  reward={env.cum_reward:+7.2f}  "
        f"success={env.is_success}  health={env.mean_health:.3f}  -> {path}"
    )
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agent", choices=[*TRAINED, *SCRIPTED, "all"], default="ppo")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--all", action="store_true",
                    help="record every trained agent plus both scripted references")
    args = ap.parse_args()

    agents = list(TRAINED + SCRIPTED) if (args.all or args.agent == "all") else [args.agent]
    for agent in agents:
        record(agent, args.seed)


if __name__ == "__main__":
    main()
