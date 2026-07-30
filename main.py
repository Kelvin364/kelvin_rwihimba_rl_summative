"""AgriScout: entry point and demo CLI.

    uv run main.py                     # play ONE episode of the PPO agent with the
                                       # PyBullet GUI (falls back to headless DIRECT),
                                       # stochastic actions, verbose per-step output,
                                       # and a JSON trace written to logs/traces/.
    uv run main.py --episodes 3        # play 3 episodes
    uv run main.py --agent dqn         # play a different trained agent
    uv run main.py --deterministic     # greedy actions instead of sampled
    uv run main.py --mode evaluate     # evaluate all 4 agents (det + stochastic table)

Rendering (PyBullet) is imported lazily, ONLY in play mode; evaluate mode stays
headless/pybullet-free. AGRISCOUT_HEADLESS=1 forces DIRECT (no GUI window).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

REPO = Path(__file__).resolve().parent
MODELS = {
    "dqn": REPO / "models" / "dqn" / "dqn.zip",
    "ppo": REPO / "models" / "pg" / "ppo.zip",
    "a2c": REPO / "models" / "pg" / "a2c.zip",
    "reinforce": REPO / "models" / "pg" / "reinforce.pt",
}
EVAL_SEEDS = list(range(9000, 9020))
# GUI playback pacing: each env step is drawn as TWEEN_FRAMES interpolated frames
# spread over STEP_SECONDS, so the rover glides between cells.
TWEEN_FRAMES = 6
STEP_SECONDS = 0.12


@dataclass
class Args:
    """AgriScout demo / evaluation CLI."""

    mode: Literal["play", "evaluate"] = "play"
    """play a rendered episode, or evaluate all four trained agents."""
    agent: Literal["ppo", "dqn", "a2c", "reinforce", "oracle", "random"] = "ppo"
    """which agent to play (play mode only). 'oracle' and 'random' are the scripted
    reference policies, useful for showing what a perfect or a careless run looks
    like next to a trained one."""
    episodes: int = 1
    """number of episodes to play."""
    deterministic: bool = False
    """use greedy (argmax) actions instead of sampled/stochastic ones."""
    seed: int = 0
    """base RNG seed for the played episodes."""
    step_seconds: float = STEP_SECONDS
    """wall-clock seconds per environment step in the GUI (bigger = slower).
    At the 0.12 default a 150-step episode plays in ~18s; use ~0.5 to slow it to
    roughly 75s so it can be narrated."""
    view: Literal["pybullet", "web"] = "pybullet"
    """which renderer to use.

    'pybullet' opens a live 3D window that steps alongside the terminal output, so
    you can watch the agent and read its per-step numbers at the same time.

    'web' instead records the episode, builds my browser viewer and opens it. That
    one is the better-looking renderer (real lighting, shadows, treatment effects)
    and it can be scrubbed and replayed, but it is built after the episode has run
    rather than live."""


class _ScriptedPredictor:
    """Wrap a scripted ``policy(env) -> int`` in the SB3 ``.predict`` contract.

    Lets the demo drive the oracle and random reference policies through exactly
    the same play loop as a trained agent, with no branching at the call site.
    Needs the live env (it reads privileged state), so ``bind`` is called by
    :func:`play` once the env exists.
    """

    def __init__(self, policy) -> None:
        self._policy, self._env = policy, None

    def bind(self, env) -> None:
        self._env = env

    def predict(self, obs, deterministic: bool = True):
        return self._policy(self._env), None


def load_predictor(algo: str):
    """Load an agent as a ``.predict(obs, deterministic=...)`` predictor."""
    if algo in ("oracle", "random"):
        if algo == "oracle":
            from tests.test_oracle import oracle_policy

            return _ScriptedPredictor(oracle_policy)
        from tests.test_oracle import make_random_policy

        return _ScriptedPredictor(make_random_policy())

    path = MODELS[algo]
    if not path.exists():
        raise FileNotFoundError(f"model not found: {path} (train it or check models/)")
    if algo in ("dqn", "ppo", "a2c"):
        from stable_baselines3 import A2C, DQN, PPO

        return {"dqn": DQN, "ppo": PPO, "a2c": A2C}[algo].load(path)

    # REINFORCE: reconstruct the policy MLP and load weights (hparams for the
    # network shape come from the committed finals DONE.json when available).
    import json

    import numpy as np
    import torch

    from environment.agriscout_env import make_env
    from training.reinforce import REINFORCEAgent

    hp = {"hidden_sizes": [128, 128], "baseline": "none"}
    done = REPO / "logs" / "finals" / "reinforce_best" / "DONE.json"
    if done.exists():
        hp.update(json.loads(done.read_text()).get("hparams", {}))
    env = make_env()
    agent = REINFORCEAgent(int(np.prod(env.observation_space.shape)),
                           int(env.action_space.n), hp)
    agent.load_state_dict(torch.load(path, weights_only=False))
    return agent


def _predict(predictor, obs, deterministic: bool) -> int:
    import numpy as np

    action, _ = predictor.predict(obs, deterministic=deterministic)
    return int(np.asarray(action).reshape(-1)[0])


# --------------------------------------------------------------------------- #
# play mode (rendered)
# --------------------------------------------------------------------------- #
def play(args: Args) -> None:
    from environment.agriscout_env import AgriScoutEnv
    from environment.rendering import AgriScoutRenderer, EpisodeRecorder  # lazy: pulls pybullet

    predictor = load_predictor(args.agent)
    env = AgriScoutEnv()
    if isinstance(predictor, _ScriptedPredictor):
        predictor.bind(env)
    renderer = AgriScoutRenderer(env, mode="human")  # GUI -> DIRECT fallback inside
    mode_str = "GUI" if renderer.gui else "DIRECT (headless)"
    style = "deterministic" if args.deterministic else "stochastic"
    print(f"Playing {args.episodes} episode(s) with agent '{args.agent}' "
          f"[{style} actions], PyBullet {mode_str}\n")

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            recorder = EpisodeRecorder(
                run_id=f"demo_{args.agent}_ep{ep}_seed{args.seed + ep}",
                meta={"env_version": env.ENV_VERSION, "model": args.agent, "seed": args.seed + ep},
            )
            renderer.render()
            done = False
            print(f"--- episode {ep} (seed {args.seed + ep}) ---")
            while not done:
                action = _predict(predictor, obs, args.deterministic)
                obs, reward, term, trunc, _ = env.step(action)
                if renderer.gui:
                    # Sweep alpha 0 -> 1 so the rover glides between cells rather
                    # than teleporting. Headless runs draw a single committed frame.
                    for i in range(1, TWEEN_FRAMES + 1):
                        renderer.render(i / TWEEN_FRAMES)
                        time.sleep(args.step_seconds / TWEEN_FRAMES)
                else:
                    renderer.render()
                recorder.record(env, action, reward)
                print(f"  step {env.step_count:3d} | {env.ACTION_NAMES[action]:16s} "
                      f"| r={reward:+6.2f} | cum={env.cum_reward:+7.2f} "
                      f"| batt={env.battery:.2f} water={env.water:.2f} "
                      f"health={env.mean_health:.3f}")
                done = term or trunc
            trace = recorder.save()
            print(f"  -> episode {ep}: return={env.cum_reward:+.2f} "
                  f"success={env.is_success} final_health={env.mean_health:.3f}")
            print(f"  -> trace: {trace}\n")
    finally:
        renderer.close()


# --------------------------------------------------------------------------- #
# evaluate mode (headless, all agents)
# --------------------------------------------------------------------------- #
def evaluate_all() -> None:
    from tests.test_oracle import make_random_policy, oracle_policy, run_policy
    from training.sweep import evaluate  # headless; no pybullet

    print(f"Evaluating all 4 agents on held-out seeds {EVAL_SEEDS[0]}-{EVAL_SEEDS[-1]}\n")
    print(f"{'agent':12s}{'det reward':>12}{'stoch reward':>14}{'det succ':>10}"
          f"{'stoch succ':>12}{'health':>9}")
    print("-" * 69)
    for algo in ("ppo", "a2c", "dqn", "reinforce"):
        predictor = load_predictor(algo)
        det = evaluate(predictor, EVAL_SEEDS, deterministic=True)
        sto = evaluate(predictor, EVAL_SEEDS, deterministic=False)
        print(f"{algo:12s}{det['mean_reward']:>12.2f}{sto['mean_reward']:>14.2f}"
              f"{det['success_rate']:>10.2f}{sto['success_rate']:>12.2f}"
              f"{det['mean_final_health']:>9.3f}")

    # I compute the reference policies here rather than printing stored numbers.
    # When these were hard-coded, they survived a change to the reward function and
    # I ended up printing the old scale underneath the new results.
    print("-" * 69)
    for name, policy in (("random", make_random_policy()), ("oracle", oracle_policy)):
        m = run_policy(policy, EVAL_SEEDS)
        print(f"{name:12s}{m['mean_reward']:>12.2f}{'':>14}{m['success_rate']:>10.2f}"
              f"{'':>12}{m['mean_final_health']:>9.3f}")
    print("\n(random and oracle are the 0% / 100% reference points for these seeds)")


def play_web(args: Args) -> None:
    """Record the episode(s), build the browser viewer and open it.

    No pybullet involved. This is the renderer I use for the write-up: it has real
    lighting and shadows, shows a treatment effect at the treated cell, and can be
    scrubbed back and forth, at the cost of being built after the run rather than
    streamed live.
    """
    import subprocess
    import sys

    from scripts.make_demo_html import build_episode, render_html
    from scripts.record_episode import record

    traces = [record(args.agent, args.seed + ep) for ep in range(args.episodes)]
    # Write to its own file rather than index.html: that one is the curated
    # multi-agent viewer the README links to, and a single-agent run should not
    # quietly replace it.
    out = REPO / "assets" / "demo" / "last_run.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html([build_episode(t) for t in traces]))
    print(f"\n  -> viewer: {out}")

    opener = {"darwin": "open", "win32": "start"}.get(sys.platform, "xdg-open")
    try:
        subprocess.run([opener, str(out)], check=False)
        print("  -> opened in your browser (press Play, drag to orbit)")
    except FileNotFoundError:
        print(f"  -> open it manually: {out}")


def main() -> None:
    args = tyro.cli(Args)
    if args.mode == "evaluate":
        evaluate_all()
    elif args.view == "web":
        play_web(args)
    else:
        play(args)


if __name__ == "__main__":
    main()
