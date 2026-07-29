"""Unified training interface for all four algorithms.

``run_experiment(algo, run_id, hparams, total_steps)`` is the ONE entry point shared
by DQN / PPO / A2C (Stable-Baselines3) and REINFORCE (from scratch). The budget is
counted in ENVIRONMENT STEPS for a fair comparison. Every run is resumable via
checkpoints + a ``DONE.json`` marker and is idempotent to re-execution.
"""

from __future__ import annotations

# --- thread pinning: MUST happen before torch / numpy do their thread setup ---
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import csv
import fcntl
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# Ensure subprocess workers (SubprocVecEnv / pool) can import the project.
os.environ["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")

import numpy as np
import torch

torch.set_num_threads(1)

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

CHECKPOINT_FREQ = 25_000
EVAL_EPISODES = 20
EVAL_SEEDS = list(range(9000, 9020))
ROLLING_WINDOW = 50

_SB3_ALGOS = {"dqn": DQN, "ppo": PPO, "a2c": A2C}
_CONTROL_KEYS = {"n_envs", "seed"}  # hparam keys consumed here, not passed to the model


def results_root() -> Path:
    return Path(os.environ.get("AGRISCOUT_RESULTS", "./logs"))


def _run_dir(algo: str, run_id: str) -> Path:
    return results_root() / "sweeps" / algo / run_id


# =============================================================================
# Episode-reward logging (needed for episodes_to_converge, survives resume)
# =============================================================================
class _EpisodeRewardCallback(BaseCallback):
    """Append every finished episode's reward to ``episode_rewards.csv``."""

    def __init__(self, folder: Path) -> None:
        super().__init__()
        self.folder = folder
        self._f = None
        self._w = None
        self._episode = 0

    def _on_training_start(self) -> None:
        path = self.folder / "episode_rewards.csv"
        new = not path.exists()
        # Continue episode numbering across a resume.
        if not new:
            with open(path) as fh:
                self._episode = max(0, sum(1 for _ in fh) - 1)
        self._f = open(path, "a", newline="")
        self._w = csv.writer(self._f)
        if new:
            self._w.writerow(["episode", "total_timesteps", "reward"])

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._episode += 1
                self._w.writerow([self._episode, self.num_timesteps, round(float(ep["r"]), 4)])
                self._f.flush()
        return True

    def _on_training_end(self) -> None:
        if self._f is not None:
            self._f.close()


# =============================================================================
# SB3 training + resume
# =============================================================================
def _build_vec_env(algo: str, hparams: dict, seed: int):
    from environment.agriscout_env import make_env

    if algo == "dqn":
        return make_vec_env(make_env, n_envs=1, seed=seed, vec_env_cls=DummyVecEnv)
    n_envs = int(hparams.get("n_envs", 4))
    return make_vec_env(make_env, n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv)


def _latest_sb3_checkpoint(folder: Path) -> Path | None:
    ckpts = list((folder / "checkpoints").glob("ckpt_*_steps.zip"))
    if ckpts:
        return max(ckpts, key=lambda p: int(p.stem.split("_")[1]))
    model_zip = folder / "model.zip"
    return model_zip if model_zip.exists() else None


def _train_sb3(algo: str, folder: Path, hparams: dict, total_steps: int, seed: int):
    algo_cls = _SB3_ALGOS[algo]
    n_envs = 1 if algo == "dqn" else int(hparams.get("n_envs", 4))
    vec_env = _build_vec_env(algo, hparams, seed)
    logger = configure(str(folder), ["csv"])

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=max(1, CHECKPOINT_FREQ // n_envs),
            save_path=str(folder / "checkpoints"),
            name_prefix="ckpt",
        ),
        _EpisodeRewardCallback(folder),
    ])
    log_interval = 1

    ckpt = _latest_sb3_checkpoint(folder)
    if ckpt is not None:
        model = algo_cls.load(ckpt, env=vec_env)
        model.set_logger(logger)
        steps_done = int(model.num_timesteps)
        remaining = max(0, total_steps - steps_done)
        print(
            f"[resume] {algo}/{folder.name}: loaded {ckpt.name} at {steps_done} steps, "
            f"remaining={remaining}",
            flush=True,
        )
        if remaining > 0:
            model.learn(
                remaining, reset_num_timesteps=False,
                callback=callbacks, log_interval=log_interval,
            )
    else:
        algo_kwargs = {k: v for k, v in hparams.items() if k not in _CONTROL_KEYS}
        model = algo_cls("MlpPolicy", vec_env, seed=seed, verbose=0, **algo_kwargs)
        model.set_logger(logger)
        model.learn(total_steps, callback=callbacks, log_interval=log_interval)

    model.save(str(folder / "model.zip"))
    vec_env.close()
    return model


def _load_sb3_for_eval(algo: str, folder: Path):
    return _SB3_ALGOS[algo].load(folder / "model.zip")


# =============================================================================
# Evaluation (shared by all algorithms)
# =============================================================================
def evaluate(predictor, seeds=EVAL_SEEDS) -> dict[str, float]:
    """Roll out a greedy policy on held-out seeds; return aggregate metrics."""
    from environment.agriscout_env import make_env

    env = make_env()
    rewards, healths, waters, successes = [], [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        done = False
        ep_r = 0.0
        while not done:
            action, _ = predictor.predict(obs, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
            obs, r, term, trunc, _ = env.step(action)
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
        healths.append(env.mean_health)
        waters.append(env.water_used)
        successes.append(1.0 if env.is_success else 0.0)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "mean_final_health": float(np.mean(healths)),
        "water_used": float(np.mean(waters)),
    }


def _episodes_to_converge(folder: Path) -> int:
    """First episode whose 50-ep rolling mean reaches 80% of the final rolling mean.

    NB: interpreted literally per spec. When the final rolling mean is negative,
    the 0.8x threshold is closer to zero, so this can trigger early -- an inherent
    quirk of the definition for negative-reward regimes.
    """
    path = folder / "episode_rewards.csv"
    if not path.exists():
        return -1
    rewards: list[float] = []
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rewards.append(float(row["reward"]))
    n = len(rewards)
    if n == 0:
        return -1
    rolling = [
        float(np.mean(rewards[max(0, i - ROLLING_WINDOW + 1): i + 1]))
        for i in range(n)
    ]
    final = rolling[-1]
    threshold = 0.8 * final
    for i, val in enumerate(rolling):
        if val >= threshold:
            return i + 1  # 1-based episode index
    return n


# =============================================================================
# DONE.json + summary.csv
# =============================================================================
def _write_done(folder: Path, payload: dict) -> None:
    tmp = folder / "DONE.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(folder / "DONE.json")


def _summary_row(algo: str, run_id: str, hparams: dict, total_steps: int, metrics: dict, etc: int) -> dict:
    row = {"algo": algo, "run_id": run_id, "total_steps": total_steps}
    for k in sorted(hparams):
        row[f"hp_{k}"] = hparams[k]
    row.update(metrics)
    row["episodes_to_converge"] = etc
    return row


def _append_summary(algo: str, row: dict) -> None:
    path = results_root() / "sweeps" / algo / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)  # safe under the parallel dispatcher
        try:
            # De-dupe: drop any existing row for this run_id, then rewrite.
            existing_header, existing_rows = _read_summary(path)
            header = existing_header or list(row.keys())
            for key in row:
                if key not in header:
                    header.append(key)
            existing_rows = [r for r in existing_rows if r.get("run_id") != row["run_id"]]
            existing_rows.append({k: row.get(k, "") for k in header})
            fh.seek(0)
            fh.truncate()
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            for r in existing_rows:
                writer.writerow({k: r.get(k, "") for k in header})
            fh.flush()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _read_summary(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None, []
    with open(path) as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


# =============================================================================
# THE shared entry point
# =============================================================================
def run_experiment(
    algo: str,
    run_id: str,
    hparams: dict,
    total_steps: int,
    out_dir: str | os.PathLike | None = None,
) -> dict:
    """Train (or resume) one run to `total_steps` env steps, evaluate, and finalize.

    Idempotent: if ``DONE.json`` already exists the run is skipped and its payload
    returned. Otherwise, resumes from the latest checkpoint if present.

    ``out_dir`` overrides the output directory (used by the finals phase which
    writes to ``finals/<algo>_best/`` instead of ``sweeps/...``); when overridden
    the sweep ``summary.csv`` is not appended.
    """
    torch.set_num_threads(1)
    algo = algo.lower()
    folder = Path(out_dir) if out_dir is not None else _run_dir(algo, run_id)
    folder.mkdir(parents=True, exist_ok=True)
    done_path = folder / "DONE.json"

    if done_path.exists():
        print(f"[skip] {algo}/{run_id}: DONE.json present -> skipping", flush=True)
        return json.loads(done_path.read_text())

    seed = int(hparams.get("seed", 0))

    if algo in _SB3_ALGOS:
        model = _train_sb3(algo, folder, hparams, total_steps, seed)
        predictor = model
    elif algo == "reinforce":
        from training.reinforce import train_reinforce

        predictor, _steps, _ep = train_reinforce(
            folder, hparams, total_steps, seed=seed, checkpoint_freq=CHECKPOINT_FREQ
        )
    else:
        raise ValueError(f"Unknown algo: {algo!r}")

    metrics = evaluate(predictor, EVAL_SEEDS)
    etc = _episodes_to_converge(folder)

    payload = {
        "algo": algo,
        "run_id": run_id,
        "total_steps": total_steps,
        "hparams": hparams,
        "eval_seeds": [EVAL_SEEDS[0], EVAL_SEEDS[-1]],
        **metrics,
        "episodes_to_converge": etc,
    }
    _write_done(folder, payload)
    if out_dir is None:
        _append_summary(algo, _summary_row(algo, run_id, hparams, total_steps, metrics, etc))
    return payload
