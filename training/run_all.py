"""Orchestrator CLI: sweeps -> finals -> generalization.

    uv run python -m training.run_all --phase {sweeps,finals,all}

Fully automatic (no human input). The sweeps phase uses a cost-weighted dispatcher:
DQN/REINFORCE cost 1, PPO/A2C cost 5, admitting runs while the in-flight cost stays
within AGRISCOUT_WORKERS. A one-line progress table prints every 30s.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.configs import ALGOS, FINAL_STEPS, core_cost, get_sweep_configs
from training.sweep import results_root, run_experiment


# ---------------------------------------------------------------------------
# Worker budget
# ---------------------------------------------------------------------------
def physical_cores() -> int:
    try:
        if platform.system() == "Darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).strip())
        if platform.system() == "Linux":
            out = subprocess.check_output(["lscpu", "-p=Core,Socket"], text=True)
            cores = {ln for ln in out.splitlines() if ln and not ln.startswith("#")}
            if cores:
                return len(cores)
    except Exception:
        pass
    return os.cpu_count() or 2


def worker_budget() -> int:
    env = os.environ.get("AGRISCOUT_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, physical_cores() - 1)


# ---------------------------------------------------------------------------
# Sweeps: cost-weighted dispatcher
# ---------------------------------------------------------------------------
def _run_dir(algo: str, run_id: str) -> Path:
    return results_root() / "sweeps" / algo / run_id


def _is_done(cfg: dict) -> bool:
    return (_run_dir(cfg["algo"], cfg["run_id"]) / "DONE.json").exists()


def _worker(cfg: dict) -> dict:
    # Runs in a child process; sweep.py sets thread pinning on import.
    return run_experiment(cfg["algo"], cfg["run_id"], cfg["hparams"], cfg["total_steps"])


def run_sweeps() -> None:
    budget = worker_budget()
    configs = get_sweep_configs()
    pending = [c for c in configs if not _is_done(c)]
    skipped = len(configs) - len(pending)
    print(
        f"[sweeps] {len(configs)} configs, {skipped} already DONE, "
        f"{len(pending)} to run; worker budget={budget}",
        flush=True,
    )
    if not pending:
        print("[sweeps] nothing to do.", flush=True)
        return

    completed = 0
    failed = 0
    current_cost = 0
    inflight: dict = {}          # future -> (cost, cfg)
    last_print = 0.0

    with ProcessPoolExecutor(max_workers=budget) as ex:
        def admit() -> None:
            nonlocal current_cost
            i = 0
            while i < len(pending):
                cfg = pending[i]
                cost = core_cost(cfg["algo"])
                if current_cost + cost <= budget:
                    pending.pop(i)
                    fut = ex.submit(_worker, cfg)
                    inflight[fut] = (cost, cfg)
                    current_cost += cost
                else:
                    i += 1
            # Nothing fits but nothing is running: force the smallest-cost run.
            if not inflight and pending:
                pending.sort(key=lambda c: core_cost(c["algo"]))
                cfg = pending.pop(0)
                cost = core_cost(cfg["algo"])
                fut = ex.submit(_worker, cfg)
                inflight[fut] = (cost, cfg)
                current_cost += cost

        def progress_line() -> None:
            running = ", ".join(sorted(c["run_id"] for _, c in inflight.values()))
            print(
                f"[sweeps] done={completed} failed={failed} "
                f"inflight={len(inflight)}(cost {current_cost}/{budget}) "
                f"pending={len(pending)} | running: {running}",
                flush=True,
            )

        admit()
        last_print = time.monotonic()
        while inflight:
            done, _ = wait(inflight, timeout=30, return_when=FIRST_COMPLETED)
            now = time.monotonic()
            if not done or now - last_print >= 30:
                progress_line()
                last_print = now
            for fut in done:
                cost, cfg = inflight.pop(fut)
                current_cost -= cost
                try:
                    res = fut.result()
                    completed += 1
                    print(
                        f"[sweeps] finished {cfg['run_id']}: "
                        f"mean_reward={res.get('mean_reward'):.2f}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    print(f"[sweeps] FAILED {cfg['run_id']}: {exc!r}", flush=True)
            admit()

    print(f"[sweeps] complete: {completed} done, {failed} failed.", flush=True)


# ---------------------------------------------------------------------------
# Finals: best config per algo, retrained at FINAL_STEPS
# ---------------------------------------------------------------------------
def _run_score(payload: dict) -> float:
    """Rank a sweep run by its BETTER action-selection mode.

    Ranking on the deterministic reward alone silently mis-selects: a stochastic
    policy whose argmax is a near-tie can score far worse greedily than when
    sampled (measured REINFORCE: -20.4 deterministic vs -4.6 stochastic on the same
    weights). Selecting the finals config on the deterministic number would then
    promote a config chosen by how bad its argmax is. Both modes are reported in
    the finals, so both count here.
    """
    return max(
        float(payload.get("mean_reward", -1e18)),
        float(payload.get("mean_reward_stochastic", -1e18)),
    )


def _best_run_for_algo(algo: str) -> dict | None:
    """Return the best sweep DONE.json payload for an algo (see :func:`_run_score`)."""
    algo_dir = results_root() / "sweeps" / algo
    best = None
    for done in algo_dir.glob("*/DONE.json"):
        payload = json.loads(done.read_text())
        if best is None or _run_score(payload) > _run_score(best):
            best = payload
    return best


def run_finals() -> None:
    for algo in ALGOS:
        best = _best_run_for_algo(algo)
        if best is None:
            print(f"[finals] {algo}: no completed sweep runs found, skipping.", flush=True)
            continue
        out_dir = results_root() / "finals" / f"{algo}_best"
        print(
            f"[finals] {algo}: best sweep run {best['run_id']} "
            f"(det={best.get('mean_reward', float('nan')):.2f}, "
            f"stoch={best.get('mean_reward_stochastic', float('nan')):.2f}) "
            f"-> retrain {FINAL_STEPS} steps",
            flush=True,
        )
        res = run_experiment(
            algo, f"{algo}_best", best["hparams"], FINAL_STEPS, out_dir=out_dir
        )
        print(f"[finals] {algo}: done, mean_reward={res['mean_reward']:.2f}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Publish: copy the finals into models/ so the demo loads the trained agents
# ---------------------------------------------------------------------------
# main.py loads these fixed paths. I was copying the finals across by hand, which let
# models/ drift out of step with logs/finals/ -- after I changed the observation space
# it was still holding weights that no longer loaded. So publishing is a pipeline step.
_MODEL_TARGETS = {
    "dqn": ("dqn/dqn.zip", "model.zip"),
    "ppo": ("pg/ppo.zip", "model.zip"),
    "a2c": ("pg/a2c.zip", "model.zip"),
    "reinforce": ("pg/reinforce.pt", "model.pt"),
}


def publish_models() -> None:
    import shutil

    for algo, (dest_rel, src_name) in _MODEL_TARGETS.items():
        src = results_root() / "finals" / f"{algo}_best" / src_name
        if not src.exists():
            print(f"[publish] {algo}: no final model at {src}, skipping.", flush=True)
            continue
        dest = _REPO_ROOT / "models" / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"[publish] {algo}: {src} -> {dest.relative_to(_REPO_ROOT)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="AgriScout training orchestrator")
    parser.add_argument(
        "--phase",
        choices=["sweeps", "finals", "generalization", "publish", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.phase in ("sweeps", "all"):
        run_sweeps()
    if args.phase in ("finals", "all"):
        run_finals()
    if args.phase in ("generalization", "all"):
        from training.generalization import run_generalization

        run_generalization()
    if args.phase in ("publish", "all"):
        publish_models()


if __name__ == "__main__":
    main()
