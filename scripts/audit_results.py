"""Audit copied training results under $AGRISCOUT_RESULTS (default ./logs).

Verifies sweep completeness, finals models, generalization coverage, and clean
summaries; prints tables and exits non-zero if any hard check fails.

    uv run python scripts/audit_results.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

ALGOS = ["dqn", "ppo", "a2c", "reinforce"]
EXPECTED_PER_ALGO = 10
EXPECTED_SEED_SETS = {"heldout_10000-10049", "train_dist_0-19"}


def results_root() -> Path:
    return Path(os.environ.get("AGRISCOUT_RESULTS", "./logs"))


def _bad_reward(val) -> bool:
    """True if mean_reward is missing, empty, or NaN."""
    if val is None or val == "":
        return True
    try:
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return True


def _hr(width: int = 78) -> None:
    print("-" * width)


def audit() -> int:
    root = results_root()
    issues: list[str] = []
    print(f"AGRISCOUT_RESULTS = {root.resolve()}\n")

    # ------------------------------------------------------------------ sweeps
    print("== SWEEPS ==")
    print(f"{'algo':<10}{'runs':>6}{'DONE':>6}{'progress':>10}{'bad_reward':>12}"
          f"{'summary_rows':>14}{'dupes':>7}")
    _hr()
    total_runs = 0
    for algo in ALGOS:
        algo_dir = root / "sweeps" / algo
        run_dirs = sorted(p for p in algo_dir.glob("*") if p.is_dir()) if algo_dir.exists() else []
        n_runs = len(run_dirs)
        total_runs += n_runs
        n_done = n_prog = n_bad = 0
        for rd in run_dirs:
            done = rd / "DONE.json"
            prog = rd / "progress.csv"
            has_done = done.exists()
            has_prog = prog.exists() and prog.stat().st_size > 0
            n_done += has_done
            n_prog += has_prog
            if not has_done:
                issues.append(f"sweeps/{algo}/{rd.name}: missing DONE.json")
                continue
            if not has_prog:
                issues.append(f"sweeps/{algo}/{rd.name}: missing/empty progress.csv")
            try:
                payload = json.loads(done.read_text())
            except Exception as exc:  # noqa: BLE001
                issues.append(f"sweeps/{algo}/{rd.name}: unreadable DONE.json ({exc})")
                n_bad += 1
                continue
            if _bad_reward(payload.get("mean_reward")):
                issues.append(f"sweeps/{algo}/{rd.name}: NaN/empty mean_reward")
                n_bad += 1

        # summary.csv rows + duplicate run_ids
        summ = algo_dir / "summary.csv"
        summary_rows = 0
        dupes = 0
        if summ.exists():
            with open(summ) as fh:
                rows = list(csv.DictReader(fh))
            summary_rows = len(rows)
            run_ids = [r.get("run_id") for r in rows]
            dupes = len(run_ids) - len(set(run_ids))
        else:
            issues.append(f"sweeps/{algo}/summary.csv: missing")

        if n_runs != EXPECTED_PER_ALGO:
            issues.append(f"sweeps/{algo}: {n_runs} run dirs (expected {EXPECTED_PER_ALGO})")
        if summary_rows != EXPECTED_PER_ALGO:
            issues.append(f"sweeps/{algo}/summary.csv: {summary_rows} rows (expected {EXPECTED_PER_ALGO})")
        if dupes:
            issues.append(f"sweeps/{algo}/summary.csv: {dupes} duplicate run_id(s)")

        flag = "" if (n_runs == n_done == n_prog == EXPECTED_PER_ALGO and n_bad == 0
                      and summary_rows == EXPECTED_PER_ALGO and dupes == 0) else "  <-- CHECK"
        print(f"{algo:<10}{n_runs:>6}{n_done:>6}{n_prog:>10}{n_bad:>12}"
              f"{summary_rows:>14}{dupes:>7}{flag}")
    _hr()
    print(f"total sweep run dirs: {total_runs} (expected {EXPECTED_PER_ALGO * len(ALGOS)})\n")

    # ------------------------------------------------------------------ finals
    print("== FINALS ==")
    print(f"{'dir':<20}{'DONE.json':>11}{'model_file':>14}")
    _hr()
    for algo in ALGOS:
        fdir = root / "finals" / f"{algo}_best"
        has_done = (fdir / "DONE.json").exists()
        model = next((m.name for m in (fdir / "model.zip", fdir / "model.pt") if m.exists()), None)
        if not fdir.exists():
            issues.append(f"finals/{algo}_best: directory missing")
        else:
            if not has_done:
                issues.append(f"finals/{algo}_best: missing DONE.json")
            if model is None:
                issues.append(f"finals/{algo}_best: missing model.zip/model.pt")
        print(f"{algo + '_best':<20}{('yes' if has_done else 'NO'):>11}{(model or 'MISSING'):>14}")
    print()

    # ---------------------------------------------------------- generalization
    print("== GENERALIZATION ==")
    gen = root / "generalization" / "results.csv"
    if not gen.exists():
        issues.append("generalization/results.csv: missing")
        print("results.csv: MISSING")
    else:
        with open(gen) as fh:
            grows = list(csv.DictReader(fh))
        present = {(r["algo"], r["seed_set"]) for r in grows}
        print(f"{'algo':<12}" + "".join(f"{s:>26}" for s in sorted(EXPECTED_SEED_SETS)))
        _hr()
        for algo in ALGOS:
            cells = []
            for s in sorted(EXPECTED_SEED_SETS):
                ok = (algo, s) in present
                cells.append("yes" if ok else "MISSING")
                if not ok:
                    issues.append(f"generalization: missing row for {algo} / {s}")
            print(f"{algo:<12}" + "".join(f"{c:>26}" for c in cells))
    print()

    # --------------------------------------------------- per-algorithm best run
    print("== BEST RUN PER ALGORITHM (by sweep mean_reward) ==")
    print(f"{'algo':<10}{'best_run_id':<16}{'mean_reward':>14}{'success_rate':>14}")
    _hr()
    for algo in ALGOS:
        algo_dir = root / "sweeps" / algo
        best = None
        for done in algo_dir.glob("*/DONE.json"):
            try:
                p = json.loads(done.read_text())
            except Exception:  # noqa: BLE001
                continue
            mr = p.get("mean_reward")
            if mr is None or (isinstance(mr, float) and math.isnan(mr)):
                continue
            if best is None or mr > best["mean_reward"]:
                best = {"run_id": p.get("run_id", done.parent.name),
                        "mean_reward": float(mr),
                        "success_rate": p.get("success_rate")}
        if best is None:
            print(f"{algo:<10}{'(none)':<16}{'-':>14}{'-':>14}")
            issues.append(f"sweeps/{algo}: no valid run to pick a best from")
        else:
            print(f"{algo:<10}{best['run_id']:<16}{best['mean_reward']:>14.4f}"
                  f"{(best['success_rate'] if best['success_rate'] is not None else float('nan')):>14.4f}")
    print()

    # ----------------------------------------------------------------- verdict
    _hr()
    if issues:
        print(f"RESULT: FAIL -- {len(issues)} issue(s):")
        for it in issues:
            print(f"  - {it}")
        return 1
    print("RESULT: PASS -- all checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(audit())
