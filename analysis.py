"""Generate report assets from the AgriScout training results.

Reads $AGRISCOUT_RESULTS (default ./logs) and writes:
  assets/figures/  -- 5 figures (PNG)
  assets/tables/   -- 4 per-algo hyperparameter tables + best_summary (CSV + MD)

    uv run python analysis.py

Reference lines use the oracle benchmark from tests/test_oracle.py (a scripted
greedy policy) and a random policy, on eval seeds 9000-9019.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table (no tabulate dep)."""
    cols = [str(c) for c in df.columns]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([head, sep, *body])


ALGOS = ["dqn", "ppo", "a2c", "reinforce"]
LABELS = {"dqn": "DQN", "ppo": "PPO", "a2c": "A2C", "reinforce": "REINFORCE"}
# Validated categorical palette (dataviz skill), fixed per-algo; markers give the
# required secondary encoding (CVD relief) alongside labels + legends.
COLORS = {"dqn": "#2a78d6", "ppo": "#eb6834", "a2c": "#1baf7a", "reinforce": "#e34948"}
MARKERS = {"dqn": "o", "ppo": "s", "a2c": "^", "reinforce": "D"}

# Reference policies (oracle benchmark, seeds 9000-9019) -- see tests/test_oracle.py.
RANDOM_BASELINE = -5.13
ORACLE_BASELINE = 17.0

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 11,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": BASELINE,
})


def results_root() -> Path:
    return Path(os.environ.get("AGRISCOUT_RESULTS", "./logs"))


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _refs(ax, x=0.99):
    ax.axhline(RANDOM_BASELINE, color=MUTED, ls="--", lw=1.2, zorder=1)
    ax.axhline(ORACLE_BASELINE, color="#006300", ls=":", lw=1.2, zorder=1)
    ax.text(x, RANDOM_BASELINE, " random", color=MUTED, va="bottom", ha="right",
            transform=ax.get_yaxis_transform(), fontsize=9)
    ax.text(x, ORACLE_BASELINE, " oracle", color="#006300", va="top", ha="right",
            transform=ax.get_yaxis_transform(), fontsize=9)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_sweeps() -> dict[str, pd.DataFrame]:
    out = {}
    for algo in ALGOS:
        p = results_root() / "sweeps" / algo / "summary.csv"
        out[algo] = pd.read_csv(p)
    return out


def load_finals() -> pd.DataFrame:
    rows = []
    for algo in ALGOS:
        p = results_root() / "finals" / f"{algo}_best" / "DONE.json"
        rows.append(json.loads(p.read_text()))
    return pd.DataFrame(rows)


def load_generalization() -> pd.DataFrame:
    return pd.read_csv(results_root() / "generalization" / "results.csv")


def load_curves() -> dict[str, pd.DataFrame]:
    out = {}
    for algo in ALGOS:
        p = results_root() / "finals" / f"{algo}_best" / "episode_rewards.csv"
        if p.exists():
            out[algo] = pd.read_csv(p)
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_learning_curves(curves, out: Path):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for algo in ALGOS:
        df = curves.get(algo)
        if df is None or df.empty:
            continue
        roll = df["reward"].rolling(50, min_periods=1).mean()
        ax.plot(df["total_timesteps"], roll, color=COLORS[algo], lw=2, label=LABELS[algo],
                marker=MARKERS[algo], markevery=max(1, len(df) // 12), markersize=5)
        ax.annotate(LABELS[algo], (df["total_timesteps"].iloc[-1], roll.iloc[-1]),
                    color=COLORS[algo], fontsize=9, va="center", ha="left", xytext=(4, 0),
                    textcoords="offset points")
    _refs(ax)
    _style(ax)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode reward (50-episode rolling mean)")
    ax.set_title("Learning curves — 400k-step finals", color=INK, loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", ncol=4)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_final_comparison(finals, out: Path):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(ALGOS))
    w = 0.38
    det = [finals.loc[finals.algo == a, "mean_reward"].iloc[0] for a in ALGOS]
    sto = [finals.loc[finals.algo == a, "mean_reward_stochastic"].iloc[0] for a in ALGOS]
    b1 = ax.bar(x - w / 2, det, w, color=[COLORS[a] for a in ALGOS], label="deterministic",
                zorder=3)
    b2 = ax.bar(x + w / 2, sto, w, color=[COLORS[a] for a in ALGOS], alpha=0.55,
                hatch="///", edgecolor=SURFACE, label="stochastic", zorder=3)
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:.1f}", (r.get_x() + r.get_width() / 2, h),
                        ha="center", va="bottom" if h >= 0 else "top",
                        fontsize=8.5, color=INK, xytext=(0, 2 if h >= 0 else -2),
                        textcoords="offset points")
    _refs(ax)
    _style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ALGOS])
    ax.set_ylabel("mean eval reward (seeds 9000-9019)")
    ax.set_title("Final agent performance — deterministic vs stochastic",
                 color=INK, loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_generalization(gen, out: Path):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(ALGOS))
    w = 0.38
    sets = ["train_dist_0-19", "heldout_10000-10049"]
    labels = ["training seeds", "held-out seeds"]
    hatches = ["", "///"]
    for i, (s, lab, ht) in enumerate(zip(sets, labels, hatches)):
        means = [gen[(gen.algo == a) & (gen.seed_set == s)]["mean_reward"].iloc[0] for a in ALGOS]
        stds = [gen[(gen.algo == a) & (gen.seed_set == s)]["std_reward"].iloc[0] for a in ALGOS]
        bars = ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=3,
                      color=[COLORS[a] for a in ALGOS], alpha=1.0 if i == 0 else 0.55,
                      hatch=ht, edgecolor=SURFACE, label=lab, zorder=3,
                      error_kw={"ecolor": MUTED, "lw": 1})
        for r, m in zip(bars, means):
            ax.annotate(f"{m:.1f}", (r.get_x() + r.get_width() / 2, m), ha="center",
                        va="top", fontsize=8, color=INK, xytext=(0, -3),
                        textcoords="offset points")
    ax.axhline(RANDOM_BASELINE, color=MUTED, ls="--", lw=1.2)
    ax.text(0.99, RANDOM_BASELINE, " random", color=MUTED, va="bottom", ha="right",
            transform=ax.get_yaxis_transform(), fontsize=9)
    _style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ALGOS])
    ax.set_ylabel("mean reward ± std (deterministic)")
    ax.set_title("Generalization — held-out vs training-distribution seeds",
                 color=INK, loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_hparam_sensitivity(sweeps, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharey=False)
    for ax, algo in zip(axes.ravel(), ALGOS):
        df = sweeps[algo]
        x = df["hp_learning_rate"]
        y = df["mean_reward_stochastic"]
        ax.scatter(x, y, s=70, color=COLORS[algo], marker=MARKERS[algo],
                   edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.axhline(RANDOM_BASELINE, color=MUTED, ls="--", lw=1)
        ax.set_xscale("log")
        _style(ax)
        ax.set_title(LABELS[algo], color=COLORS[algo], loc="left", fontweight="bold")
        ax.set_xlabel("learning rate (log)")
        ax.set_ylabel("stochastic eval reward")
    fig.suptitle("Hyperparameter sensitivity — sweep configs (learning rate)",
                 color=INK, x=0.02, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_health(finals, out: Path):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(ALGOS))
    w = 0.38
    det = [finals.loc[finals.algo == a, "mean_final_health"].iloc[0] for a in ALGOS]
    sto = [finals.loc[finals.algo == a, "mean_final_health_stochastic"].iloc[0] for a in ALGOS]
    b1 = ax.bar(x - w / 2, det, w, color=[COLORS[a] for a in ALGOS], label="deterministic",
                zorder=3)
    b2 = ax.bar(x + w / 2, sto, w, color=[COLORS[a] for a in ALGOS], alpha=0.55,
                hatch="///", edgecolor=SURFACE, label="stochastic", zorder=3)
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:.2f}", (r.get_x() + r.get_width() / 2, h), ha="center",
                        va="bottom", fontsize=8.5, color=INK, xytext=(0, 2),
                        textcoords="offset points")
    ax.axhline(0.60, color="#006300", ls=":", lw=1.2)
    ax.text(0.99, 0.60, " success threshold (0.60)", color="#006300", va="bottom",
            ha="right", transform=ax.get_yaxis_transform(), fontsize=9)
    _style(ax)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ALGOS])
    ax.set_ylabel("mean final field health")
    ax.set_title("Final field health — graceful, but below the success threshold",
                 color=INK, loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
_HP_METRICS = [
    "mean_reward", "mean_reward_stochastic", "success_rate",
    "success_rate_stochastic", "mean_final_health", "episodes_to_converge",
]


def write_hparam_tables(sweeps, tdir: Path):
    for algo in ALGOS:
        df = sweeps[algo].copy()
        hp_cols = [c for c in df.columns if c.startswith("hp_")]
        cols = ["run_id"] + hp_cols + _HP_METRICS
        cols = [c for c in cols if c in df.columns]
        tab = df[cols].sort_values("mean_reward_stochastic", ascending=False)
        tab.to_csv(tdir / f"hparams_{algo}.csv", index=False)
        (tdir / f"hparams_{algo}.md").write_text(
            f"### {LABELS[algo]} sweep configurations (100k steps, sorted by stochastic reward)\n\n"
            + _md_table(tab.round(3)) + "\n"
        )


def write_best_summary(finals, gen, tdir: Path):
    rows = []
    for algo in ALGOS:
        f = finals.loc[finals.algo == algo].iloc[0]
        heldout = gen[(gen.algo == algo) & (gen.seed_set == "heldout_10000-10049")].iloc[0]
        rows.append({
            "algo": LABELS[algo],
            "best_run": f["run_id"],
            "det_reward": round(f["mean_reward"], 2),
            "stoch_reward": round(f["mean_reward_stochastic"], 2),
            "success_det": round(f["success_rate"], 2),
            "success_stoch": round(f["success_rate_stochastic"], 2),
            "final_health": round(f["mean_final_health"], 3),
            "heldout_reward": round(heldout["mean_reward"], 2),
            "episodes_to_converge": int(f["episodes_to_converge"]),
        })
    df = pd.DataFrame(rows).sort_values("stoch_reward", ascending=False)
    df.to_csv(tdir / "best_summary.csv", index=False)
    (tdir / "best_summary.md").write_text(
        "### Best agent per algorithm (400k-step finals)\n\n"
        f"Reference: random policy {RANDOM_BASELINE}, oracle {ORACLE_BASELINE} "
        "(eval seeds 9000-9019).\n\n" + _md_table(df) + "\n"
    )
    return df


def main():
    root = results_root()
    fdir = Path("assets/figures")
    tdir = Path("assets/tables")
    fdir.mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)

    sweeps = load_sweeps()
    finals = load_finals()
    gen = load_generalization()
    curves = load_curves()

    print(f"AGRISCOUT_RESULTS = {root.resolve()}")
    fig_learning_curves(curves, fdir / "fig1_learning_curves.png")
    fig_final_comparison(finals, fdir / "fig2_final_comparison.png")
    fig_generalization(gen, fdir / "fig3_generalization.png")
    fig_hparam_sensitivity(sweeps, fdir / "fig4_hparam_sensitivity.png")
    fig_health(finals, fdir / "fig5_final_health.png")
    print(f"wrote 5 figures -> {fdir}")

    write_hparam_tables(sweeps, tdir)
    best = write_best_summary(finals, gen, tdir)
    print(f"wrote 4 hparam tables + best_summary -> {tdir}\n")
    print("=== best_summary ===")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
