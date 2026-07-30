"""Generate report assets from the AgriScout training results.

Reads $AGRISCOUT_RESULTS (default ./logs) and writes:
  assets/figures/  : 9 figures (PNG)
  assets/tables/   : 4 per-algo hyperparameter tables + best_summary (CSV + MD)

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
# I fix one colour per algorithm so a given method looks the same in every figure,
# and I checked the four for colour-blind separation. Each series also carries its
# own marker shape, so identity never depends on hue alone.
COLORS = {"dqn": "#2a78d6", "ppo": "#eb6834", "a2c": "#1baf7a", "reinforce": "#e34948"}
MARKERS = {"dqn": "o", "ppo": "s", "a2c": "^", "reinforce": "D"}

# Reference policies (oracle benchmark, seeds 9000-9019), defined in tests/test_oracle.py.
# I compute these from the live environment instead of writing the numbers in by hand.
# I had them hard-coded at first, and when I changed the reward function the stale pair
# stayed put and would have mislabelled the reference line on every figure. I cache
# them in <results>/baselines.json keyed by ENV_VERSION, so changing the reward
# invalidates the cache for me automatically.
def baselines() -> tuple[float, float]:
    """Return (random_mean_reward, oracle_mean_reward) for the CURRENT env."""
    from environment.agriscout_env import ENV_VERSION

    cache = results_root() / "baselines.json"
    if cache.exists():
        payload = json.loads(cache.read_text())
        if payload.get("env_version") == ENV_VERSION:
            return payload["random"], payload["oracle"]

    from tests.test_oracle import make_random_policy, oracle_policy, run_policy

    rnd = run_policy(make_random_policy())["mean_reward"]
    orc = run_policy(oracle_policy)["mean_reward"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(
        {"env_version": ENV_VERSION, "random": rnd, "oracle": orc}, indent=2
    ))
    return rnd, orc


RANDOM_BASELINE: float = 0.0   # populated by main() via baselines()
ORACLE_BASELINE: float = 0.0

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


def _steps_axis(ax):
    """Format an env-step x-axis as 0 / 100k / 200k.

    Raw step counts are six digits wide, so on a half-width subplot the default
    ticks run into each other ("50000010000015000...").
    """
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: "0" if v == 0 else f"{v / 1000:.0f}k"))


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


def load_progress() -> dict[str, pd.DataFrame]:
    """Per-algorithm training diagnostics (losses, entropy, exploration rate)."""
    out = {}
    for algo in ALGOS:
        p = results_root() / "finals" / f"{algo}_best" / "progress.csv"
        if p.exists():
            df = pd.read_csv(p)
            if "time/total_timesteps" in df.columns:
                df = df.dropna(subset=["time/total_timesteps"])
            out[algo] = df
    return out


def _entropy_series(df: pd.DataFrame) -> pd.Series | None:
    """Policy entropy in nats, sign-normalised across implementations.

    Stable-Baselines3 logs ``train/entropy_loss = -mean(entropy)``, i.e. NEGATIVE
    entropy, while the from-scratch REINFORCE logs entropy directly. Plotting the
    raw columns together would show A2C/PPO upside-down against REINFORCE.
    """
    if "train/entropy" in df.columns:
        return df["train/entropy"]
    if "train/entropy_loss" in df.columns:
        return -df["train/entropy_loss"]
    return None


def _smooth(s: pd.Series, frac: float = 0.02) -> pd.Series:
    """Rolling mean sized relative to the series, so densely- and sparsely-logged
    algorithms (A2C logs 20k rows, PPO 200) get comparable visual smoothing."""
    return s.rolling(max(2, int(len(s) * frac)), min_periods=1).mean()


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
    _steps_axis(ax)
    ax.set_title("Learning curves: 400k-step finals", color=INK, loc="left", fontweight="bold")
    # Opaque frame: at "upper center" with no frame the legend text sat directly on
    # the oracle reference line.
    ax.legend(loc="lower right", ncol=2, frameon=True, facecolor=SURFACE,
              edgecolor=GRID, framealpha=1.0)
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
    ax.set_title("Final agent performance: deterministic vs stochastic",
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
    # Plot the STOCHASTIC policy: every agent here is markedly better sampled than
    # greedy, so a deterministic-only chart would describe a policy nobody runs.
    for i, (s, lab, ht) in enumerate(zip(sets, labels, hatches)):
        rows = [gen[(gen.algo == a) & (gen.seed_set == s)].iloc[0] for a in ALGOS]
        means = [r["mean_reward_stochastic"] for r in rows]
        stds = [r["std_reward_stochastic"] for r in rows]
        bars = ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=3,
                      color=[COLORS[a] for a in ALGOS], alpha=1.0 if i == 0 else 0.55,
                      hatch=ht, edgecolor=SURFACE, label=lab, zorder=3,
                      error_kw={"ecolor": MUTED, "lw": 1})
        for r, m in zip(bars, means):
            # The error bar is drawn at the bar centre, so a centred label gets
            # struck through by it ("4.4" reading as "4 4"). Offset off-centre and
            # sit the text on a surface-coloured patch.
            ax.annotate(
                f"{m:.1f}", (r.get_x() + r.get_width() / 2, m),
                ha="center", va="top", fontsize=8, color=INK,
                xytext=(0, -5), textcoords="offset points", zorder=5,
                bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE,
                      "edgecolor": "none"},
            )
    ax.axhline(RANDOM_BASELINE, color=MUTED, ls="--", lw=1.2)
    ax.text(0.99, RANDOM_BASELINE, " random", color=MUTED, va="bottom", ha="right",
            transform=ax.get_yaxis_transform(), fontsize=9)
    _style(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ALGOS])
    ax.set_ylabel("mean reward ± std (stochastic)")
    ax.set_title("Generalization: held-out vs training-distribution seeds",
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
    fig.suptitle("Hyperparameter sensitivity across sweep configs (learning rate)",
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
    ax.set_title("Final field health: graceful, but below the success threshold",
                 color=INK, loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_reward_subplots(curves, out: Path):
    """Cumulative-reward curves, one panel per method, shared axes.

    The overlaid version (fig 1) is better for ranking; this one is better for
    reading an individual method's stability, which the overlay hides.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, algo in zip(axes.ravel(), ALGOS):
        df = curves.get(algo)
        if df is None or df.empty:
            continue
        roll = df["reward"].rolling(50, min_periods=1).mean()
        std = df["reward"].rolling(50, min_periods=1).std()
        ax.fill_between(df["total_timesteps"], roll - std, roll + std,
                        color=COLORS[algo], alpha=0.16, linewidth=0, zorder=2)
        ax.plot(df["total_timesteps"], roll, color=COLORS[algo], lw=1.9, zorder=3)
        ax.axhline(RANDOM_BASELINE, color=MUTED, ls="--", lw=1, zorder=1)
        ax.axhline(ORACLE_BASELINE, color="#006300", ls=":", lw=1, zorder=1)
        final = roll.iloc[-1]
        pct = 100 * (final - RANDOM_BASELINE) / (ORACLE_BASELINE - RANDOM_BASELINE)
        ax.set_title(f"{LABELS[algo]}:  final {final:.1f}  ({pct:.0f}% of oracle)",
                     color=COLORS[algo], loc="left", fontweight="bold", fontsize=11)
        _style(ax)
    for ax in axes[1]:
        ax.set_xlabel("environment steps")
        _steps_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("episode reward (50-ep mean ± 1σ)")
    fig.suptitle("Cumulative reward per method, with the 50-episode rolling σ shaded "
                 "(dashed = random, dotted = oracle)",
                 color=INK, x=0.02, ha="left", fontweight="bold", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_dqn_objective(progress, out: Path):
    """DQN's own objective: the TD loss it minimises, plus the epsilon schedule."""
    df = progress.get("dqn")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    if df is not None and "train/loss" in df.columns:
        x = df["time/total_timesteps"]
        a1.plot(x, df["train/loss"], color=COLORS["dqn"], lw=0.7, alpha=0.35, zorder=2)
        a1.plot(x, _smooth(df["train/loss"], 0.03), color=COLORS["dqn"], lw=2, zorder=3)
        a1.set_yscale("log")
        a1.set_ylabel("TD loss (log scale)")
        if "rollout/exploration_rate" in df.columns:
            a2.plot(x, df["rollout/exploration_rate"], color=COLORS["dqn"], lw=2)
            a2.set_ylabel("exploration rate ε")
            a2.set_ylim(0, 1.02)
    for ax, t in ((a1, "Temporal-difference loss, the quantity DQN minimises"),
                  (a2, "ε-greedy schedule, annealing exploration")):
        ax.set_xlabel("environment steps")
        ax.set_title(t, color=INK, loc="left", fontweight="bold", fontsize=10.5)
        _style(ax); _steps_axis(ax)
    fig.suptitle("DQN objective curves", color=INK, x=0.02, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_pg_entropy(progress, out: Path):
    """Policy entropy for the three policy-gradient methods.

    Entropy is the direct read-out of exploration-vs-exploitation: it starts at the
    uniform maximum and falls as the policy commits. A flat line at the maximum
    means the policy never committed; a crash to zero means it collapsed onto one
    action.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5))
    max_ent = float(np.log(9))
    for algo in ("ppo", "a2c", "reinforce"):
        df = progress.get(algo)
        if df is None:
            continue
        ent = _entropy_series(df)
        if ent is None or "time/total_timesteps" not in df.columns:
            continue
        x = df["time/total_timesteps"]
        ax.plot(x, ent, color=COLORS[algo], lw=0.7, alpha=0.25, zorder=2)
        ax.plot(x, _smooth(ent, 0.03), color=COLORS[algo], lw=2.2, label=LABELS[algo],
                marker=MARKERS[algo], markevery=max(1, len(x) // 10), markersize=5,
                zorder=3)
    ax.axhline(max_ent, color=MUTED, ls="--", lw=1.2)
    ax.text(0.99, max_ent, " uniform policy (ln 9)", color=MUTED, va="bottom",
            ha="right", transform=ax.get_yaxis_transform(), fontsize=9)
    ax.set_ylim(0, max_ent * 1.08)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("policy entropy (nats)")
    _steps_axis(ax)
    ax.set_title("Policy-gradient entropy: exploration decaying into exploitation",
                 color=INK, loc="left", fontweight="bold")
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=1.0,
              loc="lower left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_convergence(curves, finals, out: Path):
    """Convergence: when each run captured 80% of its total improvement.

    The threshold is measured against the improvement RANGE (start → final), not
    80% of the final value, which is meaningless when rewards are negative.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, algo in zip(axes.ravel(), ALGOS):
        df = curves.get(algo)
        if df is None or df.empty:
            continue
        roll = df["reward"].rolling(50, min_periods=1).mean()
        start, final = roll.iloc[0], roll.iloc[-1]
        thresh = start + 0.8 * (final - start)
        ax.plot(roll.index + 1, roll, color=COLORS[algo], lw=1.9, zorder=3)
        ax.axhline(thresh, color=MUTED, ls=":", lw=1.3, zorder=2)
        hit = next((i + 1 for i, v in enumerate(roll) if v >= thresh), None)
        if hit:
            ax.axvline(hit, color=INK, ls="--", lw=1.2, zorder=2)
            ax.annotate(f"converged\nep {hit}", (hit, thresh), xytext=(8, -28),
                        textcoords="offset points", fontsize=9, color=INK,
                        bbox={"boxstyle": "round,pad=0.25", "facecolor": SURFACE,
                              "edgecolor": GRID})
        ax.set_title(f"{LABELS[algo]}:  {start:.1f} to {final:.1f}",
                     color=COLORS[algo], loc="left", fontweight="bold", fontsize=11)
        _style(ax)
    for ax in axes[1]:
        ax.set_xlabel("training episode")
    for ax in axes[:, 0]:
        ax.set_ylabel("episode reward (50-ep mean)")
    fig.suptitle("Convergence: the dotted line is 80% of each run's total improvement",
                 color=INK, x=0.02, ha="left", fontweight="bold", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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
        traindist = gen[(gen.algo == algo) & (gen.seed_set == "train_dist_0-19")].iloc[0]
        best_reward = max(f["mean_reward"], f["mean_reward_stochastic"])
        rows.append({
            "algo": LABELS[algo],
            "best_run": f["run_id"],
            "det_reward": round(f["mean_reward"], 2),
            "stoch_reward": round(f["mean_reward_stochastic"], 2),
            "success_det": round(f["success_rate"], 2),
            "success_stoch": round(f["success_rate_stochastic"], 2),
            "final_health": round(f["mean_final_health"], 3),
            # Held-out reward is reported in the SAME action mode as the headline
            # (stochastic). Mixing a deterministic held-out number into an otherwise
            # stochastic table made PPO read as -4.18, as if it generalized badly,
            # when its stochastic held-out score is +7.09, ABOVE its own
            # training-distribution score. The generalization gap column makes that
            # explicit: positive means it does better on unseen seeds.
            "heldout_reward": round(heldout["mean_reward_stochastic"], 2),
            "gen_gap": round(
                heldout["mean_reward_stochastic"] - traindist["mean_reward_stochastic"], 2
            ),
            # Scale-free: where the agent sits between the two reference policies.
            # Raw rewards are NOT comparable across reward-function versions; this is.
            "pct_of_oracle": round(
                100.0 * (best_reward - RANDOM_BASELINE)
                / (ORACLE_BASELINE - RANDOM_BASELINE), 1
            ),
            "episodes_to_converge": int(f["episodes_to_converge"]),
        })
    df = pd.DataFrame(rows).sort_values("stoch_reward", ascending=False)
    df.to_csv(tdir / "best_summary.csv", index=False)
    (tdir / "best_summary.md").write_text(
        "### Best agent per algorithm (400k-step finals)\n\n"
        f"Reference policies on eval seeds 9000-9019: random {RANDOM_BASELINE:.2f}, "
        f"oracle {ORACLE_BASELINE:.2f}. `pct_of_oracle` places each agent on that "
        "scale (0% = random, 100% = oracle) and is the only column comparable "
        "across reward-function versions.\n\n" + _md_table(df) + "\n"
    )
    return df


def main():
    global RANDOM_BASELINE, ORACLE_BASELINE

    root = results_root()
    fdir = Path("assets/figures")
    tdir = Path("assets/tables")
    fdir.mkdir(parents=True, exist_ok=True)
    tdir.mkdir(parents=True, exist_ok=True)

    RANDOM_BASELINE, ORACLE_BASELINE = baselines()

    sweeps = load_sweeps()
    finals = load_finals()
    gen = load_generalization()
    curves = load_curves()
    progress = load_progress()

    print(f"AGRISCOUT_RESULTS = {root.resolve()}")
    print(f"reference policies: random={RANDOM_BASELINE:.2f} oracle={ORACLE_BASELINE:.2f}")
    fig_learning_curves(curves, fdir / "fig1_learning_curves.png")
    fig_final_comparison(finals, fdir / "fig2_final_comparison.png")
    fig_generalization(gen, fdir / "fig3_generalization.png")
    fig_hparam_sensitivity(sweeps, fdir / "fig4_hparam_sensitivity.png")
    fig_health(finals, fdir / "fig5_final_health.png")
    fig_reward_subplots(curves, fdir / "fig6_reward_subplots.png")
    fig_dqn_objective(progress, fdir / "fig7_dqn_objective.png")
    fig_pg_entropy(progress, fdir / "fig8_pg_entropy.png")
    fig_convergence(curves, finals, fdir / "fig9_convergence.png")
    print(f"wrote 9 figures -> {fdir}")

    write_hparam_tables(sweeps, tdir)
    best = write_best_summary(finals, gen, tdir)
    print(f"wrote 4 hparam tables + best_summary -> {tdir}\n")
    print("=== best_summary ===")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
