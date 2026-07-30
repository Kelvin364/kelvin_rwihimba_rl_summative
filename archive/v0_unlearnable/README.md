# Reward v0 — the unlearnable baseline (archived)

These are the complete results from `ENV_VERSION = "agriscout-v0"`, kept because the
before/after is the central experimental finding of this project, not because the
numbers are useful on their own.

**Do not compare these rewards to v1 rewards directly** — the reward function itself
changed, so the scales differ. Compare *position between the random and oracle
reference policies*, which is scale-free.

## What happened

v0 passed its scripted-oracle winnability gate (oracle +17.03, 100% success) and was
still unlearnable. Across 40 sweep configurations plus four 400k-step finals —
roughly 8M environment steps — **not one run ever recorded a single success**, and
three of the four algorithms finished *worse than a uniform-random policy*.

| algo | det reward | stoch reward | success | % of random→oracle span |
| --- | --- | --- | --- | --- |
| PPO | −6.45 | 0.32 | 0.05 | 24.6% |
| A2C | −9.73 | −6.46 | 0.00 | −6.0% |
| DQN | −8.46 | −7.88 | 0.00 | −12.4% |
| REINFORCE | −16.98 | −16.86 | 0.00 | −52.9% |

Reference policies: random −5.13, oracle +17.03 (seeds 9000–9019).

## Diagnosis

Decomposing the oracle's return into its dense (per-step) and terminal parts:

| policy | total | dense | terminal bonus |
| --- | --- | --- | --- |
| oracle | +17.03 | **−2.97** | +20.00 |
| random | −5.13 | **−5.13** | 0 |

The entire *learnable* gap between an optimal and a random policy was **2.16 reward
over 150 steps — 0.014 per step** — against per-episode noise of σ ≈ 2–5. Signal-to-
noise below 1. Ninety percent of the oracle's advantage (20 of 22.2) sat in a single
all-or-nothing terminal bonus gated on two simultaneous thresholds, which no agent
ever once received and therefore could never learn from.

Compounding it, the navigation shaping was scaled against the per-*episode* time cost
instead of the per-*step* one, so approaching work scored **−0.015/step**: the
shaping term penalised the exact behaviour it had been added to encourage.

## The lesson

A scripted-oracle gate proves a task is **winnable**. It says nothing about whether
the task is **learnable**. Those are different properties, and only the second one
predicts whether training will work.

v1 adds the missing check: `scripts/learnability_gate.py` trains a real agent briefly
and requires it to beat random before any sweep compute is spent, and
`tests/test_oracle.py` asserts a minimum dense-reward gap and that moving toward work
is net-positive.
