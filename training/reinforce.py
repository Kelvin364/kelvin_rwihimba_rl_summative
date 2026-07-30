"""From-scratch REINFORCE (Monte-Carlo policy gradient) in PyTorch.

Shares the training layer's contract: the budget is counted in ENVIRONMENT STEPS
(not episodes), it writes the same ``progress.csv`` columns as the SB3 algorithms,
checkpoints every 25k steps as ``{model, optimizer, steps_done}``, and resumes.

Baselines: {"none", "mean", "value"} where "value" learns a critic (advantage =
return - V(s)). Includes an entropy bonus and discount ``gamma``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

PROGRESS_COLUMNS = [
    "time/total_timesteps",
    "rollout/ep_rew_mean",
    "train/entropy",
    "train/policy_loss",
    "train/value_loss",
]


def _mlp(in_dim: int, out_dim: int, hidden: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.Tanh()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class REINFORCEAgent:
    """Categorical-policy MLP agent with optional learned critic baseline."""

    def __init__(self, obs_dim: int, n_actions: int, hparams: dict[str, Any]) -> None:
        self.gamma = float(hparams.get("gamma", 0.99))
        self.ent_coef = float(hparams.get("ent_coef", 0.0))
        self.baseline = str(hparams.get("baseline", "none"))
        # Monte-Carlo policy gradients are high-variance by construction. My first
        # version updated from a SINGLE episode, and after 150k steps the policy was
        # still sitting at entropy 1.95 out of a 2.20 maximum, essentially uniform and
        # learning nothing. Averaging the gradient over several episodes fixed that.
        self.episodes_per_batch = max(1, int(hparams.get("episodes_per_batch", 8)))
        # I rescale advantages to unit scale so my learning rate is decoupled from
        # the reward magnitude. This only divides, it never re-centres: if it did,
        # the "none" and "mean" baselines would collapse into the same thing and the
        # baseline comparison I am running would measure nothing.
        self.normalize_advantage = bool(hparams.get("normalize_advantage", True))
        # Clipping the gradient norm is what let me use a usable learning rate at
        # all. Unclipped, every lr >= 3e-3 drove entropy to 0 within a few updates and
        # froze the policy on a single action, and one run collapsed to always-WAIT.
        self.max_grad_norm = float(hparams.get("max_grad_norm", 0.5))
        # I give the critic its own learning rate and its own update count. When I
        # fitted it jointly with the policy, one step per batch, its loss stuck at
        # ~15, so `returns - V` was still basically raw returns and my "value"
        # baseline was reducing no variance at all.
        self.value_lr = float(hparams.get("value_lr", 3e-3))
        self.value_epochs = max(1, int(hparams.get("value_epochs", 5)))
        hidden = list(hparams.get("hidden_sizes", [128, 128]))
        lr = float(hparams.get("learning_rate", 1e-3))

        torch.manual_seed(int(hparams.get("seed", 0)))
        self.policy = _mlp(obs_dim, n_actions, hidden)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value = None
        self.value_optimizer = None
        if self.baseline == "value":
            self.value = _mlp(obs_dim, 1, hidden)
            self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.value_lr)

    # -- inference (shared predict() contract with SB3 models) -----------------
    def predict(self, obs, deterministic: bool = True):
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        single = obs_t.ndim == 1
        if single:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            logits = self.policy(obs_t)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = Categorical(logits=logits).sample()
        action = action.cpu().numpy()
        return (int(action[0]) if single else action), None

    # -- checkpoint ------------------------------------------------------------
    def state_dict(self, steps_done: int) -> dict[str, Any]:
        return {
            "model": self.policy.state_dict(),
            "value": self.value.state_dict() if self.value is not None else None,
            "optimizer": self.optimizer.state_dict(),
            "value_optimizer": (
                self.value_optimizer.state_dict() if self.value_optimizer is not None else None
            ),
            "steps_done": steps_done,
        }

    def load_state_dict(self, state: dict[str, Any]) -> int:
        self.policy.load_state_dict(state["model"])
        if self.value is not None and state.get("value") is not None:
            self.value.load_state_dict(state["value"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.value_optimizer is not None and state.get("value_optimizer") is not None:
            self.value_optimizer.load_state_dict(state["value_optimizer"])
        return int(state.get("steps_done", 0))


def _latest_checkpoint(folder: Path) -> Path | None:
    ckpts = sorted(
        (folder / "checkpoints").glob("ckpt_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if ckpts:
        return ckpts[-1]
    model_pt = folder / "model.pt"  # completed run whose DONE.json was removed
    return model_pt if model_pt.exists() else None


def train_reinforce(
    folder: Path,
    hparams: dict[str, Any],
    total_steps: int,
    seed: int = 0,
    checkpoint_freq: int = 25_000,
) -> tuple[REINFORCEAgent, int, list[float]]:
    """Train (or resume) REINFORCE until `total_steps` env steps.

    Returns (agent, steps_done, episode_rewards_this_session).
    """
    folder = Path(folder)
    (folder / "checkpoints").mkdir(parents=True, exist_ok=True)

    from environment.agriscout_env import make_env

    env = make_env()
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    hparams = {**hparams, "seed": seed}
    agent = REINFORCEAgent(obs_dim, n_actions, hparams)

    # Resume if a checkpoint exists (caller guarantees no DONE.json).
    steps_done = 0
    ckpt = _latest_checkpoint(folder)
    if ckpt is not None:
        steps_done = agent.load_state_dict(torch.load(ckpt, weights_only=False))
        print(
            f"[resume] reinforce/{folder.name}: loaded {ckpt.name} at {steps_done} steps, "
            f"remaining={max(0, total_steps - steps_done)}",
            flush=True,
        )

    # progress.csv + episode_rewards.csv (append on resume, else write header).
    prog_path = folder / "progress.csv"
    ep_path = folder / "episode_rewards.csv"
    prog_new = not prog_path.exists()
    ep_new = not ep_path.exists()
    prog_f = open(prog_path, "a", newline="")
    ep_f = open(ep_path, "a", newline="")
    prog_w = csv.writer(prog_f)
    ep_w = csv.writer(ep_f)
    if prog_new:
        prog_w.writerow(PROGRESS_COLUMNS)
    if ep_new:
        ep_w.writerow(["episode", "total_timesteps", "reward"])

    ep_rewards: list[float] = []
    episode_idx = 0
    last_ckpt = steps_done - (steps_done % checkpoint_freq)

    obs, _ = env.reset(seed=seed)
    while steps_done < total_steps:
        # --- collect a BATCH of episodes, then make one update ---
        b_log_probs, b_entropies, b_returns, b_obs = [], [], [], []
        for _ in range(agent.episodes_per_batch):
            log_probs, entropies, rewards, obs_steps = [], [], [], []
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                obs_steps.append(obs_t)
                logits = agent.policy(obs_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
                obs, reward, term, trunc, _ = env.step(int(action.item()))
                rewards.append(reward)
                ep_reward += reward
                steps_done += 1
                done = term or trunc

            # --- Monte-Carlo returns for this episode ---
            returns = np.zeros(len(rewards), dtype=np.float32)
            g = 0.0
            for t in reversed(range(len(rewards))):
                g = rewards[t] + agent.gamma * g
                returns[t] = g

            b_log_probs.append(torch.cat(log_probs))
            b_entropies.append(torch.cat(entropies))
            b_returns.append(torch.as_tensor(returns))
            b_obs.append(torch.cat(obs_steps))

            episode_idx += 1
            ep_rewards.append(ep_reward)
            ep_w.writerow([episode_idx, steps_done, round(ep_reward, 4)])
            if steps_done >= total_steps:
                break
        ep_f.flush()

        log_probs_t = torch.cat(b_log_probs)
        entropy_t = torch.cat(b_entropies).mean()
        returns_t = torch.cat(b_returns)

        value_loss_val = 0.0
        if agent.baseline == "value":
            obs_t_all = torch.cat(b_obs)
            # Baseline from the CURRENT critic, then refit it for the next batch.
            with torch.no_grad():
                advantages = returns_t - agent.value(obs_t_all).squeeze(-1)
            for _ in range(agent.value_epochs):
                v_loss = nn.functional.mse_loss(agent.value(obs_t_all).squeeze(-1), returns_t)
                agent.value_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(agent.value.parameters(), agent.max_grad_norm)
                agent.value_optimizer.step()
            value_loss_val = float(v_loss.item())
        elif agent.baseline == "mean":
            advantages = returns_t - returns_t.mean()
        else:  # "none"
            advantages = returns_t

        if agent.normalize_advantage:
            # Scale only, with deliberately no re-centring, so "none"/"mean"/"value"
            # remain genuinely different baselines rather than collapsing together.
            advantages = advantages / (advantages.std() + 1e-8)

        policy_loss = -(log_probs_t * advantages).mean() - agent.ent_coef * entropy_t

        agent.optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(agent.policy.parameters(), agent.max_grad_norm)
        agent.optimizer.step()

        # --- logging (per update) ---
        prog_w.writerow([
            steps_done,
            round(float(np.mean(ep_rewards[-100:])), 4),
            round(float(entropy_t.item()), 4),
            round(float(policy_loss.item()), 4),
            round(value_loss_val, 4),
        ])
        prog_f.flush()

        # --- checkpoint every `checkpoint_freq` env steps ---
        if steps_done - last_ckpt >= checkpoint_freq:
            last_ckpt = steps_done
            torch.save(
                agent.state_dict(steps_done),
                folder / "checkpoints" / f"ckpt_{steps_done}.pt",
            )

    prog_f.close()
    ep_f.close()
    torch.save(agent.state_dict(steps_done), folder / "model.pt")
    return agent, steps_done, ep_rewards


def load_agent(folder: Path, hparams: dict[str, Any]) -> REINFORCEAgent:
    """Load a trained REINFORCE agent for evaluation."""
    from environment.agriscout_env import make_env

    env = make_env()
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    agent = REINFORCEAgent(obs_dim, n_actions, hparams)
    state = torch.load(Path(folder) / "model.pt", weights_only=False)
    agent.load_state_dict(state)
    return agent
