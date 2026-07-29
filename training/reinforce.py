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
        hidden = list(hparams.get("hidden_sizes", [128, 128]))
        lr = float(hparams.get("learning_rate", 1e-3))

        torch.manual_seed(int(hparams.get("seed", 0)))
        self.policy = _mlp(obs_dim, n_actions, hidden)
        params = list(self.policy.parameters())
        self.value = None
        if self.baseline == "value":
            self.value = _mlp(obs_dim, 1, hidden)
            params += list(self.value.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

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
            "steps_done": steps_done,
        }

    def load_state_dict(self, state: dict[str, Any]) -> int:
        self.policy.load_state_dict(state["model"])
        if self.value is not None and state.get("value") is not None:
            self.value.load_state_dict(state["value"])
        self.optimizer.load_state_dict(state["optimizer"])
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
        # --- collect one episode ---
        log_probs, entropies, rewards, values = [], [], [], []
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = agent.policy(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            if agent.value is not None:
                values.append(agent.value(obs_t).squeeze(-1))
            obs, reward, term, trunc, _ = env.step(int(action.item()))
            rewards.append(reward)
            ep_reward += reward
            steps_done += 1
            done = term or trunc

        # --- Monte-Carlo returns ---
        returns = np.zeros(len(rewards), dtype=np.float32)
        g = 0.0
        for t in reversed(range(len(rewards))):
            g = rewards[t] + agent.gamma * g
            returns[t] = g
        returns_t = torch.as_tensor(returns)

        log_probs_t = torch.cat(log_probs)
        entropy_t = torch.cat(entropies).mean()

        value_loss_val = 0.0
        if agent.baseline == "value":
            values_t = torch.cat(values)
            advantages = returns_t - values_t.detach()
            value_loss = nn.functional.mse_loss(values_t, returns_t)
        elif agent.baseline == "mean":
            advantages = returns_t - returns_t.mean()
            value_loss = torch.tensor(0.0)
        else:  # "none"
            advantages = returns_t
            value_loss = torch.tensor(0.0)

        policy_loss = -(log_probs_t * advantages).mean() - agent.ent_coef * entropy_t
        loss = policy_loss + (0.5 * value_loss if agent.baseline == "value" else 0.0)

        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()
        if agent.baseline == "value":
            value_loss_val = float(value_loss.item())

        # --- logging (per update) ---
        episode_idx += 1
        ep_rewards.append(ep_reward)
        prog_w.writerow([
            steps_done,
            round(float(np.mean(ep_rewards[-100:])), 4),
            round(float(entropy_t.item()), 4),
            round(float(policy_loss.item()), 4),
            round(value_loss_val, 4),
        ])
        prog_f.flush()
        ep_w.writerow([episode_idx, steps_done, round(ep_reward, 4)])
        ep_f.flush()

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
