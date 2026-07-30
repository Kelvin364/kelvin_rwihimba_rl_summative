"""AgriScout serving API — the environment and trained agents behind HTTP/JSON.

Shows the path from "research artifact" to "product component": the same trained
policies the report evaluates are exposed as a stateless inference endpoint and a
stateful session, both returning plain JSON that a web or mobile client can render
directly. `assets/demo/index.html` is served from `/` as a worked example of a
frontend consuming this data.

    uv run uvicorn api:app --reload
    open http://127.0.0.1:8000/docs        # interactive OpenAPI schema

Endpoints
---------
    GET  /health                  liveness + which agents are loadable
    GET  /agents                  agent registry with headline eval metrics
    POST /episode                 run a whole episode, return the full trace
    POST /session                 open a stateful session (returns session_id)
    POST /session/{id}/step       advance one step; the agent picks the action
    POST /session/{id}/act        advance one step with a CLIENT-chosen action
    GET  /session/{id}            current state without advancing
    DELETE /session/{id}          release it

The step endpoints return the same frame schema the recorder writes, so a client
can drive a live agent or replay a recorded episode with one rendering path.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from environment.agriscout_env import (
    ACTION_NAMES,
    SUCCESS_HEALTH,
    SUCCESS_PEST,
    AgriScoutEnv,
)

REPO = Path(__file__).resolve().parent
AGENTS = ("ppo", "dqn", "a2c", "reinforce", "oracle", "random")
MAX_SESSIONS = 64

app = FastAPI(
    title="AgriScout RL API",
    version="1.0.0",
    description="Trained crop-scouting agents exposed as JSON over HTTP.",
)
# Open CORS: this is a local demo service with no auth and no mutable server state
# worth protecting. A deployment would restrict origins and add a key.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_predictors: dict[str, Any] = {}
_sessions: dict[str, dict[str, Any]] = {}


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class EpisodeRequest(BaseModel):
    agent: Literal[AGENTS] = "ppo"  # type: ignore[valid-type]
    seed: int = Field(0, ge=0, le=2**31 - 1)
    deterministic: bool = False
    include_grids: bool = Field(True, description="embed per-step health/pest grids")


class SessionRequest(BaseModel):
    agent: Literal[AGENTS] = "ppo"  # type: ignore[valid-type]
    seed: int = Field(0, ge=0, le=2**31 - 1)
    deterministic: bool = False


class ActRequest(BaseModel):
    action: str = Field(..., description=f"one of {ACTION_NAMES}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _predictor(agent: str):
    """Load and cache a predictor. Models are ~1MB, so caching matters under load."""
    if agent not in _predictors:
        from main import load_predictor

        try:
            _predictors[agent] = load_predictor(agent)
        except FileNotFoundError as exc:
            raise HTTPException(503, f"agent {agent!r} is not trained yet: {exc}") from exc
    return _predictors[agent]


def _act(agent: str, predictor, env: AgriScoutEnv, obs, deterministic: bool) -> int:
    if agent in ("oracle", "random"):
        predictor.bind(env)
    action, _ = predictor.predict(obs, deterministic=deterministic)
    return int(np.asarray(action).reshape(-1)[0])


def _frame(env: AgriScoutEnv, action: int, reward: float, cum: float,
           grids: bool = True) -> dict[str, Any]:
    """One step of state as JSON — the same shape the trace recorder writes."""
    r, c = int(round(env.rover_row)), int(round(env.rover_col))
    out: dict[str, Any] = {
        "t": int(env.step_count),
        "action": ACTION_NAMES[action],
        "reward": round(float(reward), 3),
        "cum_reward": round(float(cum), 3),
        "rover": {"row": r, "col": c, "heading": round(float(env.rover_heading), 3)},
        "resources": {
            "battery": round(float(env.battery), 3),
            "water": round(float(env.water), 3),
            "pesticide": round(float(env.pesticide), 3),
        },
        "field": {
            "mean_health": round(env.mean_health, 4),
            "mean_pest": round(env.mean_pest, 4),
            "success_now": bool(env.is_success),
        },
    }
    if grids:
        out["health_grid"] = np.round(env.health_grid, 2).tolist()
        out["pest_grid"] = np.round(env.pest_grid, 2).tolist()
    return out


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict[str, Any]:
    from main import MODELS

    return {
        "status": "ok",
        "env_version": AgriScoutEnv.ENV_VERSION,
        "trained_agents_available": {a: MODELS[a].exists() for a in MODELS},
        "scripted_agents": ["oracle", "random"],
        "open_sessions": len(_sessions),
    }


@app.get("/agents")
def agents() -> dict[str, Any]:
    """Registry + headline eval metrics, so a client can label what it is running."""
    import json

    out = []
    for name in AGENTS:
        entry: dict[str, Any] = {"name": name,
                                 "kind": "scripted" if name in ("oracle", "random") else "learned"}
        done = REPO / "logs" / "finals" / f"{name}_best" / "DONE.json"
        if done.exists():
            d = json.loads(done.read_text())
            entry["metrics"] = {
                "mean_reward_deterministic": round(d["mean_reward"], 2),
                "mean_reward_stochastic": round(d["mean_reward_stochastic"], 2),
                "success_rate_stochastic": d["success_rate_stochastic"],
            }
            entry["hyperparameters"] = d.get("hparams", {})
        out.append(entry)
    return {
        "agents": out,
        "action_space": ACTION_NAMES,
        "success_criteria": {"mean_health_min": SUCCESS_HEALTH, "mean_pest_max": SUCCESS_PEST},
    }


@app.post("/episode")
def run_episode(req: EpisodeRequest) -> dict[str, Any]:
    """Run a full episode and return every frame — enough to replay it client-side."""
    env = AgriScoutEnv()
    predictor = _predictor(req.agent)
    obs, _ = env.reset(seed=req.seed)
    frames, cum, done = [], 0.0, False
    while not done:
        action = _act(req.agent, predictor, env, obs, req.deterministic)
        obs, reward, term, trunc, _ = env.step(action)
        cum += reward
        frames.append(_frame(env, action, reward, cum, req.include_grids))
        done = term or trunc
    return {
        "meta": {
            "agent": req.agent, "seed": req.seed, "deterministic": req.deterministic,
            "env_version": env.ENV_VERSION, "n_rows": env.n_rows, "n_cols": env.n_cols,
            "steps": env.step_count, "total_reward": round(env.cum_reward, 3),
            "success": bool(env.is_success),
            "final_health": round(env.mean_health, 4),
            "final_pest": round(env.mean_pest, 4),
        },
        "frames": frames,
    }


@app.post("/session")
def open_session(req: SessionRequest) -> dict[str, Any]:
    if len(_sessions) >= MAX_SESSIONS:
        raise HTTPException(429, f"too many open sessions (max {MAX_SESSIONS})")
    env = AgriScoutEnv()
    obs, _ = env.reset(seed=req.seed)
    sid = uuid.uuid4().hex[:12]
    _sessions[sid] = {"env": env, "obs": obs, "agent": req.agent,
                      "deterministic": req.deterministic, "cum": 0.0, "done": False}
    return {"session_id": sid,
            "state": _frame(env, ACTION_NAMES.index("WAIT"), 0.0, 0.0)}


def _session(sid: str) -> dict[str, Any]:
    if sid not in _sessions:
        raise HTTPException(404, f"unknown session {sid!r}")
    return _sessions[sid]


@app.get("/session/{sid}")
def get_session(sid: str) -> dict[str, Any]:
    s = _session(sid)
    return {"session_id": sid, "done": s["done"],
            "state": _frame(s["env"], ACTION_NAMES.index("WAIT"), 0.0, s["cum"])}


@app.post("/session/{sid}/step")
def step_session(sid: str) -> dict[str, Any]:
    """Advance one step with the AGENT choosing the action."""
    s = _session(sid)
    if s["done"]:
        raise HTTPException(409, "episode already finished; open a new session")
    env = s["env"]
    action = _act(s["agent"], _predictor(s["agent"]), env, s["obs"], s["deterministic"])
    return _advance(sid, s, action)


@app.post("/session/{sid}/act")
def act_session(sid: str, req: ActRequest) -> dict[str, Any]:
    """Advance one step with a CLIENT-chosen action — lets a UI drive manually."""
    s = _session(sid)
    if s["done"]:
        raise HTTPException(409, "episode already finished; open a new session")
    if req.action not in ACTION_NAMES:
        raise HTTPException(422, f"unknown action {req.action!r}; expected {ACTION_NAMES}")
    return _advance(sid, s, ACTION_NAMES.index(req.action))


def _advance(sid: str, s: dict[str, Any], action: int) -> dict[str, Any]:
    env = s["env"]
    obs, reward, term, trunc, _ = env.step(action)
    s["obs"], s["cum"] = obs, s["cum"] + reward
    s["done"] = bool(term or trunc)
    return {"session_id": sid, "done": s["done"],
            "state": _frame(env, action, reward, s["cum"]),
            "success": bool(env.is_success) if s["done"] else None}


@app.delete("/session/{sid}")
def close_session(sid: str) -> dict[str, str]:
    _session(sid)
    del _sessions[sid]
    return {"status": "closed", "session_id": sid}


@app.get("/")
def demo() -> FileResponse:
    """The bundled viewer — a worked example of a frontend consuming this API."""
    page = REPO / "assets" / "demo" / "index.html"
    if not page.exists():
        raise HTTPException(404, "build it: uv run python scripts/make_demo_html.py")
    return FileResponse(page)
