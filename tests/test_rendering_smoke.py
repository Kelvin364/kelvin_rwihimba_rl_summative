"""Smoke test for the PyBullet renderer + EpisodeRecorder.

Runs 200 random steps in headless DIRECT mode, records a trace, and asserts a
valid JSON file is produced with the expected schema. No display required.

Run via pytest, or directly to print the trace size and first frame::

    uv run pytest tests/test_rendering_smoke.py -q
    uv run python tests/test_rendering_smoke.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ENV_VERSION = "agriscout-mock-v0"


class MockAgriScoutEnv:
    """Minimal stand-in exposing the render-state contract of the real env.

    Dynamics are only rich enough to make the grids/levels visibly evolve so the
    renderer and recorder are exercised; this is NOT the real environment.
    """

    ACTION_NAMES = [
        "MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W",
        "SCAN", "IRRIGATE", "SPRAY", "RETURN_TO_DEPOT", "WAIT",
    ]
    ENV_VERSION = ENV_VERSION

    def __init__(self, n_rows: int = 8, n_cols: int = 12, seed: int = 0) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self.health_grid = self.rng.uniform(0.5, 1.0, (self.n_rows, self.n_cols))
        self.pest_grid = self.rng.uniform(0.0, 0.3, (self.n_rows, self.n_cols))
        self.irrigation_grid = np.zeros((self.n_rows, self.n_cols), dtype=int)
        self.rover_row = 0.0
        self.rover_col = 0.0
        self.rover_heading = 0.0
        self.battery = 1.0
        self.water = 1.0
        self.pesticide = 1.0
        self.step_count = 0
        self.last_action = -1
        self.last_reward = 0.0
        self.cum_reward = 0.0

    def _cell(self) -> tuple[int, int]:
        r = int(np.clip(round(self.rover_row), 0, self.n_rows - 1))
        c = int(np.clip(round(self.rover_col), 0, self.n_cols - 1))
        return r, c

    def step(self, action: int) -> float:
        self.step_count += 1
        self.last_action = int(action)
        name = self.ACTION_NAMES[action]

        # World drift each step.
        self.health_grid = np.clip(self.health_grid - self.rng.uniform(0, 0.01, self.health_grid.shape), 0, 1)
        self.pest_grid = np.clip(self.pest_grid + self.rng.uniform(0, 0.02, self.pest_grid.shape), 0, 1)
        self.irrigation_grid = np.maximum(0, self.irrigation_grid - 1)
        self.battery = max(0.0, self.battery - 0.003)

        r, c = self._cell()
        reward = -0.01
        if name == "MOVE_N":
            self.rover_row = min(self.n_rows - 1, self.rover_row + 1); self.rover_heading = math.pi / 2
        elif name == "MOVE_S":
            self.rover_row = max(0, self.rover_row - 1); self.rover_heading = -math.pi / 2
        elif name == "MOVE_E":
            self.rover_col = min(self.n_cols - 1, self.rover_col + 1); self.rover_heading = 0.0
        elif name == "MOVE_W":
            self.rover_col = max(0, self.rover_col - 1); self.rover_heading = math.pi
        elif name == "IRRIGATE" and self.water > 0:
            self.irrigation_grid[r, c] = 10
            self.health_grid[r, c] = min(1.0, self.health_grid[r, c] + 0.2)
            self.water = max(0.0, self.water - 0.05)
            reward = 0.5
        elif name == "SPRAY" and self.pesticide > 0:
            reward = 0.5 * self.pest_grid[r, c]
            self.pest_grid[r, c] = max(0.0, self.pest_grid[r, c] - 0.4)
            self.pesticide = max(0.0, self.pesticide - 0.05)
        elif name == "SCAN":
            self.battery = max(0.0, self.battery - 0.005)
        elif name == "RETURN_TO_DEPOT":
            if (r, c) == (0, 0):
                self.water = 1.0; self.pesticide = 1.0

        self.last_reward = float(reward)
        self.cum_reward += reward
        return reward


def run_episode(results_dir: Path, steps: int = 200, seed: int = 0):
    """Run `steps` random steps headless, record, and save. Returns the path."""
    from environment.rendering import AgriScoutRenderer, EpisodeRecorder

    env = MockAgriScoutEnv(seed=seed)
    renderer = AgriScoutRenderer(env, mode="direct")  # forced headless DIRECT
    recorder = EpisodeRecorder(
        run_id=f"smoke_{seed}",
        meta={"env_version": env.ENV_VERSION, "model": "random", "seed": seed},
    )
    rng = np.random.default_rng(seed)
    try:
        for _ in range(steps):
            action = int(rng.integers(0, len(env.ACTION_NAMES)))
            reward = env.step(action)
            renderer.render()
            recorder.record(env, action, reward)
    finally:
        renderer.close()
    return recorder.save(results_dir=results_dir)


def test_render_direct_writes_valid_trace(tmp_path):
    path = run_episode(tmp_path, steps=200, seed=0)

    assert path.exists()
    assert path == tmp_path / "traces" / "smoke_0.json"

    data = json.loads(path.read_text())
    assert set(data) == {"meta", "frames"}
    assert data["meta"] == {"env_version": ENV_VERSION, "model": "random", "seed": 0}
    assert len(data["frames"]) == 200

    f0 = data["frames"][0]
    assert set(f0) == {
        "t", "rover", "battery", "water", "pesticide",
        "health_grid", "pest_grid", "action", "reward", "cum_reward",
    }
    assert set(f0["rover"]) == {"x", "y", "heading"}
    assert f0["action"] in MockAgriScoutEnv.ACTION_NAMES
    # Grids are downsampled to 8x12 lists of 2dp floats.
    assert len(f0["health_grid"]) == 8 and len(f0["health_grid"][0]) == 12
    assert all(round(v, 2) == v for row in f0["health_grid"] for v in row)
    # cum_reward is a running sum (rounded from the exact float sum, so allow
    # for accumulated rounding noise vs. summing already-rounded per-frame rewards).
    assert math.isclose(
        data["frames"][-1]["cum_reward"],
        sum(fr["reward"] for fr in data["frames"]),
        abs_tol=0.2,
    )
    # cum_reward must be monotone in the sense of matching a prefix running sum.
    running = 0.0
    for fr in data["frames"]:
        running += fr["reward"]
        assert math.isclose(fr["cum_reward"], running, abs_tol=0.2)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
    out = Path(os.environ.get("AGRISCOUT_RESULTS", "./logs"))
    path = run_episode(out, steps=200, seed=0)
    size = path.stat().st_size
    data = json.loads(path.read_text())
    print(f"trace path : {path}")
    print(f"trace size : {size} bytes ({size / 1024:.1f} KiB)")
    print(f"frames     : {len(data['frames'])}")
    print("meta       :", json.dumps(data["meta"]))
    print("first frame:")
    print(json.dumps(data["frames"][0], indent=2))
