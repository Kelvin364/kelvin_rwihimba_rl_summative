"""Canonical entry point for my custom environment.

I keep the implementation in :mod:`environment.agriscout_env` under its
domain-specific name, which is what I refer to throughout the report and the
training code. This module is the stable import path:

    from environment.custom_env import AgriScoutEnv, make_env
"""

from __future__ import annotations

from environment.agriscout_env import (  # noqa: F401
    ACTION_NAMES,
    ENV_VERSION,
    SUCCESS_HEALTH,
    SUCCESS_PEST,
    AgriScoutEnv,
    make_env,
)

# Generic alias, so `from environment.custom_env import CustomEnv` also works.
CustomEnv = AgriScoutEnv

__all__ = [
    "AgriScoutEnv", "CustomEnv", "make_env", "ACTION_NAMES", "ENV_VERSION",
    "SUCCESS_HEALTH", "SUCCESS_PEST",
]
