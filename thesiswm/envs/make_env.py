from __future__ import annotations
from typing import Optional

import gymnasium as gym
import numpy as np


def make_env(env_id: str, seed: int, render_mode: Optional[str] = None) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode)
    env.reset(seed=seed)

    # Agent operates in [-1,1]
    env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
    env = gym.wrappers.ClipAction(env)

    # Force the exposed action_space to be finite [-1,1] (some gymnasium versions
    # end up showing [-inf, inf] after wrappers)
    env.action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=env.unwrapped.action_space.shape, dtype=np.float32
    )
    return env


def make_vec_env(env_id: str, num_envs: int, seed: int) -> gym.vector.VectorEnv:
    """
    Create N parallel environments using AsyncVectorEnv (each in its own process).
    Each sub-env gets RescaleAction + ClipAction wrappers identical to make_env.
    Auto-resets on episode end — no manual env.reset() needed in the collection loop.
    """
    def _make_fn(seed_offset: int):
        def _thunk():
            env = gym.make(env_id)
            env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
            env = gym.wrappers.ClipAction(env)
            env.action_space = gym.spaces.Box(
                low=-1.0, high=1.0,
                shape=env.unwrapped.action_space.shape,
                dtype=np.float32,
            )
            env.reset(seed=seed + seed_offset)
            return env
        return _thunk

    return gym.vector.AsyncVectorEnv([_make_fn(i) for i in range(num_envs)])
