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
