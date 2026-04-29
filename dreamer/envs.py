"""MuJoCo Gymnasium environment factory."""
import numpy as np
import gymnasium as gym

from dreamer.parallel import ParallelEnv


class MuJoCoWrapper:
    """Wrap a gymnasium MuJoCo env to return r2dreamer-compatible obs dicts.

    Obs dict keys: obs, is_first, is_last, is_terminal.
    Reward is returned separately from step().
    """

    def __init__(
        self,
        env_id: str,
        seed: int = 0,
        action_repeat: int = 1,
        render_mode: str | None = None,
    ):
        self._env = gym.make(env_id, render_mode=render_mode)
        self._env.reset(seed=seed)
        self._action_repeat = action_repeat
        self._obs_shape = self._env.observation_space.shape

        obs_space = gym.spaces.Dict({
            "obs": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=self._obs_shape, dtype=np.float32
            ),
            "is_first": gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "is_last": gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            "is_terminal": gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32),
        })
        self.observation_space = obs_space

        act_space = self._env.action_space
        if not hasattr(act_space, "n"):
            # Mark continuous action spaces with normalized [-1,1] bounds
            low = np.full(act_space.shape, -1.0, dtype=np.float32)
            high = np.full(act_space.shape, 1.0, dtype=np.float32)
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            self._action_low = self._env.action_space.low
            self._action_high = self._env.action_space.high
            self._normalize_action = True
        else:
            self.action_space = act_space
            self._normalize_action = False

    def _denorm_action(self, action):
        """Map [-1,1] action back to env's native range."""
        lo, hi = self._action_low, self._action_high
        return (np.clip(action, -1.0, 1.0) + 1.0) / 2.0 * (hi - lo) + lo

    def _make_obs(self, raw_obs, is_first: bool, is_last: bool, is_terminal: bool):
        return {
            "obs": np.asarray(raw_obs, dtype=np.float32),
            "is_first": np.array([float(is_first)], dtype=np.float32),
            "is_last": np.array([float(is_last)], dtype=np.float32),
            "is_terminal": np.array([float(is_terminal)], dtype=np.float32),
        }

    def reset(self, seed: int | None = None):
        obs, _ = self._env.reset(seed=seed)
        return self._make_obs(obs, is_first=True, is_last=False, is_terminal=False)

    def step(self, action):
        if self._normalize_action:
            action = self._denorm_action(action)
        total_reward = 0.0
        for _ in range(self._action_repeat):
            obs, reward, terminated, truncated, info = self._env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        done = terminated or truncated
        result_obs = self._make_obs(obs, is_first=False, is_last=done, is_terminal=terminated)
        return result_obs, total_reward, done, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()


def make_envs(config):
    """Create train and eval parallel environments.

    Returns (train_envs, eval_envs, obs_space, act_space).
    """
    env_id = str(config.env.id)
    seed = int(config.seed)
    num_envs = int(config.env.num_envs)
    eval_num = int(config.env.eval_num_envs)
    action_repeat = int(config.env.action_repeat)
    device = str(config.device)

    def env_constructor(base_seed):
        def _init():
            return MuJoCoWrapper(env_id, seed=base_seed, action_repeat=action_repeat)
        return _init

    train_envs = ParallelEnv(env_constructor, num_envs, device)
    eval_envs = ParallelEnv(
        lambda i: (lambda: MuJoCoWrapper(env_id, seed=seed + 10000 + i, action_repeat=action_repeat)),
        eval_num,
        device,
    )
    obs_space = train_envs.observation_space
    act_space = train_envs.action_space
    return train_envs, eval_envs, obs_space, act_space
