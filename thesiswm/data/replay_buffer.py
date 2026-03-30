from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class ReplayState:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    is_last: np.ndarray
    next_obs: np.ndarray
    idx: int
    full: bool


class ReplayBuffer:
    """
    Simple numpy replay buffer storing transitions:
      (obs, action, reward, done, next_obs)

    Also supports sampling contiguous sequences for world-model training.
    """
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.terminated = np.zeros((capacity,), dtype=np.float32)
        self.is_last = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        next_obs: np.ndarray,
    ) -> None:
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = np.float32(reward)
        self.terminated[self.idx] = np.float32(terminated)
        self.is_last[self.idx] = np.float32(bool(terminated or truncated))
        self.next_obs[self.idx] = next_obs

        self.idx += 1
        if self.idx >= self.capacity:
            self.idx = 0
            self.full = True

    def state_dict(self) -> Dict:
        st = ReplayState(
            obs=self.obs,
            actions=self.actions,
            rewards=self.rewards,
            terminated=self.terminated,
            is_last=self.is_last,
            next_obs=self.next_obs,
            idx=self.idx,
            full=self.full,
        )
        return {"replay_state": st}

    def load_state_dict(self, d: Dict) -> None:
        st: ReplayState = d["replay_state"]
        self.obs = st.obs
        self.actions = st.actions
        self.rewards = st.rewards
        self.terminated = st.terminated
        self.is_last = st.is_last
        self.next_obs = st.next_obs
        self.idx = int(st.idx)
        self.full = bool(st.full)

    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        n = len(self)
        assert n > 0
        idxs = np.random.randint(0, n, size=(batch_size,))
        return {
            "obs": self.obs[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "terminated": self.terminated[idxs],
            "is_last": self.is_last[idxs],
            "next_obs": self.next_obs[idxs],
        }

    def sample_sequences(self, batch_size: int, seq_len: int) -> Dict[str, np.ndarray]:
        """
        Sample sequences of length seq_len. Avoids crossing episode boundaries by rejecting
        starts where any is_last occurs in the first (seq_len-1) steps.

        Uses fully-vectorized numpy operations instead of a Python loop — O(1) numpy
        calls regardless of batch size, eliminating the main CPU bottleneck as the buffer
        grows large and causes cache pressure.

        Returns arrays shaped:
          obs:        [B, T, obs_dim]
          actions:    [B, T, act_dim]
          rewards:    [B, T]
          terminated: [B, T]
          is_last:    [B, T]
          next_obs:   [B, T, obs_dim]
        """
        n = len(self)
        assert n >= seq_len + 1, "Not enough data for sequence sampling."
        B = int(batch_size)
        T = int(seq_len)
        upper = n - T  # exclusive upper bound for start index

        # --- Vectorized valid-start selection ---
        # Generate a large pool, check all boundaries at once, take first B valid.
        pool_size = min(B * 16, upper)
        candidates = np.random.randint(0, upper, size=pool_size)

        # Boundary check: is_last[s : s+T-1] must be all zeros.
        # Shape: [pool_size, T-1] — single numpy fancy-index, no Python loop.
        win_idx = candidates[:, None] + np.arange(T - 1)[None, :]   # [P, T-1]
        has_boundary = self.is_last[win_idx].any(axis=1)              # [P]
        valid = candidates[~has_boundary]

        if len(valid) >= B:
            starts = valid[:B]
        else:
            # Not enough clean sequences — top up with possibly boundary-crossing ones.
            shortfall = B - len(valid)
            extra = np.random.randint(0, upper, size=(shortfall,))
            starts = np.concatenate([valid, extra])

        # --- Vectorized gather: one numpy fancy-index call per field ---
        idxs = starts[:, None] + np.arange(T)[None, :]  # [B, T]

        return {
            "obs":        self.obs[idxs],         # [B, T, obs_dim]
            "actions":    self.actions[idxs],      # [B, T, act_dim]
            "rewards":    self.rewards[idxs],      # [B, T]
            "terminated": self.terminated[idxs],   # [B, T]
            "is_last":    self.is_last[idxs],      # [B, T]
            "next_obs":   self.next_obs[idxs],     # [B, T, obs_dim]
        }
