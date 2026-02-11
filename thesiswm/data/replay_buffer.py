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
        Sample sequences of length seq_len. We avoid crossing episode boundaries by rejecting starts
        where any episode boundary occurs in the first (seq_len-1) steps.

        Returns arrays shaped:
          obs:        [B, T, obs_dim]   (current obs)
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

        starts = []
        max_tries = B * 50
        tries = 0
        while len(starts) < B and tries < max_tries:
            s = np.random.randint(0, n - T)
            # Reject if episode boundary is inside the sequence (excluding final step).
            if np.any(self.is_last[s : s + T - 1] > 0.5):
                tries += 1
                continue
            starts.append(s)
            tries += 1

        if len(starts) < B:
            # Fallback: allow boundaries if data is sparse, but this is less ideal.
            starts = list(np.random.randint(0, n - T, size=(B,)))

        obs = np.stack([self.obs[s : s + T] for s in starts], axis=0)
        actions = np.stack([self.actions[s : s + T] for s in starts], axis=0)
        rewards = np.stack([self.rewards[s : s + T] for s in starts], axis=0)
        terminated = np.stack([self.terminated[s : s + T] for s in starts], axis=0)
        is_last = np.stack([self.is_last[s : s + T] for s in starts], axis=0)
        next_obs = np.stack([self.next_obs[s : s + T] for s in starts], axis=0)

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "terminated": terminated,
            "is_last": is_last,
            "next_obs": next_obs,
        }
