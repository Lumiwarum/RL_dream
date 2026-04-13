from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from thesiswm.models.networks import MLP

_LOG_2PI = math.log(2.0 * math.pi)


class TanhGaussianPolicy(nn.Module):
    """
    Gaussian policy with tanh squashing.
    """
    def __init__(self, in_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.net = MLP(in_dim, 2 * act_dim, hidden_dim=hidden_dim, num_layers=2)
        self.act_dim = act_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.net(x)
        mean, log_std = torch.chunk(params, 2, dim=-1)
        # Tanh bounds mean to (-1,1): prevents unbounded MLP output that saturates the
        # outer tanh in sample() and zeroes out the pathwise gradient.
        mean = torch.tanh(mean)
        # std ∈ [exp(-1), exp(0)] = [0.37, 1.00].
        # Floor: prevents collapse. Ceiling: keeps pre-tanh samples in (-3,+3) where
        # the tanh Jacobian ≥ 0.09, preserving non-zero pathwise gradient.
        log_std = torch.clamp(log_std, -1.0, 0.0)
        return mean, log_std

    def sample(self, x: torch.Tensor):
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        pre_tanh = mean + eps * std
        action = torch.tanh(pre_tanh)
        log_prob = -0.5 * (
            ((pre_tanh - mean) / (std + 1e-8)) ** 2
            + 2 * log_std
            + _LOG_2PI
        ).sum(dim=-1)
        # Tanh Jacobian correction: log|d(tanh(u))/du| = log(1 - tanh²(u))
        log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob

    def mean_action(self, x: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(x)
        return mean  # forward() already applies tanh


class ValueNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = MLP(in_dim, 1, hidden_dim=hidden_dim, num_layers=2)
        # Near-zero output init: with symlog critic, symexp(~0)=0 at startup so the baseline
        # stays ≈0 until the critic has seen data. Without this, random MLP outputs of ±15
        # map through symexp to ±3e6, causing advantage explosion in the first few batches.
        nn.init.constant_(self.net.net[-1].bias, 0.0)
        nn.init.uniform_(self.net.net[-1].weight, -0.01, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Actor-Critic operating on RSSM features (h,z).
    """
    def __init__(self, feat_dim: int, act_dim: int, actor_hidden: int, critic_hidden: int):
        super().__init__()
        self.actor = TanhGaussianPolicy(feat_dim, act_dim, actor_hidden)
        self.critic = ValueNet(feat_dim, critic_hidden)

