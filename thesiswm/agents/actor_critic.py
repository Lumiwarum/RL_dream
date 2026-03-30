from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from thesiswm.models.networks import MLP
from thesiswm.models.rssm import EnsembleWorldModel, RSSMState


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
        log_std = torch.clamp(log_std, -5.0, 2.0)
        return mean, log_std

    def sample(self, x: torch.Tensor):
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        pre_tanh = mean + eps * std
        action = torch.tanh(pre_tanh)

        # log_prob with tanh correction
        # Use math.pi for precision and clamp action to prevent boundary issues
        log_prob = -0.5 * (
            ((pre_tanh - mean) / (std + 1e-8)) ** 2
            + 2 * log_std
            + torch.log(torch.tensor(2.0 * math.pi, device=x.device, dtype=std.dtype))
        ).sum(dim=-1)

        # Jacobian correction for tanh squashing: log|d(tanh(u))/du| = log(1 - tanh²(u))
        # Use epsilon inside the log (not clamp) so gradient flows correctly for saturated actions.
        log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        return action, log_prob

    def mean_action(self, x: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(x)
        return torch.tanh(mean)


class ValueNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = MLP(in_dim, 1, hidden_dim=hidden_dim, num_layers=2)

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

    @torch.no_grad()
    def act_deterministic_from_obs(
        self,
        obs: torch.Tensor,  # [B, obs_dim]
        ensemble: EnsembleWorldModel,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Deterministic evaluation action using model[0] posterior features from the single-step obs.
        """
        wm0 = ensemble.models[0]
        obs_seq = obs.unsqueeze(1)
        # dummy action (zeros) to update GRU once; we only need a feature embedding from obs.
        act_dim = wm0.rssm.act_dim
        act_seq = torch.zeros((obs.shape[0], 1, act_dim), device=device)
        states, post, prior = wm0.rssm.observe_sequence(obs_seq, act_seq, device=device, sample=False)
        feat = wm0.features(states[-1])
        return self.actor.mean_action(feat)

    @torch.no_grad()
    def act_stochastic_from_obs(
        self,
        obs: torch.Tensor,
        ensemble: EnsembleWorldModel,
        device: torch.device,
    ) -> torch.Tensor:
        wm0 = ensemble.models[0]
        obs_seq = obs.unsqueeze(1)
        act_dim = wm0.rssm.act_dim
        act_seq = torch.zeros((obs.shape[0], 1, act_dim), device=device)
        states, _, _ = wm0.rssm.observe_sequence(obs_seq, act_seq, device=device, sample=True)
        feat = wm0.features(states[-1])
        a, _ = self.actor.sample(feat)
        return a
