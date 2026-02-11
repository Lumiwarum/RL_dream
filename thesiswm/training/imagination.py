from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from thesiswm.models.rssm import EnsembleWorldModel

def lambda_returns(
    rewards: torch.Tensor,        # [B,H]
    values: torch.Tensor,         # [B,H+1]
    discounts: torch.Tensor,      # [B,H]  (already includes gamma and (1-done))
    lambda_: float,
) -> torch.Tensor:
    """
    Generalized lambda-returns with per-step discounts.

    returns[t] = r_t + d_t * ((1-lam) * V_{t+1} + lam * returns[t+1])
    """
    B, H = rewards.shape
    returns = torch.zeros((B, H), device=rewards.device, dtype=rewards.dtype)

    next_ret = values[:, -1]  # bootstrap V_{H}
    for t in reversed(range(H)):
        next_val = values[:, t + 1]
        next_ret = rewards[:, t] + discounts[:, t] * ((1.0 - lambda_) * next_val + lambda_ * next_ret)
        returns[:, t] = next_ret
    return returns


@torch.no_grad()
def decide_horizon(
    ensemble: EnsembleWorldModel,
    obs: torch.Tensor,        # [B, obs_dim]
    action: torch.Tensor,     # [B, act_dim]
    metric: str,
    thresh_high: float,
    thresh_mid: float,
    horizons: Tuple[int, int, int] = (5, 10, 20),
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, torch.Tensor]:
    """
    Choose a single horizon for the batch based on mean disagreement.

    If mean_uncertainty > high -> H=5
    elif > mid -> H=10
    else -> H=20
    """
    u = ensemble.disagreement(obs=obs, action=action, metric=metric, device=device)  # [B]
    m = u.mean()
    h5, h10, h20 = horizons
    if float(m) > float(thresh_high):
        return h5, u
    if float(m) > float(thresh_mid):
        return h10, u
    return h20, u
