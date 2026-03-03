from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: Callable = nn.ELU,
        out_activation: Optional[Callable] = None,
    ):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(activation())
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.out_activation = out_activation() if out_activation is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.out_activation is not None:
            y = self.out_activation(y)
        return y


@dataclass
class DiagGaussian:
    mean: torch.Tensor
    log_std: torch.Tensor

    def sample(self) -> torch.Tensor:
        std = torch.exp(self.log_std)
        eps = torch.randn_like(std)
        return self.mean + eps * std

    def kl_div(self, other: "DiagGaussian") -> torch.Tensor:
        """
        KL(self || other) for diagonal Gaussians, summed over the latent dimension.
        Returns shape [batch].

        KL(N(m1,s1) || N(m2,s2)) = sum_i [ log(s2_i/s1_i)
                                            + (s1_i^2 + (m1_i - m2_i)^2) / (2 s2_i^2)
                                            - 1/2 ]

        FIX: previous code computed log(s2/(s1+eps) + eps) which is numerically
        inconsistent (double epsilon distorts the ratio when s1 is small).
        Using log_std directly — log(s2/s1) = log_std_other - log_std_self — is
        exact and avoids any epsilon artefact since log_std is already stored.
        """
        s1_sq = torch.exp(2.0 * self.log_std)   # s1^2, numerically stable
        s2_sq = torch.exp(2.0 * other.log_std)
        term1 = other.log_std - self.log_std                                   # log(s2/s1)
        term2 = (s1_sq + (self.mean - other.mean).pow(2)) / (2.0 * s2_sq + 1e-8)
        kl = (term1 + term2 - 0.5).sum(dim=-1)
        return kl
