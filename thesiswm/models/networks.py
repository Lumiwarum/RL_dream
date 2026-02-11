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
        KL(self || other) for diagonal Gaussians.
        Returns shape [batch].
        """
        # KL between N(m1,s1) and N(m2,s2):
        # log(s2/s1) + (s1^2 + (m1-m2)^2) / (2 s2^2) - 1/2
        s1 = torch.exp(self.log_std)
        s2 = torch.exp(other.log_std)
        term1 = torch.log(s2 / (s1 + 1e-8) + 1e-8)
        term2 = (s1.pow(2) + (self.mean - other.mean).pow(2)) / (2.0 * s2.pow(2) + 1e-8)
        kl = (term1 + term2 - 0.5).sum(dim=-1)
        return kl
