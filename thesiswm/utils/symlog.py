"""
DreamerV3 §A.2 symlog/symexp transforms.

The critic is trained to predict symlog(V). Large values are compressed
(symlog(500)≈6.2), which breaks the bootstrap feedback loop that causes
V→targets→V divergence. Bootstrap values are converted back via symexp
before lambda-return computation so reward/return units stay correct.
"""
from __future__ import annotations

import torch


def symlog(x: torch.Tensor) -> torch.Tensor:
    """symlog(x) = sign(x) * log(|x| + 1). Near-linear for small x, log-compressed for large."""
    return x.sign() * (x.abs() + 1.0).log()


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: symexp(x) = sign(x) * (exp(|x|) - 1)."""
    return x.sign() * (x.abs().exp() - 1.0)
