"""
SIGReg: Characteristic-function Gaussian regularizer from Le-WM (Maes et al., 2025).

Penalises deviations of a batch of embeddings from an isotropic N(0,I) distribution
by comparing the empirical characteristic function to the ideal Gaussian one.

Reference: https://github.com/lucas-maes/le-wm
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """
    Pushes the marginal distribution of latent vectors toward N(0, I).

    Algorithm:
      1. Project the batch onto `num_proj` random unit vectors.
      2. Evaluate the empirical characteristic function (ECF) at `knots` points
         via cos/sin (real/imaginary parts of E[exp(i·t·x)]).
      3. Compare the ECF to the ideal Gaussian window exp(-t²/2).
      4. Return the weighted L2 error (Simpson-rule integration over t ∈ [0, 3]).

    Args:
        knots:    Number of integration points on [0, 3].  (default: 17)
        num_proj: Number of random 1-D projections.        (default: 1024)
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)

        # Trapezoid-rule weights scaled by 2: [dt, 2dt, ..., 2dt, dt].
        # Standard trapezoid would be [dt/2, dt, ..., dt, dt/2]; the 2× scaling is
        # absorbed into sigreg_weight and matches the Le-WM reference convention.
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt

        # Ideal Gaussian characteristic function: phi(t) = exp(-t²/2)
        phi = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)  # pre-multiplied for efficiency

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent tensor of shape [..., D].  The last dimension is the feature dim;
               all leading dimensions are treated as the batch.

        Returns:
            Scalar loss.
        """
        # Flatten to (N, D)
        N, D = z.shape[0], z.shape[-1]
        flat = z.reshape(-1, D).float()  # always float32 for numerical stability

        # Random unit projections: (D, num_proj)
        A = torch.randn(D, self.num_proj, device=flat.device, dtype=torch.float32)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)

        # Projected values: (N, num_proj)
        proj = flat @ A

        # ECF evaluation: (N, num_proj, knots)
        x_t = proj.unsqueeze(-1) * self.t  # broadcast over knots

        # Squared error between ECF and ideal Gaussian, averaged over projections
        # real part error: (E[cos(t·x)] - phi(t))²
        # imag part error: (E[sin(t·x)])²   (target imaginary part is 0 for symmetric dist)
        cos_mean = x_t.cos().mean(0)  # (num_proj, knots)
        sin_mean = x_t.sin().mean(0)  # (num_proj, knots)

        err = (cos_mean - self.phi).square() + sin_mean.square()  # (num_proj, knots)

        # Weighted integration over t, scale by N (match Le-WM convention)
        statistic = (err @ self.weights) * flat.shape[0]  # (num_proj,)
        return statistic.mean()
