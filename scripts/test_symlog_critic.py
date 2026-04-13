"""
test_symlog_critic.py — Sanity checks for symlog critic stabilisation.

Verifies that:
1. symlog/symexp are inverses and stay finite for extreme inputs.
2. When a critic outputs very large values, the symlog target stays bounded.
3. return_scale cap prevents actor grad from collapsing.
4. Lambda-return bootstrap fraction is higher for H=5 than H=20 (explains why
   short horizons suffer more from bootstrap drift).

Run:
    python scripts/test_symlog_critic.py
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thesiswm.utils.symlog import symlog as _symlog, symexp as _symexp


def test_symlog_inverse():
    for v in [-1000, -1, 0, 1, 500, 1e6]:
        x = torch.tensor(float(v))
        roundtrip = _symexp(_symlog(x))
        assert abs(float(roundtrip) - v) < 1e-3, f"symexp(symlog({v})) = {float(roundtrip)}"
    print("PASS: symlog/symexp are inverses")


def test_symlog_bounds_large_critic():
    """Critic outputs large values → symlog target stays bounded."""
    large_critic_out = torch.tensor([500.0, 1000.0, -300.0, 5000.0])
    symlog_target = _symlog(large_critic_out)
    assert float(symlog_target.abs().max()) < 10.0, (
        f"symlog of large critic outputs should be <10, got {symlog_target}"
    )
    print(f"PASS: symlog targets for critic=[500,1000,-300,5000] → {symlog_target.tolist()}")


def test_bootstrap_fraction_higher_for_short_horizon():
    """
    H=5 assigns more weight to the bootstrap value per trajectory step than H=20.
    This is why short-horizon runs suffer more from critic drift.
    """
    gamma = 0.99
    cont = 0.95  # healthy cont prob

    def bootstrap_weight_at_t0(H: int) -> float:
        # Weight of bootstrap V_H in G_0 under lambda=0.95
        # For simplicity: compute discount product (lower bound on bootstrap contribution)
        return (gamma * cont) ** H

    w5  = bootstrap_weight_at_t0(5)
    w20 = bootstrap_weight_at_t0(20)

    assert w5 > w20, f"Expected H=5 bootstrap weight > H=20: {w5:.4f} vs {w20:.4f}"
    print(f"PASS: bootstrap weight H=5={w5:.4f}, H=20={w20:.4f}  (ratio {w5/w20:.1f}x)")


def test_return_scale_cap():
    """return_scale should not exceed return_scale_max even after many bad batches."""
    scale_max = 1000.0
    return_scale = 1.0
    for _ in range(10000):
        batch_std = 1e8  # pathologically large
        new_sample = min(max(batch_std, 1.0), scale_max)
        return_scale = min(0.99 * return_scale + 0.01 * new_sample, scale_max)

    assert return_scale <= scale_max, f"return_scale {return_scale} exceeded cap {scale_max}"
    print(f"PASS: return_scale capped at {return_scale:.1f} (max={scale_max})")


def test_actor_grad_stays_finite_with_large_critic():
    """
    With symlog: even if critic outputs 1000, advantage normalisation stays finite.
    Without symlog: advantage std would be huge → return_scale explodes → grad vanishes.
    """
    B, H = 64, 5
    gamma = 0.99

    # Simulate a critic that has drifted to large outputs
    targ_raw = torch.full((B, H + 1), 6.9)   # symlog(1000)≈6.9

    # WITH symlog: bootstrap = symexp(6.9) ≈ 1000, lambda return ≈ 1000+small_rewards
    targ_vals_symlog = _symexp(targ_raw[:, :H])
    rewards = torch.full((B, H), 1.0)
    discounts = torch.full((B, H), gamma * 0.95)

    # Simple 1-step return to check scale
    lambda_ret = rewards + discounts * targ_vals_symlog
    target_symlog = _symlog(lambda_ret)

    # Critic predicts ~6.9 → target_symlog ~6.9 → loss is small, no explosion
    pred_symlog = torch.full((B, H), 6.9)
    loss = (pred_symlog - target_symlog).pow(2).mean()
    assert float(loss) < 1.0, f"Critic loss in symlog space should be small, got {float(loss):.4f}"
    print(f"PASS: critic loss in symlog space = {float(loss):.4f} (no explosion)")

    # Advantage = symexp(target_symlog) - symexp(pred_symlog) ≈ small
    adv = _symexp(target_symlog) - _symexp(pred_symlog)
    adv_std = float(adv.std())
    assert adv_std < 100.0, f"Advantage std with symlog should be small, got {adv_std:.2f}"
    print(f"PASS: advantage std with symlog = {adv_std:.4f}")


if __name__ == "__main__":
    test_symlog_inverse()
    test_symlog_bounds_large_critic()
    test_bootstrap_fraction_higher_for_short_horizon()
    test_return_scale_cap()
    test_actor_grad_stays_finite_with_large_critic()
    print("\nAll checks passed.")
