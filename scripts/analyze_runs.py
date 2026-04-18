"""
analyze_runs.py — Full diagnostic report for all ThesisWM experiment runs.

Phase 1 (always): parallel TB scan with JSON caching — instant on repeated calls.
Phase 2 (always): per-flagged-run time-series breakdown (trend, phases, correlations).
Phase 3 (opt): --deep N: load the N worst checkpoints and run diagnose-style analysis.

Output:
    report_<timestamp>.txt  — compact text file, easy to paste to Claude/read yourself.

Usage:
    python scripts/analyze_runs.py                    # scan + TB analysis
    python scripts/analyze_runs.py --deep 3           # + checkpoint analysis for worst 3
    python scripts/analyze_runs.py --filter hopper    # subset of runs
    python scripts/analyze_runs.py --out my_report    # custom output prefix
    python scripts/analyze_runs.py --no_save          # print only, no file
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    # Minimal fallback — no progress bar, just pass-through
    def _tqdm(it, **kwargs):  # type: ignore[misc]
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", "?")
        print(f"{desc} ({total} items) …", flush=True)
        return it

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── reuse scan_logs helpers ────────────────────────────────────────────────────
from scan_logs import (  # noqa: E402
    CONT_THRESHOLDS,
    _parse_name, _tail, _head, _fmt,
    _cache_valid, _load_cache_arrays, _save_cache,
    _stream_scalars, _compute_stats,
)

# ══════════════════════════════════════════════════════════════════════════════
# TB-only deep analysis (no checkpoint needed)
# ══════════════════════════════════════════════════════════════════════════════

def _trend(vals: np.ndarray, label: str) -> str:
    """Linear trend over last half of series."""
    if len(vals) < 10:
        return f"{label}: too few points"
    half = vals[len(vals) // 2:]
    x = np.arange(len(half), dtype=float)
    slope = np.polyfit(x, half, 1)[0]
    direction = "↑" if slope > 0.5 else ("↓" if slope < -0.5 else "→")
    return f"{label}: slope={slope:+.2f}/step {direction}"


def _phase_detect(eval_steps: np.ndarray, eval_vals: np.ndarray) -> str:
    """Detect training phase: warmup / learning / plateau / degrading."""
    if len(eval_vals) < 6:
        return "phase=INSUFFICIENT_DATA"
    q1 = eval_vals[:len(eval_vals)//3]
    q3 = eval_vals[2*len(eval_vals)//3:]
    delta = np.mean(q3) - np.mean(q1)
    std_q3 = np.std(q3)
    max_v = np.max(eval_vals)
    final = np.mean(eval_vals[-5:])
    if delta > 30:
        return f"phase=LEARNING  Δ={delta:+.0f} (q1→q3)"
    if delta > -10 and std_q3 < 15:
        return f"phase=PLATEAU  std_last_third={std_q3:.1f}"
    if max_v - final > 30:
        return f"phase=DEGRADING  peak={max_v:.0f} → final={final:.0f} (drop={max_v-final:.0f})"
    return f"phase=FLAT  Δ={delta:+.0f}"


def _horizon_dist(hor_vals: np.ndarray) -> str:
    if len(hor_vals) == 0:
        return "N/A"
    unique, counts = np.unique(hor_vals, return_counts=True)
    total = len(hor_vals)
    parts = [f"H={int(u)}:{counts[i]/total*100:.0f}%" for i, u in enumerate(unique)]
    return "  ".join(parts)


def _horizon_trend(hor_vals: np.ndarray) -> str:
    """
    Show horizon distribution split early/late to detect dynamic switching.

    Expected adaptive behaviour: H=5 more frequent early (high WM uncertainty),
    H=20 more frequent late (low uncertainty after WM converges).
    Returns a string showing avg horizon in first/last third and the trend direction.
    """
    if len(hor_vals) < 10:
        return ""
    n = len(hor_vals)
    third = max(1, n // 3)
    early = hor_vals[:third]
    late  = hor_vals[-third:]
    h_early = float(np.mean(early))
    h_late  = float(np.mean(late))
    delta   = h_late - h_early
    sign    = "↑" if delta > 0.5 else ("↓" if delta < -0.5 else "→")
    return (
        f"horizon_trend: early_avg={h_early:.1f}  late_avg={h_late:.1f}  "
        f"Δ={delta:+.1f} {sign}  "
        f"[early: {_horizon_dist(early)}  |  late: {_horizon_dist(late)}]"
    )


def _corr(a: np.ndarray, b: np.ndarray, label: str) -> str:
    n = min(len(a), len(b))
    if n < 5:
        return ""
    a2, b2 = a[-n:], b[-n:]
    if a2.std() < 1e-8 or b2.std() < 1e-8:
        return ""
    r = float(np.corrcoef(a2, b2)[0, 1])
    return f"  corr({label})={r:+.2f}"


def _phase_vals(vals: np.ndarray, which: str) -> np.ndarray:
    """Return early/mid/late third of an array."""
    if len(vals) < 3:
        return vals
    n = len(vals)
    t = n // 3
    if which == "early": return vals[:t]
    if which == "mid":   return vals[t:2*t]
    return vals[2*t:]


def _wm_health(arrays: dict) -> list[str]:
    """
    Diagnose world model learning from loss components.

    Healthy WM:
      obs_loss:    starts high (~1-3), trends DOWN to <0.3
      rew_loss:    starts high, trends DOWN to <0.05
      kl_loss:     initially 0 (prior = posterior), climbs to ~free_nats*latent_dim,
                   then stabilises. If stuck at 0 → posterior collapse.
      sigreg_loss: small (<0.01), stable
      wm_loss:     trending DOWN overall
    """
    lines = []

    def _vals(tag):
        _, v = arrays.get(tag, (np.array([]), np.array([])))
        return v

    obs_l    = _vals("loss/obs")
    rew_l    = _vals("loss/reward")
    kl_l     = _vals("loss/kl")
    sigreg_l = _vals("loss/sigreg")
    wm_l     = _vals("loss/world_model")

    if len(wm_l) == 0:
        lines.append("  wm_health: NO DATA (loss/* tags not in cache — delete .scan_cache.json and re-run)")
        return lines

    def _summary(v, name, good_fn, unit=""):
        if len(v) == 0:
            return
        early_m = float(np.mean(_phase_vals(v, "early")))
        late_m  = float(np.mean(_phase_vals(v, "late")))
        delta   = late_m - early_m
        sign    = "↓" if delta < -0.01 * abs(early_m + 1e-9) else ("↑" if delta > 0.01 * abs(early_m + 1e-9) else "→")
        ok      = "✓" if good_fn(early_m, late_m) else "✗"
        lines.append(f"  {name:<12} early={early_m:.4f}  late={late_m:.4f}  {sign}  {ok}{unit}")

    _summary(obs_l,    "obs_loss",    lambda e, l: l < e * 0.8 or l < 0.5,  "  (↓ = WM learns obs)")
    _summary(rew_l,    "rew_loss",    lambda e, l: l < e * 0.8 or l < 0.1,  "  (↓ = WM learns rew)")
    _summary(kl_l,     "kl_loss",     lambda e, l: l > 0.5,                 "  (>0 = posterior active)")
    _summary(sigreg_l, "sigreg_loss", lambda e, l: l < 5.0,                 "  (×0.01 weight; <5.0 raw = normal)")
    _summary(wm_l,     "wm_total",    lambda e, l: l < e * 0.9,             "")

    # Flag posterior collapse: kl stays near 0
    if len(kl_l) > 20 and float(np.mean(_phase_vals(kl_l, "late"))) < 0.1:
        lines.append("  ⚠ POSTERIOR_COLLAPSE: kl_loss ≈ 0 → posterior matches prior, latent encodes nothing")

    # Flag obs not learning
    if len(obs_l) > 20:
        late_obs = float(np.mean(_phase_vals(obs_l, "late")))
        early_obs = float(np.mean(_phase_vals(obs_l, "early")))
        if late_obs > early_obs * 0.95 and late_obs > 0.5:
            lines.append("  ⚠ WM_NOT_LEARNING: obs_loss not decreasing → check lr, batch size, replay")

    return lines


def _actor_critic_health(arrays: dict) -> list[str]:
    """
    Diagnose actor-critic learning.

    Healthy AC:
      policy_std:  starts ~1.0, DECREASES over training → policy becoming deterministic
      rscale:      stable below 100; if at cap (=100) → gradients too small
      actor_grad:  0.02-0.2 range, not trending to zero
      critic_loss: trending DOWN
      actor_loss:  should be negative (maximising returns)
    """
    lines = []

    def _vals(tag):
        # try both "imagine/*" and "imagination/*" variants
        _, v = arrays.get(tag, (np.array([]), np.array([])))
        if len(v) == 0:
            alt = tag.replace("imagine/", "imagination/").replace("imagination/", "imagine/")
            _, v = arrays.get(alt, (np.array([]), np.array([])))
        return v

    std_v    = _vals("policy/std_mean")
    rscale_v = _vals("value/return_scale")
    agrad_v  = _vals("grad/actor_norm")
    cgrad_v  = _vals("grad/critic_norm")
    aloss_v  = _vals("loss/actor")
    closs_v  = _vals("loss/critic")

    if len(std_v) == 0 and len(rscale_v) == 0:
        lines.append("  ac_health: NO DATA")
        return lines

    def _row(v, name, good_fn, note=""):
        if len(v) == 0:
            return
        early = float(np.mean(_phase_vals(v, "early")))
        late  = float(np.mean(_phase_vals(v, "late")))
        delta = late - early
        sign  = "↑" if delta > 0.001 * abs(early + 1e-9) else ("↓" if delta < -0.001 * abs(early + 1e-9) else "→")
        ok    = "✓" if good_fn(early, late) else "✗"
        lines.append(f"  {name:<14} early={early:.4f}  late={late:.4f}  {sign}  {ok}  {note}")

    _row(std_v,    "policy_std",   lambda e, l: l < e - 0.02,           "(↓ = policy learning)")
    _row(rscale_v, "return_scale", lambda e, l: np.mean(rscale_v) < 95, "(< 100 = gradients OK)")
    _row(agrad_v,  "actor_grad",   lambda e, l: np.mean(agrad_v) > 0.01,"(> 0.01 = learning)")
    _row(cgrad_v,  "critic_grad",  lambda e, l: True,                    "")
    _row(aloss_v,  "actor_loss",   lambda e, l: True,                    "(negative = maximising reward)")
    _row(closs_v,  "critic_loss",  lambda e, l: l < e * 0.8 or l < 0.1,"(↓ = critic converging)")

    # Flags
    if len(rscale_v) > 20 and float(np.mean(_phase_vals(rscale_v, "late"))) >= 99:
        lines.append("  ⚠ RSCALE_CAPPED: return_scale at max → actor_grad tiny → policy frozen")
    if len(std_v) > 20 and float(np.mean(_phase_vals(std_v, "late"))) > 0.97:
        lines.append("  ⚠ POLICY_FROZEN: policy_std ≈ 1.0 throughout → actor not learning")
    if len(agrad_v) > 20 and float(np.mean(_phase_vals(agrad_v, "late"))) < 0.005:
        lines.append("  ⚠ ACTOR_GRAD_TINY: gradient near zero → no policy improvement possible")

    return lines


def _diagnose_summary(arrays: dict, flags: list[str]) -> list[str]:
    """
    One-line root cause diagnosis with recommended fix.
    Returns list of "DIAGNOSIS: ..." lines.
    """
    diag = []

    def _vals(tag):
        _, v = arrays.get(tag, (np.array([]), np.array([])))
        return v

    std_v      = _vals("policy/std_mean")
    rscale_v   = _vals("value/return_scale")
    agrad_v    = _vals("grad/actor_norm")
    obs_l      = _vals("loss/obs")
    kl_l       = _vals("loss/kl")
    eval_v     = _vals("eval/return_mean")
    aloss_v    = _vals("loss/actor")

    # Check specific patterns
    wm_slow = (len(obs_l) > 20 and
               float(np.mean(_phase_vals(obs_l, "late"))) > 0.8)

    wm_broken = (len(obs_l) > 20 and
                 float(np.mean(_phase_vals(obs_l, "late"))) > float(np.mean(_phase_vals(obs_l, "early"))) * 0.95
                 and float(np.mean(_phase_vals(obs_l, "late"))) > 0.5)

    rscale_capped = (len(rscale_v) > 20 and
                     float(np.mean(_phase_vals(rscale_v, "late"))) >= 99)

    policy_frozen = (len(std_v) > 20 and
                     float(np.mean(_phase_vals(std_v, "late"))) > 0.97)

    grad_tiny = (len(agrad_v) > 20 and
                 float(np.mean(_phase_vals(agrad_v, "late"))) < 0.005)

    actor_loss_positive = (len(aloss_v) > 20 and
                           float(np.mean(_phase_vals(aloss_v, "late"))) > 0.003)

    spike_drop = any("SPIKE_THEN_DROP" in f for f in flags)
    plateau    = any("PLATEAU" in f for f in flags)

    if wm_broken:
        diag.append("DIAGNOSIS: WM not learning obs → actor gets garbage bootstrap values")
        diag.append("  FIX: check replay is populated; try wm.lr: 3e-4→1e-4; disable use_amp")
    if rscale_capped and not wm_broken:
        diag.append("DIAGNOSIS: return_scale at cap → actor_grad dying → policy frozen")
        diag.append("  FIX: symlog_clamp already at 5.0; check entropy_coef=3e-4 is applied")
    if policy_frozen and spike_drop:
        diag.append("DIAGNOSIS: SPIKE_THEN_DROP — entropy dominated early, then sudden collapse")
        diag.append("  FIX: entropy_coef=3e-4 (was 1e-2); critic_lr=3e-5 (was 1e-4) — apply these!")
    elif policy_frozen and not spike_drop:
        diag.append("DIAGNOSIS: Policy std stuck at 1.0 without spike — actor not getting signal")
        diag.append("  FIX: check actor_start_step; check rscale; check WM reward head")
    if spike_drop and not policy_frozen and wm_slow:
        diag.append("DIAGNOSIS: SPIKE_THEN_DROP with WM still learning — complex env (Walker2d?)")
        diag.append("  Likely: actor learned on inaccurate WM, then WM updated and invalidated policy")
        diag.append("  FIX: needs more steps (300k+); consider wm_lr schedule or higher ensemble size")
    if actor_loss_positive and not rscale_capped and not policy_frozen:
        diag.append("DIAGNOSIS: actor_loss positive → critic overestimates → negative advantage")
        diag.append("  Seen with: short horizons (H≤10) or WM still learning (obs_loss > 0.5)")
        diag.append("  FIX: wait for critic to converge; ensure horizons ≥ 10; check WM quality")
    if grad_tiny and not rscale_capped:
        diag.append("DIAGNOSIS: actor_grad tiny but rscale OK → check actor_lr (should be 3e-5)")

    if len(eval_v) > 10 and not diag:
        late_eval = float(np.mean(_phase_vals(eval_v, "late")))
        early_eval = float(np.mean(_phase_vals(eval_v, "early")))
        if late_eval > 30:
            diag.append("DIAGNOSIS: ✓ LEARNING — eval return positive and growing")
        elif late_eval > early_eval + 10:
            diag.append(f"DIAGNOSIS: ✓ IMPROVING — eval rising from {early_eval:.0f} → {late_eval:.0f}")

    if spike_drop and not diag:
        diag.append("DIAGNOSIS: SPIKE_THEN_DROP — policy degraded; rollback fired")
        diag.append("  If eval has positive slope now: run is recovering — continue training")
        diag.append("  If eval still declining: check WM quality and actor stability")

    # EMA obs_loss stuck above threshold (adaptive-specific)
    _, ema_obs = arrays.get("wm/ema_obs_loss", (np.array([]), np.array([])))
    if len(ema_obs) > 20:
        ema_late = float(np.mean(ema_obs[-len(ema_obs)//3:]))
        if ema_late > 0.28:  # above Hopper high threshold (Walker2d threshold is 1.50)
            diag.append(f"DIAGNOSIS: EMA obs_loss={ema_late:.3f} still above thresh_high → H=10 selected throughout")
            diag.append("  This is correct if WM is genuinely struggling (e.g. policy exploring new states)")
            diag.append("  If obs_loss INCREASING: exploring policy → WM can't keep up (normal, wait for convergence)")
            diag.append("  If obs_loss STUCK: WM may need lower lr or larger replay")

    if not diag:
        diag.append("DIAGNOSIS: unclear — check WM and AC health sections above")

    return diag


def tb_deep_analysis(run_dir: Path, arrays: dict, flags: list[str],
                     last_n: int = 100) -> list[str]:
    """Return a list of text lines for the per-run deep section."""
    lines = []

    def get(tag):
        # Accept both imagine/* and imagination/* variants
        entry = arrays.get(tag)
        if entry is None:
            alt = tag.replace("imagine/", "imagination/").replace("imagination/", "imagine/")
            entry = arrays.get(alt)
        if entry is None:
            return np.array([]), np.array([])
        return entry

    def tl(tag):
        _, v = get(tag)
        return _tail(v, last_n)

    eval_steps, eval_ret = get("eval/return_mean")
    _, rscale  = get("value/return_scale")
    _, act_g   = get("grad/actor_norm")
    _, wm_loss = get("loss/world_model")
    hor_steps, hor_vals = get("imagine/horizon_used")
    _, unc_vals = get("imagine/uncertainty_mean")

    # ── Training phase ────────────────────────────────────────────────────────
    lines.append(_phase_detect(eval_steps, eval_ret))

    # ── Diagnosis (root cause + recommended fix) ──────────────────────────────
    for d in _diagnose_summary(arrays, flags):
        lines.append(d)

    # ── Trends ────────────────────────────────────────────────────────────────
    lines.append("--- Trends ---")
    if len(eval_ret) >= 10:
        lines.append(_trend(eval_ret, "eval_return"))
    if len(rscale) >= 10:
        lines.append(_trend(tl("value/return_scale"), "return_scale"))
    if len(act_g) >= 10:
        lines.append(_trend(tl("grad/actor_norm"), "actor_grad"))
    if len(wm_loss) >= 10:
        lines.append(_trend(tl("loss/world_model"), "wm_loss"))

    # ── World model health ────────────────────────────────────────────────────
    lines.append("--- WM Health ---")
    lines.extend(_wm_health(arrays))

    # ── Actor-Critic health ───────────────────────────────────────────────────
    lines.append("--- AC Health ---")
    lines.extend(_actor_critic_health(arrays))

    # ── EMA obs_loss + adaptive signal ───────────────────────────────────────
    ema_steps, ema_obs = get("wm/ema_obs_loss")
    if len(ema_obs) >= 3:
        n3 = max(1, len(ema_obs) // 3)
        ema_early = float(np.mean(ema_obs[:n3]))
        ema_late  = float(np.mean(ema_obs[-n3:]))
        ema_min   = float(np.min(ema_obs))
        ema_max   = float(np.max(ema_obs))
        lines.append("--- EMA obs_loss (adaptive signal) ---")
        lines.append(f"  early={ema_early:.4f}  late={ema_late:.4f}  "
                     f"min={ema_min:.4f}  max={ema_max:.4f}  "
                     f"trend={'↓' if ema_late < ema_early * 0.95 else ('↑' if ema_late > ema_early * 1.05 else '→')}")
        # Annotate which horizons would be selected at early/late obs_loss
        # (approximate — actual thresholds depend on env config)
        lines.append(f"  [Hopper thresholds: >0.28→H=10, >0.17→H=15, else H=20]")
        lines.append(f"  [Walker2d thresh:   >1.50→H=10, >1.00→H=15, else H=20]")
        h_early_hint = "H=10" if ema_early > 1.50 else ("H=10" if ema_early > 0.28 else ("H=15" if ema_early > 0.17 else "H=20"))
        h_late_hint  = "H=10" if ema_late  > 1.50 else ("H=10" if ema_late  > 0.28 else ("H=15" if ema_late  > 0.17 else "H=20"))
        lines.append(f"  signal implies: early→{h_early_hint}  late→{h_late_hint}")
        # Correlation: ema_obs_loss vs horizon_used (should be +ve for adaptive)
        if len(hor_vals) > 10:
            n = min(len(ema_obs), len(hor_vals))
            if n >= 10 and np.std(ema_obs[-n:]) > 1e-6 and np.std(hor_vals[-n:]) > 1e-6:
                r_corr = float(np.corrcoef(ema_obs[-n:], hor_vals[-n:])[0, 1])
                lines.append(f"  corr(ema_obs_loss, horizon_used) = {r_corr:+.2f}"
                             f"  {'← obs_loss drives horizon ✓' if r_corr < -0.2 else ('← adaptive signal WEAK' if abs(r_corr) < 0.1 else '')}")

    # ── Rollback + WM freeze events ───────────────────────────────────────────
    rb_steps_arr, rb_events = get("rollback/event")
    _, from_steps_arr = get("rollback/from_step")
    _, frozen_arr = get("wm/frozen")
    if len(rb_events) > 0 and rb_events.sum() > 0:
        lines.append("--- Rollback events ---")
        lines.append(f"  count={int(rb_events.sum())}")
        fired_at = rb_steps_arr[rb_events > 0] if len(rb_steps_arr) == len(rb_events) else rb_steps_arr[:min(5, len(rb_steps_arr))]
        if len(fired_at) > 0:
            fired_strs = ", ".join(f"{int(s)//1000}k" for s in fired_at[:5])
            lines.append(f"  fired at steps: {fired_strs}")
        wm_freeze_ops = int(frozen_arr.sum()) if len(frozen_arr) > 0 else 0
        lines.append(f"  wm_frozen_updates={wm_freeze_ops}"
                     + ("  (freeze NOT active — wm_freeze_after_rollback=0?)" if wm_freeze_ops == 0 and rb_events.sum() > 0 else ""))

    # ── Horizon distribution ──────────────────────────────────────────────────
    if len(hor_vals) > 0:
        lines.append(f"--- Horizon ---")
        lines.append(f"horizon_dist: {_horizon_dist(hor_vals)}")
        trend_str = _horizon_trend(hor_vals)
        if trend_str:
            lines.append(f"  {trend_str}")

    # ── Key metric percentiles (last N steps) ─────────────────────────────────
    lines.append(f"--- Percentiles (last {last_n} steps) ---")
    for tag, label in [
        ("value/return_scale",    "rscale"),
        ("grad/actor_norm",       "actor_grad"),
        ("policy/std_mean",       "policy_std"),
        ("imagine/uncertainty_mean", "uncertainty"),
        ("imagine/cont_prob_mean",   "cont_prob"),
        ("value/bootstrap",          "bootstrap_V"),
    ]:
        v = tl(tag)
        if len(v) == 0:
            continue
        lines.append(
            f"  {label:<14} p5={np.percentile(v,5):.3f}  "
            f"med={np.median(v):.3f}  p95={np.percentile(v,95):.3f}  "
            f"max={np.max(v):.3f}"
        )

    # Death-spiral correlation
    _, rs_full = get("value/return_scale")
    _, ag_full = get("grad/actor_norm")
    n = min(len(rs_full), len(ag_full))
    if n >= 20:
        r = float(np.corrcoef(rs_full[-n:], ag_full[-n:])[0, 1])
        lines.append(f"  corr(rscale, actor_grad) = {r:+.2f}"
                     f"  {'← death spiral' if r < -0.3 else ''}")

    # Eval early vs late
    if len(eval_ret) >= 6:
        e = np.mean(eval_ret[:len(eval_ret)//3])
        l = np.mean(eval_ret[2*len(eval_ret)//3:])
        lines.append(f"  eval early={e:.1f}  late={l:.1f}  Δ={l-e:+.1f}")

    return lines


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint-based deep analysis (diagnose.py style)
# ══════════════════════════════════════════════════════════════════════════════

def checkpoint_analysis(run_dir: Path, device: str = "cpu") -> list[str]:
    """Load checkpoint and run a forward pass; return compact text lines."""
    lines = []
    ckpt_dir = run_dir / "checkpoints"
    ckpt_path = ckpt_dir / "latest.pt"
    if not ckpt_path.exists():
        best = ckpt_dir / "best.pt"
        ckpt_path = best if best.exists() else None
    if ckpt_path is None:
        return ["  [SKIP] no checkpoint found"]

    cfg_path = run_dir / "config_snapshot.yaml"
    if not cfg_path.exists():
        return ["  [SKIP] no config_snapshot.yaml"]

    try:
        import torch
        import torch.nn.functional as F
        from omegaconf import OmegaConf
        from thesiswm.training.trainer import Trainer
        from thesiswm.training.world_model_updater import update_world_model
        from thesiswm.training.actor_critic_updater import update_actor_critic
        from thesiswm.training.imagination import lambda_returns

        cfg = OmegaConf.load(cfg_path)
        cfg.device = device
        cfg.training.resume = False
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "cfg" in ckpt:
            ckpt_cfg = OmegaConf.create(ckpt["cfg"])
            cfg.world_model.obs_dim = ckpt_cfg.world_model.obs_dim
            cfg.world_model.act_dim = ckpt_cfg.world_model.act_dim

        trainer = Trainer(cfg, build_env=False)
        trainer.checkpointer.load_into(trainer, ckpt)
        env_step = trainer.state.env_step
        lines.append(f"  ckpt_step={env_step:,}  updates={trainer.state.updates:,}  "
                     f"replay_size={len(trainer.replay):,}")

        if len(trainer.replay) < int(cfg.replay.seq_len) + 2:
            lines.append("  [SKIP] replay too small for forward pass")
            return lines

        dev = trainer.device
        wm  = trainer.world_model_ensemble.models[0]
        B   = int(cfg.imagination.rollout_batch)
        ctx = int(getattr(cfg.imagination, "context_len", cfg.replay.seq_len))
        H   = (int(cfg.imagination.horizon_fixed)
               if cfg.method.name.startswith("fixed_h")
               else int(cfg.imagination.horizons[-1]))

        seq = trainer.replay.sample_sequences(B, ctx)
        obs_seq = torch.as_tensor(seq["obs"],     dtype=torch.float32, device=dev)
        act_seq = torch.as_tensor(seq["actions"], dtype=torch.float32, device=dev)

        wm.eval()
        for p in wm.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            states_ctx, _, _ = wm.rssm.observe_sequence(obs_seq, act_seq, device=dev, sample=False)
            s = states_ctx[-1]
            feats, rews, logps, conts = [], [], [], []
            for _ in range(H):
                feat = wm.features(s)
                feats.append(feat)
                a, logp = trainer.agent.actor.sample(feat)
                logps.append(logp)
                s, _ = wm.rssm.imagine_step(s, a, sample=True)
                fn = wm.features(s)
                rews.append(wm.predict_reward(fn))
                conts.append(torch.sigmoid(wm.predict_continue_logit(fn)).clamp(0, 1))

            feat_last  = wm.features(s)
            feats_t    = torch.stack(feats,  dim=1)
            rewards_t  = torch.stack(rews,   dim=1)
            logps_t    = torch.stack(logps,  dim=1)
            conts_t    = torch.stack(conts,  dim=1)

            flat_feats  = feats_t.reshape(-1, feats_t.shape[-1])
            values_pred = trainer.agent.critic(flat_feats).reshape(B, H)
            v_last      = trainer.agent.critic(feat_last)
            values_ext  = torch.cat([values_pred, v_last.unsqueeze(1)], dim=1)
            discounts   = float(cfg.agent.discount) * conts_t
            target      = lambda_returns(rewards=rewards_t, values=values_ext,
                                         discounts=discounts, lambda_=float(cfg.agent.lambda_))
            advantage   = target - values_pred

            mean_t, log_std_t = trainer.agent.actor.forward(flat_feats)
            std_t   = torch.exp(log_std_t)
            gauss_e = (0.5 * (1.0 + math.log(2.0 * math.pi)) + log_std_t).sum(-1).mean()
            logp_e  = (-logps_t).mean()
            sat     = (torch.tanh(mean_t).abs() > 0.98).float().mean()

            def _s(name, x):
                x = x.detach().float()
                return (f"  {name:<28} mean={x.mean():+.3f}  std={x.std():.3f}  "
                        f"min={x.min():+.3f}  max={x.max():+.3f}"
                        + ("  NaN!" if torch.isnan(x).any() else ""))

            lines += [
                _s("imagined_reward",   rewards_t),
                _s("lambda_return",     target),
                _s("value_pred",        values_pred),
                _s("advantage",         advantage),
                _s("policy_std",        std_t),
                f"  gaussian_entropy={float(gauss_e):.3f}  logp_entropy={float(logp_e):.3f}"
                f"  saturated={float(sat):.1%}",
            ]

            # Verdict
            ok = []
            bad = []
            (ok if float(gauss_e) > 1.0       else bad).append("entropy_ok")
            (ok if float(logp_e) > -2.0        else bad).append("not_saturated")
            (ok if float(sat) < 0.10           else bad).append("saturation_ok")
            (ok if advantage.std() > 1.0       else bad).append("advantage_spread")
            lines.append(f"  VERDICT  OK={ok}  BAD={bad}")

        wm.train()
        for p in wm.parameters():
            p.requires_grad_(True)

    except Exception as exc:
        lines.append(f"  [ERROR] checkpoint analysis failed: {exc}")

    return lines


# ══════════════════════════════════════════════════════════════════════════════
# Report assembly
# ══════════════════════════════════════════════════════════════════════════════

def load_run(run_dir: Path, last_n: int = 100):
    """Return (summary_dict, arrays_dict) or (None, None)."""
    tb_dir = run_dir / "tb"
    if not tb_dir.exists():
        return None, None

    if _cache_valid(run_dir, tb_dir):
        arrays = _load_cache_arrays(run_dir)
        if arrays is not None:
            return _compute_stats(run_dir, arrays, last_n), arrays

    arrays = _stream_scalars(tb_dir)
    if arrays is None:
        return None, None
    _save_cache(run_dir, tb_dir, arrays)
    return _compute_stats(run_dir, arrays, last_n), arrays


def build_report(runs_dir: Path, filter_str: Optional[str],
                 last_n: int, deep_n: int, device: str, workers: int = 0) -> str:
    run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir())
    if filter_str:
        run_dirs = [d for d in run_dirs if filter_str.lower() in d.name.lower()]

    # ── Phase 1: parallel TB scan ──────────────────────────────────────────────
    # TB parsing is I/O-bound → threads; use all logical CPUs if not specified.
    n_workers = workers if workers > 0 else min(len(run_dirs), os.cpu_count() or 8)
    summaries = {}   # name → summary dict
    arrays_map = {}  # name → arrays dict
    skipped = []

    pbar = _tqdm(total=len(run_dirs), desc="Scanning runs", unit="run",
                 dynamic_ncols=True, leave=True)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(load_run, rd, last_n): rd for rd in run_dirs}
        for fut in as_completed(futs):
            rd = futs[fut]
            s, a = fut.result()
            if s is None:
                skipped.append(rd.name)
            else:
                summaries[rd.name] = s
                arrays_map[rd.name] = a
            cached = "(cached)" if (rd / ".scan_cache.json").exists() else "(parsed)"
            pbar.set_postfix_str(f"{rd.name[:30]} {cached}", refresh=False)
            pbar.update(1)
    pbar.close()

    lines = []
    W = "=" * 70

    lines.append(W)
    lines.append(f"  ThesisWM Analysis Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  runs_dir: {runs_dir}  |  total={len(run_dirs)}  loaded={len(summaries)}  "
                 f"skipped={len(skipped)}")
    lines.append(W)

    # ── summary table ──────────────────────────────────────────────────────────
    results = sorted(summaries.values(),
                     key=lambda r: -(r["final_eval"] or -9999))

    COL = 24
    lines.append(
        f"\n{'RUN':<{COL}} {'STEPS':>8} {'F_EVAL':>7} {'MAX_E':>7} "
        f"{'CONT':>6} {'H_AVG':>6} {'UNC':>6} {'STD':>5} "
        f"{'AGRAD':>6} {'RSCALE':>8}  FLAGS"
    )
    lines.append("─" * 150)
    for r in results:
        env, mth, seed = r["env"], r["method"], r["seed"]
        label = f"{env[:7]}/{mth[:10]}/s{seed if seed is not None else '?'}"
        spike = "▲▼" if r["spike"] else "  "
        flags = "  ".join(r["problems"]) if r["problems"] else "OK"
        lines.append(
            f"{label:<{COL}} "
            f"{r['max_step']:>8,} "
            f"{_fmt(r['final_eval'],1):>7} "
            f"{spike}{_fmt(r['max_eval'],1):>5} "
            f"{_fmt(r['mean_cont'],2):>6} "
            f"{_fmt(r['mean_horizon'],1):>6} "
            f"{_fmt(r['mean_unc'],2):>6} "
            f"{_fmt(r['mean_std'],2):>5} "
            f"{_fmt(r['actor_grad'],4):>6} "
            f"{_fmt(r['max_rscale'],0):>8}  "
            f"{flags}"
        )

    # ── per-method aggregates ──────────────────────────────────────────────────
    # Deduplicate by (env, method, seed) — keep highest-step run for each slot.
    # This prevents double-counting when both old-named (adaptive_5_10_20) and
    # new-named (adaptive_10_15_20) runs exist in the same directory for the same seed.
    from collections import defaultdict
    seen_slot: dict = {}   # (env, method, seed) → max_step seen so far
    deduped: list = []
    for r in results:
        slot = (r["env"], r["method"], r["seed"])
        if slot not in seen_slot or r["max_step"] > seen_slot[slot]:
            seen_slot[slot] = r["max_step"]
            deduped = [x for x in deduped
                       if (x["env"], x["method"], x["seed"]) != slot]
            deduped.append(r)

    by_method: dict = defaultdict(list)
    for r in deduped:
        by_method[f"{r['env']}/{r['method']}"].append(r)

    lines.append(f"\n{'─'*70}")
    lines.append("  METHOD AGGREGATES  (deduplicated — one entry per env/method/seed)")
    lines.append("─" * 70)
    for key in sorted(by_method):
        runs = by_method[key]
        evals  = [r["final_eval"] for r in runs if r["final_eval"]  is not None]
        spikes = sum(1 for r in runs if r["spike"])
        flags  = sum(1 for r in runs if r["problems"])
        e_mu   = f"{np.mean(evals):.1f}±{np.std(evals):.1f}" if evals else "N/A"
        lines.append(f"  {key:<32}  n={len(runs)}  eval={e_mu}"
                     + (f"  ⚡{spikes}spike" if spikes else "")
                     + (f"  ⚠{flags}flagged" if flags else "  ✓clean"))

    # ── Phase 2: deep TB analysis for flagged runs ─────────────────────────────
    flagged = [r for r in results if r["problems"]]
    if flagged:
        lines.append(f"\n{'─'*70}")
        lines.append(f"  DEEP TB ANALYSIS — {len(flagged)} flagged run(s)")
        lines.append("─" * 70)
        for r in flagged:
            lines.append(f"\n  ── {r['run_dir']} ──")
            lines.append(f"  FLAGS: {' | '.join(r['problems'])}")
            arr = arrays_map.get(r["run_dir"], {})
            for line in tb_deep_analysis(Path(runs_dir / r["run_dir"]), arr, r["problems"], last_n):
                lines.append(f"  {line}")

    # ── Phase 3: checkpoint analysis for worst N ───────────────────────────────
    if deep_n > 0:
        worst = sorted(
            [r for r in results if r["problems"]],
            key=lambda r: (r["final_eval"] or 0)
        )[:deep_n]
        lines.append(f"\n{'─'*70}")
        lines.append(f"  CHECKPOINT ANALYSIS — worst {len(worst)} run(s)")
        lines.append("─" * 70)
        for r in _tqdm(worst, desc="Loading checkpoints", unit="ckpt",
                       dynamic_ncols=True, leave=True):
            lines.append(f"\n  ── {r['run_dir']} ──")
            rd = runs_dir / r["run_dir"]
            for line in checkpoint_analysis(rd, device=device):
                lines.append(line)

    lines.append(f"\n{W}")
    lines.append("  END OF REPORT")
    lines.append(W)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_dir", default="runs",    help="Root runs directory")
    p.add_argument("--filter",   default=None,      help="Only process runs matching this string")
    p.add_argument("--last_n",   type=int, default=100,
                   help="Last N TB points used for statistics")
    p.add_argument("--deep",     type=int, default=0,
                   help="Load checkpoints for the N worst flagged runs (slow, needs GPU)")
    p.add_argument("--device",   default="cpu",
                   help="Device for --deep checkpoint analysis")
    p.add_argument("--workers",  type=int, default=0,
                   help="TB-scan threads (default: all logical CPUs)")
    p.add_argument("--out",      default=None,
                   help="Output file prefix (default: report_<timestamp>)")
    p.add_argument("--no_save",  action="store_true",
                   help="Print to stdout only, do not save file")
    args = p.parse_args()

    runs_dir = ROOT / args.runs_dir
    if not runs_dir.exists():
        print(f"[ERROR] runs_dir not found: {runs_dir}")
        sys.exit(1)

    print(f"Building report for {runs_dir} …")
    report = build_report(
        runs_dir=runs_dir,
        workers=args.workers,
        filter_str=args.filter,
        last_n=args.last_n,
        deep_n=args.deep,
        device=args.device,
    )

    print(report)

    if not args.no_save:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = args.out or f"report_{ts}"
        out_path = ROOT / f"{stem}.txt"
        out_path.write_text(report, encoding="utf-8")
        print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()
