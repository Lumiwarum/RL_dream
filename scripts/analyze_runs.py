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


def _corr(a: np.ndarray, b: np.ndarray, label: str) -> str:
    n = min(len(a), len(b))
    if n < 5:
        return ""
    a2, b2 = a[-n:], b[-n:]
    if a2.std() < 1e-8 or b2.std() < 1e-8:
        return ""
    r = float(np.corrcoef(a2, b2)[0, 1])
    return f"  corr({label})={r:+.2f}"


def tb_deep_analysis(run_dir: Path, arrays: dict, last_n: int = 100) -> list[str]:
    """Return a list of text lines for the per-run deep section."""
    lines = []

    def get(tag):
        return arrays.get(tag, (np.array([]), np.array([])))

    def tl(tag):
        _, v = get(tag)
        return _tail(v, last_n)

    eval_steps, eval_ret = get("eval/return_mean")
    _, rscale  = get("value/return_scale")
    _, act_g   = get("grad/actor_norm")
    _, wm_loss = get("loss/world_model")
    hor_steps, hor_vals = get("imagine/horizon_used")
    _, unc_vals = get("imagine/uncertainty_mean")
    _, std_vals = get("policy/std_mean")
    _, cont_vals = get("imagine/cont_prob_mean")

    # Phase
    lines.append(_phase_detect(eval_steps, eval_ret))

    # Trends
    if len(eval_ret) >= 10:
        lines.append(_trend(eval_ret, "eval_return"))
    if len(rscale) >= 10:
        lines.append(_trend(tl("value/return_scale"), "return_scale"))
    if len(act_g) >= 10:
        lines.append(_trend(tl("grad/actor_norm"), "actor_grad"))

    # Horizon distribution
    lines.append(f"horizon_dist: {_horizon_dist(tl('imagine/horizon_used'))}")

    # Percentiles of key metrics (last_n points)
    for tag, label in [
        ("value/return_scale", "rscale"),
        ("grad/actor_norm",    "actor_grad"),
        ("policy/std_mean",    "policy_std"),
        ("imagine/uncertainty_mean", "uncertainty"),
        ("imagine/cont_prob_mean",   "cont_prob"),
    ]:
        v = tl(tag)
        if len(v) == 0:
            continue
        lines.append(
            f"  {label:<18} p5={np.percentile(v,5):.3f}  "
            f"med={np.median(v):.3f}  p95={np.percentile(v,95):.3f}  "
            f"max={np.max(v):.3f}"
        )

    # Correlations: does high rscale predict low actor_grad? (the death spiral)
    n = min(len(rscale), len(act_g))
    if n >= 20:
        r = float(np.corrcoef(rscale[-n:], act_g[-n:])[0, 1])
        lines.append(f"  corr(rscale, actor_grad) = {r:+.2f}"
                     f"  ← negative confirms ACTOR_GRAD_TINY death spiral")

    # Early vs late eval split
    if len(eval_ret) >= 6:
        early = np.mean(eval_ret[:len(eval_ret)//3])
        late  = np.mean(eval_ret[2*len(eval_ret)//3:])
        lines.append(f"  eval early_avg={early:.1f}  late_avg={late:.1f}  "
                     f"Δ={late-early:+.1f}")

    # WM loss trend
    if len(wm_loss) >= 10:
        lines.append(_trend(tl("loss/world_model"), "wm_loss"))

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
    from collections import defaultdict
    by_method: dict = defaultdict(list)
    for r in results:
        by_method[f"{r['env']}/{r['method']}"].append(r)

    lines.append(f"\n{'─'*70}")
    lines.append("  METHOD AGGREGATES")
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
            for line in tb_deep_analysis(Path(runs_dir / r["run_dir"]), arr, last_n):
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
