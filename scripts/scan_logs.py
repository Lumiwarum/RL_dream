"""TensorBoard log scanner.

Usage:
    python scripts/scan_logs.py
    python scripts/scan_logs.py --filter hopper --sort eval/return_mean
    python scripts/scan_logs.py --last_n 50 --csv out.csv
    python scripts/scan_logs.py --filter adaptive --sort eval/return_mean --last_n 100
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"

WANTED_TAGS = [
    "eval/return_mean",
    "eval/return_std",
    "loss/obs",
    "loss/obs_prior",
    "loss/kl",
    "loss/reward",
    "wm/ema_obs_loss",
    "policy/std_mean",
    "actor/grad_norm",
    "imagine/horizon",
    "imagine/uncertainty",
    "value/return_scale",
    # r2dreamer-mapped names
    "train/loss/obs",
    "train/loss/rep",
    "train/loss/dyn",
    "train/loss/rew",
    "train/loss/policy",
    "train/wm/prior_pred_loss",
    "train/wm/ema_obs_loss",
    "train/imagine/horizon",
    "train/action_entropy",
    "train/ret",
    "train/val",
]

# Diagnostic flag thresholds
SPIKE_RATIO        = 2.0   # max_eval > SPIKE_RATIO * final_eval
WM_NOT_LEARNING    = 0.9   # obs_loss_late > WM_NOT_LEARNING * obs_loss_early
POSTERIOR_COLLAPSE = 0.1   # kl_late < this


def load_tb(tb_dir: Path, last_n: int, cache_file: Path, use_cache: bool = True):
    """Load TensorBoard events; use JSON cache if available and fresh."""
    try:
        if use_cache and cache_file.exists():
            cache_mtime = cache_file.stat().st_mtime
            # Check if any event file is newer than cache
            event_files = list(tb_dir.glob("events.out.tfevents.*"))
            if event_files and max(f.stat().st_mtime for f in event_files) <= cache_mtime:
                with open(cache_file) as f:
                    return json.load(f)
    except Exception:
        pass

    data = {}
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(str(tb_dir))
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            data[tag] = [(e.step, e.value) for e in events]
    except Exception as exc:
        metrics_file = tb_dir / "metrics.jsonl"
        if not metrics_file.exists():
            print(f"  Warning: could not load {tb_dir}: {exc}")
            return {}
        try:
            with open(metrics_file) as f:
                for line in f:
                    row = json.loads(line)
                    step = row.get("step")
                    if step is None:
                        continue
                    for tag, value in row.items():
                        if tag == "step":
                            continue
                        if isinstance(value, (int, float)):
                            data.setdefault(tag, []).append((step, value))
        except Exception as json_exc:
            print(f"  Warning: could not load {tb_dir}: {exc}; JSONL fallback failed: {json_exc}")
            return {}

    try:
        with open(cache_file, "w") as f:
            json.dump(data, f)
    except Exception:
        pass
    return data


def tail(series, n):
    return [v for _, v in series[-n:]] if series else []


def head_frac(series, frac=0.33):
    n = max(1, int(len(series) * frac))
    return [v for _, v in series[:n]]


def tail_frac(series, frac=0.33):
    n = max(1, int(len(series) * frac))
    return [v for _, v in series[-n:]]


def safe_mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def safe_max(vals):
    return max(vals) if vals else float("nan")


def fmt(v, decimals=2):
    if v != v:  # nan
        return "  --  "
    return f"{v:.{decimals}f}"


def analyse_run(run_dir: Path, last_n: int, use_cache: bool = True):
    tb_dir = run_dir / "tb"
    if not tb_dir.exists():
        return None

    cache_file = tb_dir / ".scan_cache.json"
    data = load_tb(tb_dir, last_n, cache_file, use_cache=use_cache)
    if not data:
        return None

    # Helper: get tag from possible aliases
    def get(tag, *aliases):
        for t in [tag] + list(aliases):
            if t in data:
                return data[t]
            # try with train/ prefix
            train_t = f"train/{t}"
            if train_t in data:
                return data[train_t]
        return []

    eval_series   = get("eval/return_mean")
    kl_series     = get("loss/rep", "loss/dyn")
    obs_series    = get("loss/obs")
    prior_series  = get("wm/prior_pred_loss", "loss/obs_prior")
    ema_series    = get("wm/ema_obs_loss")
    horizon_series = get("imagine/horizon")
    entropy_series = get("action_entropy", "policy/std_mean")

    total_steps = max((s for s, _ in eval_series), default=0) if eval_series else 0

    eval_vals   = tail(eval_series, last_n)
    final_eval  = safe_mean(tail(eval_series, 5))
    max_eval    = safe_max([v for _, v in eval_series] if eval_series else [])
    early_eval  = safe_mean(head_frac(eval_series))
    late_eval   = safe_mean(tail_frac(eval_series))

    obs_early   = safe_mean(head_frac(obs_series))
    obs_late    = safe_mean(tail_frac(obs_series))
    kl_late     = safe_mean(tail_frac(kl_series))
    prior_late  = safe_mean(tail_frac(prior_series))
    ema_late    = safe_mean(tail(ema_series, 10))
    h_mean      = safe_mean([v for _, v in horizon_series]) if horizon_series else float("nan")
    h_std       = (
        (sum((v - h_mean)**2 for _, v in horizon_series) / len(horizon_series)) ** 0.5
        if len(horizon_series) > 1 else 0.0
    )
    entropy_late = safe_mean(tail(entropy_series, last_n))

    # Diagnostic flags
    flags = []
    if final_eval == final_eval and max_eval == max_eval:
        if max_eval > SPIKE_RATIO * max(final_eval, 1e-6):
            flags.append("SPIKE")
    if obs_early == obs_early and obs_late == obs_late and obs_early > 0:
        if obs_late > WM_NOT_LEARNING * obs_early:
            flags.append("WM_STALL")
    if kl_late == kl_late and kl_late < POSTERIOR_COLLAPSE:
        flags.append("POST_COLL")
    if h_std < 0.5 and len(horizon_series) > 10:
        flags.append("H_STUCK")

    return {
        "run": run_dir.name,
        "steps": total_steps,
        "eval_final": final_eval,
        "eval_max": max_eval,
        "eval_std": (
            (sum((v - final_eval)**2 for v in eval_vals) / max(len(eval_vals) - 1, 1))**0.5
            if len(eval_vals) > 1 else 0.0
        ),
        "obs_loss": obs_late,
        "prior_loss": prior_late,
        "kl": kl_late,
        "ema_obs": ema_late,
        "h_mean": h_mean,
        "entropy": entropy_late,
        "flags": ",".join(flags) if flags else "OK",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=str(RUNS_DIR))
    ap.add_argument("--filter", default="", help="substring filter on run name")
    ap.add_argument("--sort", default="eval/return_mean", help="column to sort by")
    ap.add_argument("--last_n", type=int, default=100, help="last N points per tag")
    ap.add_argument("--csv", default="", help="write CSV to this path")
    ap.add_argument("--no_cache", action="store_true")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"No runs directory at {runs_dir}")
        return

    run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
    if args.filter:
        run_dirs = [d for d in run_dirs if args.filter.lower() in d.name.lower()]

    results = []
    for d in run_dirs:
        r = analyse_run(d, args.last_n, use_cache=not args.no_cache)
        if r:
            results.append(r)

    if not results:
        print("No runs found.")
        return

    # Sort
    sort_map = {
        "eval/return_mean": "eval_final",
        "eval_final": "eval_final",
        "steps": "steps",
        "eval_max": "eval_max",
    }
    sort_key = sort_map.get(args.sort, "eval_final")
    results.sort(key=lambda r: -(r[sort_key] if r[sort_key] == r[sort_key] else -1e9))

    # Print table
    header = f"{'RUN':<55} {'STEPS':>6} {'EVAL':>7} {'MAX_E':>7} {'STD':>5} {'OBS_L':>6} {'PRIOR':>6} {'KL':>5} {'EMA':>6} {'H_AVG':>6} {'ENT':>6}  FLAGS"
    print(header)
    print("-" * len(header))
    for r in results:
        steps_k = f"{r['steps']//1000}k"
        print(
            f"{r['run']:<55} {steps_k:>6} {fmt(r['eval_final']):>7} {fmt(r['eval_max']):>7}"
            f" {fmt(r['eval_std']):>5} {fmt(r['obs_loss']):>6} {fmt(r['prior_loss']):>6}"
            f" {fmt(r['kl']):>5} {fmt(r['ema_obs']):>6} {fmt(r['h_mean']):>6}"
            f" {fmt(r['entropy'], 3):>6}  {r['flags']}"
        )

    print(f"\n{len(results)} runs shown.")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
