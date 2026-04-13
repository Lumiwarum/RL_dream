"""
scan_logs.py — Scan all TensorBoard logs and print a diagnostic summary table.

Reads every run under runs/<exp_name>/tb/ and extracts key metrics to identify:
  - Overall learning progress (eval + train return)
  - Spike-then-drop (peak achieved early then regressed)
  - Adaptive horizon behaviour (stuck at H_min = adaptive not working)
  - Actor / WM health
  - Continue probability (should be >0.9 after cont_target fix)
  - Rollback events

Usage:
    python scripts/scan_logs.py                         # all runs
    python scripts/scan_logs.py --filter hopper         # subset
    python scripts/scan_logs.py --sort max_eval         # sort column
    python scripts/scan_logs.py --last_n 50             # use last N points for stats
    python scripts/scan_logs.py --csv out.csv           # also write CSV
    python scripts/scan_logs.py --curve                 # print learning curves
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kwargs):  # type: ignore[misc]
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", "?")
        print(f"{desc} ({total} items) …", flush=True)
        return it

ROOT = Path(__file__).resolve().parent.parent

# Cont thresholds: cont_disc_floor=0.9 clamps the imagination discount regardless of
# what the WM predicts, so low cont_prob no longer collapses learning.
# These thresholds only flag genuine cont head failure (stuck near 0).
CONT_THRESHOLDS = {
    "fixed_h5": 0.50, "fixed_h10": 0.45, "fixed_h15": 0.40,
    "fixed_h20": 0.35, "adaptive": 0.45,
}
sys.path.insert(0, str(ROOT))

try:
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
except ImportError:
    print("[ERROR] tensorboard not installed.  pip install tensorboard")
    sys.exit(1)

# Tags we care about — anything else is skipped without allocation.
_WANTED_TAGS = frozenset([
    "eval/return_mean", "eval/best_return",
    "train/return_mean_20", "train/episode_return",
    "eval/rollback", "train/rollback",
    "imagine/horizon_used", "imagine/uncertainty_mean", "imagine/cont_prob_mean",
    "policy/std_mean", "policy/entropy_gaussian",
    "grad/actor_norm", "loss/world_model",
    "value/return_scale",
])
_MAX_PER_TAG = 2000   # deque maxlen — keeps the newest N points


# ── helpers ────────────────────────────────────────────────────────────────────

def _load(arrays: dict, tag: str):
    """Return (steps, values) arrays from pre-loaded dict."""
    entry = arrays.get(tag)
    if entry is None:
        return np.array([]), np.array([])
    return entry


def _tail(vals: np.ndarray, n: int) -> np.ndarray:
    return vals[-n:] if len(vals) >= n else vals


def _head(vals: np.ndarray, n: int) -> np.ndarray:
    return vals[:n] if len(vals) >= n else vals


def _fmt(v, dec=1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if isinstance(v, float) and (np.isinf(v) or abs(v) >= 1e6):
        return f"{v:.2e}"
    if isinstance(v, float) and abs(v) >= 1000:
        return f"{v:.0f}"
    return f"{v:.{dec}f}"


def _parse_name(name: str):
    import re
    parts = name.split("_")
    seed = None
    # New format: _s0 / _s1 / _s2  (short seed tag from run_experiments.py)
    m = re.search(r"_s(\d+)$", name)
    if m:
        seed = int(m.group(1))
    else:
        # Legacy format: _seed0 / _seed1
        for p in reversed(parts):
            if p.startswith("seed"):
                try:
                    seed = int(p[4:])
                except ValueError:
                    pass
                break
    if "adaptive" in name:
        method = "adaptive"
    elif "fixed_h5" in name:
        method = "fixed_h5"
    elif "fixed_h10" in name:
        method = "fixed_h10"
    elif "fixed_h15" in name:
        method = "fixed_h15"
    elif "fixed_h20" in name:
        method = "fixed_h20"
    else:
        method = "?"
    if "walker" in name.lower():
        env = "Walker2d"
    elif "hopper" in name.lower():
        env = "Hopper"
    elif "pendulum" in name.lower():
        env = "Pendulum"
    else:
        env = "?"
    return env, method, seed


# ── per-run summary ────────────────────────────────────────────────────────────

_CACHE_VERSION = 4   # bumped: switched from EventAccumulator to EventFileLoader


def _stream_scalars(tb_dir: Path) -> Optional[dict]:
    """
    Stream scalar events directly from TFRecord event files.

    Uses EventFileLoader (lower-level than EventAccumulator) + per-tag deques
    so only _MAX_PER_TAG newest points per tag are ever held in memory,
    and every non-scalar event is skipped immediately without allocation.

    Returns {tag: (steps_array, vals_array)} or None if no event files found.
    """
    from collections import deque

    event_files = sorted(tb_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None

    steps_dq: dict = {t: deque(maxlen=_MAX_PER_TAG) for t in _WANTED_TAGS}
    vals_dq:  dict = {t: deque(maxlen=_MAX_PER_TAG) for t in _WANTED_TAGS}

    for ef in event_files:
        try:
            loader = EventFileLoader(str(ef))
            for event in loader.Load():
                if not event.HasField("summary"):
                    continue
                step = event.step
                for val in event.summary.value:
                    tag = val.tag
                    if tag not in _WANTED_TAGS:
                        continue
                    if val.HasField("simple_value"):
                        steps_dq[tag].append(step)
                        vals_dq[tag].append(val.simple_value)
                    # tensor scalars (newer TF summary format)
                    elif val.HasField("tensor"):
                        try:
                            import struct as _struct
                            raw = val.tensor.tensor_content
                            if raw:
                                v = _struct.unpack("<f", raw[:4])[0]
                            else:
                                v = float(val.tensor.float_val[0])
                            steps_dq[tag].append(step)
                            vals_dq[tag].append(v)
                        except Exception:
                            pass
        except Exception:
            continue   # truncated / corrupt file — skip

    return {
        tag: (
            np.array(list(steps_dq[tag]), dtype=np.float64),
            np.array(list(vals_dq[tag]),  dtype=np.float64),
        )
        for tag in _WANTED_TAGS
    }


def _cache_path(run_dir: Path) -> Path:
    return run_dir / ".scan_cache.json"


def _cache_valid(run_dir: Path, tb_dir: Path) -> bool:
    cp = _cache_path(run_dir)
    if not cp.exists():
        return False
    try:
        with open(cp) as f:
            meta = json.load(f)
        if meta.get("version") != _CACHE_VERSION:
            return False
        cached_mtime = meta.get("tb_mtime", 0)
        newest_event = max(
            p.stat().st_mtime for p in tb_dir.glob("events.out.tfevents.*")
        )
        return newest_event <= cached_mtime
    except Exception:
        return False


def _load_cache(run_dir: Path) -> Optional[dict]:
    try:
        with open(_cache_path(run_dir)) as f:
            d = json.load(f)
        return d.get("series")
    except Exception:
        return None


def _arrays_to_json(arrays: dict) -> dict:
    """Convert {tag: (steps_arr, vals_arr)} → JSON-serialisable dict."""
    return {tag: {"steps": s.tolist(), "vals": v.tolist()}
            for tag, (s, v) in arrays.items()}


def _json_to_arrays(series: dict) -> dict:
    """Convert JSON-loaded dict → {tag: (steps_arr, vals_arr)}."""
    return {tag: (np.array(d["steps"], dtype=np.float64),
                  np.array(d["vals"],  dtype=np.float64))
            for tag, d in series.items()}


def _save_cache(run_dir: Path, tb_dir: Path, arrays: dict) -> None:
    newest_event = max(
        p.stat().st_mtime for p in tb_dir.glob("events.out.tfevents.*")
    )
    payload = {"version": _CACHE_VERSION, "tb_mtime": newest_event,
               "series": _arrays_to_json(arrays)}
    try:
        with open(_cache_path(run_dir), "w") as f:
            json.dump(payload, f)
    except Exception:
        pass


def _load_cache_arrays(run_dir: Path) -> Optional[dict]:
    try:
        with open(_cache_path(run_dir)) as f:
            d = json.load(f)
        return _json_to_arrays(d["series"])
    except Exception:
        return None


def summarize_run(run_dir: Path, last_n: int) -> Optional[dict]:
    tb_dir = run_dir / "tb"
    if not tb_dir.exists():
        return None

    # ── cache hit (instant) ────────────────────────────────────────────────────
    if _cache_valid(run_dir, tb_dir):
        arrays = _load_cache_arrays(run_dir)
        if arrays is not None:
            return _compute_stats(run_dir, arrays, last_n)

    # ── cold path: stream directly from TFRecord event files ──────────────────
    arrays = _stream_scalars(tb_dir)
    if arrays is None:
        return None
    _save_cache(run_dir, tb_dir, arrays)
    return _compute_stats(run_dir, arrays, last_n)


def _compute_stats(run_dir: Path, arrays: dict, last_n: int) -> dict:
    """Compute all summary statistics from pre-loaded numpy arrays."""

    def get(tag):
        return arrays.get(tag, (np.array([]), np.array([])))

    def tl(tag):
        _, v = get(tag)
        return _tail(v, last_n)

    def last_s(tag):
        s, _ = get(tag)
        return int(s[-1]) if len(s) else 0

    # ── eval ──────────────────────────────────────────────────────────────────
    eval_steps, eval_ret = get("eval/return_mean")

    final_eval  = float(np.mean(_tail(eval_ret, 5)))  if len(eval_ret) >= 1 else None
    max_eval    = float(np.max(eval_ret))              if len(eval_ret) >= 1 else None
    early_eval  = float(np.mean(_head(eval_ret, 5)))  if len(eval_ret) >= 1 else None

    max_step_at = None
    if len(eval_ret) >= 1:
        max_idx = int(np.argmax(eval_ret))
        max_step_at = int(eval_steps[max_idx]) if len(eval_steps) > max_idx else None

    spike = False
    if max_eval is not None and final_eval is not None:
        drop     = max_eval - final_eval
        max_step = max(last_s("eval/return_mean"), 1)
        spike    = (drop > 30) and (max_step_at is not None) and (max_step_at < max_step * 0.5)

    plateau = float(np.std(_tail(eval_ret, 10))) if len(eval_ret) >= 10 else None

    # ── train return ──────────────────────────────────────────────────────────
    _, train_ret20 = get("train/return_mean_20")
    final_train    = float(np.mean(_tail(train_ret20, 20))) if len(train_ret20) >= 1 else None
    peak_train     = float(np.max(train_ret20))             if len(train_ret20) >= 1 else None

    # ── rollbacks ─────────────────────────────────────────────────────────────
    _, rb_eval  = get("eval/rollback")
    _, rb_train = get("train/rollback")
    n_rollbacks = int(rb_eval.sum() + rb_train.sum()) if (len(rb_eval) + len(rb_train)) > 0 else 0

    # ── imagination / horizon ─────────────────────────────────────────────────
    hor_vals  = tl("imagine/horizon_used")
    unc_vals  = tl("imagine/uncertainty_mean")
    cont_vals = tl("imagine/cont_prob_mean")
    mean_hor  = float(np.mean(hor_vals))  if len(hor_vals) else None
    mean_unc  = float(np.mean(unc_vals))  if len(unc_vals) else None
    mean_cont = float(np.mean(cont_vals)) if len(cont_vals) else None
    h_min_pct = float(np.mean(hor_vals == hor_vals.min())) * 100 if len(hor_vals) else None

    # ── policy ────────────────────────────────────────────────────────────────
    std_vals  = tl("policy/std_mean")
    ent_vals  = tl("policy/entropy_gaussian")
    mean_std  = float(np.mean(std_vals)) if len(std_vals) else None
    mean_ent  = float(np.mean(ent_vals)) if len(ent_vals) else None

    # ── gradients & losses ────────────────────────────────────────────────────
    act_grads  = tl("grad/actor_norm")
    wm_vals    = tl("loss/world_model")
    mean_act_g = float(np.mean(act_grads)) if len(act_grads) else None
    mean_wm    = float(np.mean(wm_vals))   if len(wm_vals)   else None

    _, rscale_all = get("value/return_scale")
    mean_rscale   = float(np.mean(_tail(rscale_all, last_n))) if len(rscale_all) else None
    max_rscale    = float(np.max(rscale_all))                 if len(rscale_all) else None

    max_step = max(last_s("eval/return_mean"),
                   last_s("imagine/horizon_used"),
                   last_s("loss/world_model"))

    env, method, seed = _parse_name(run_dir.name)

    # ── problem flags ─────────────────────────────────────────────────────────
    problems = []
    if h_min_pct is not None and h_min_pct > 90 and method == "adaptive":
        problems.append(f"HORIZON_STUCK@{h_min_pct:.0f}%_min")
    if mean_act_g is not None and mean_act_g < 0.02:
        problems.append(f"ACTOR_GRAD_TINY({mean_act_g:.4f})")
    if mean_std is not None and mean_std < 0.40:
        problems.append(f"STD_COLLAPSED({mean_std:.3f})")
    if mean_std is not None and mean_std > 1.30:
        problems.append(f"STD_HIGH({mean_std:.3f})")
    if plateau is not None and plateau < 5.0 and final_eval is not None and final_eval < 200:
        problems.append(f"PLATEAU(σ={plateau:.1f})")
    if mean_unc is not None and mean_unc > 2.0:
        problems.append(f"UNC_SCALE({mean_unc:.2f})")
    if spike:
        problems.append(f"SPIKE_THEN_DROP(peak={max_eval:.0f}@{max_step_at//1000}k→now={final_eval:.0f})")
    if max_rscale is not None and max_rscale > 200.0:
        problems.append(f"VALUE_DIVERGE(rscale_max={max_rscale:.0f})")
    _cont_thresh = CONT_THRESHOLDS.get(method, 0.70)
    if mean_cont is not None and mean_cont < _cont_thresh:
        problems.append(f"CONT_LOW({mean_cont:.2f}<{_cont_thresh})")
    if n_rollbacks > 0:
        problems.append(f"ROLLBACKS={n_rollbacks}")

    return {
        "run_dir":      run_dir.name,
        "env":          env,
        "method":       method,
        "seed":         seed,
        "max_step":     max_step,
        "final_eval":   final_eval,
        "max_eval":     max_eval,
        "early_eval":   early_eval,
        "spike":        spike,
        "final_train":  final_train,
        "peak_train":   peak_train,
        "plateau_std":  plateau,
        "mean_horizon": mean_hor,
        "h_min_pct":    h_min_pct,
        "mean_unc":     mean_unc,
        "mean_cont":    mean_cont,
        "mean_std":     mean_std,
        "mean_ent":     mean_ent,
        "actor_grad":   mean_act_g,
        "return_scale": mean_rscale,
        "max_rscale":   max_rscale,
        "wm_loss":      mean_wm,
        "n_rollbacks":  n_rollbacks,
        "problems":     problems,
        "_eval_steps":  eval_steps,
        "_eval_ret":    eval_ret,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--filter",   default=None)
    p.add_argument("--env",      default=None)
    p.add_argument("--method",   default=None)
    p.add_argument("--sort",     default="final_eval",
                   choices=["final_eval", "max_eval", "max_step", "mean_horizon",
                            "actor_grad", "mean_unc", "run_dir", "final_train"])
    p.add_argument("--last_n",   type=int, default=50)
    p.add_argument("--csv",      default=None)
    p.add_argument("--problems_only", action="store_true")
    p.add_argument("--curve",    action="store_true",
                   help="Print ASCII learning curve (eval return over time) for each run")
    args = p.parse_args()

    runs_dir = ROOT / args.runs_dir
    if not runs_dir.exists():
        print(f"[ERROR] runs_dir not found: {runs_dir}")
        sys.exit(1)

    run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir())
    if args.filter:
        run_dirs = [d for d in run_dirs if args.filter.lower() in d.name.lower()]
    if args.env:
        run_dirs = [d for d in run_dirs if args.env.lower() in d.name.lower()]
    if args.method:
        run_dirs = [d for d in run_dirs if args.method.lower() in d.name.lower()]

    n_workers = min(len(run_dirs), os.cpu_count() or 8)
    print(f"\nScanning {len(run_dirs)} run(s) in {runs_dir} …  "
          f"(last_n={args.last_n}, workers={n_workers})")
    results = []
    skipped = []
    pbar = _tqdm(total=len(run_dirs), desc="Loading TB logs", unit="run",
                 dynamic_ncols=True, leave=True)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(summarize_run, rd, args.last_n): rd for rd in run_dirs}
        for fut in as_completed(futures):
            rd = futures[fut]
            try:
                r = fut.result(timeout=120)
            except Exception as exc:
                r = None
                skipped.append(f"{rd.name} (error: {exc})")
                pbar.set_postfix_str(f"{rd.name[:25]} ERROR", refresh=False)
                pbar.update(1)
                continue
            if r is None:
                skipped.append(rd.name)
            else:
                results.append(r)
            cached = "(cached)" if (rd / ".scan_cache.json").exists() else "(parsed)"
            pbar.set_postfix_str(f"{rd.name[:30]} {cached}", refresh=False)
            pbar.update(1)
    pbar.close()
    for name in sorted(skipped):
        print(f"  [SKIP] {name}  (no TB events)")
    print(f"Loaded {len(results)} run(s).\n")

    if not results:
        print("Nothing to show.")
        return

    def sort_key(r):
        v = r[args.sort]
        is_none = (v is None)
        neg = -v if isinstance(v, (int, float)) and v is not None else (v or "")
        return (is_none, neg)

    results.sort(key=sort_key)

    if args.problems_only:
        results = [r for r in results if r["problems"]]

    # ── main table ─────────────────────────────────────────────────────────────
    # RUN column: compact "env/method/sN" label — no path truncation artefacts.
    # Full run_dir is printed only in the detail sections below.
    def _run_label(r) -> str:
        env  = (r["env"]    or "?")[:7]
        mth  = (r["method"] or "?")[:10]
        seed = r["seed"] if r["seed"] is not None else "?"
        return f"{env}/{mth}/s{seed}"

    W = 24
    print(
        f"{'RUN':<{W}} {'STEPS':>8} {'F_EVAL':>7} {'MAX_E':>7} {'TRAIN_R':>8} "
        f"{'CONT_P':>7} {'H_MEAN':>7} {'UNC':>7} "
        f"{'STD':>5} {'ACT_G':>6} {'RSCALE':>9}  PROBLEMS"
    )
    print("─" * 180)

    for r in results:
        label = _run_label(r)
        probs_str = "  ".join(r["problems"]) if r["problems"] else "OK"
        spike_marker = "▲▼" if r["spike"] else "  "
        print(
            f"{label:<{W}} "
            f"{r['max_step']:>8,} "
            f"{_fmt(r['final_eval'], 1):>7} "
            f"{spike_marker}{_fmt(r['max_eval'], 1):>5} "
            f"{_fmt(r['final_train'], 1):>8} "
            f"{_fmt(r['mean_cont'],   2):>7} "
            f"{_fmt(r['mean_horizon'],1):>7} "
            f"{_fmt(r['mean_unc'],    2):>7} "
            f"{_fmt(r['mean_std'],    2):>5} "
            f"{_fmt(r['actor_grad'],  3):>6} "
            f"{_fmt(r['max_rscale'],  0):>9}  "
            f"{probs_str}"
        )

    # ── per-method summary ─────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Per-method: final_eval mean±std  |  peak_train mean  |  cont_prob mean")
    print("─" * 60)
    from collections import defaultdict
    by_method: dict[str, list] = defaultdict(list)
    for r in results:
        key = f"{r['env']}/{r['method']}"
        by_method[key].append(r)

    for key in sorted(by_method):
        runs = by_method[key]
        evals  = [r["final_eval"]  for r in runs if r["final_eval"]  is not None]
        trains = [r["peak_train"]  for r in runs if r["peak_train"]  is not None]
        conts  = [r["mean_cont"]   for r in runs if r["mean_cont"]   is not None]
        spikes = sum(1 for r in runs if r["spike"])
        e_mu, e_sd = (np.mean(evals), np.std(evals)) if evals else (None, None)
        t_mu       = np.mean(trains) if trains else None
        c_mu       = np.mean(conts)  if conts  else None
        print(
            f"  {key:<32}  n={len(runs)}"
            f"  eval={_fmt(e_mu,1)} ± {_fmt(e_sd,1)}"
            f"  train_peak={_fmt(t_mu,1)}"
            f"  cont={_fmt(c_mu,2)}"
            + (f"  ⚡ {spikes} spike-then-drop" if spikes else "")
        )

    # ── spike-then-drop analysis ───────────────────────────────────────────────
    spikes = [r for r in results if r["spike"]]
    if spikes:
        print("\n" + "─" * 60)
        print(f"  ⚡ Spike-then-drop runs ({len(spikes)}):")
        print("─" * 60)
        for r in spikes:
            drop = (r["max_eval"] or 0) - (r["final_eval"] or 0)
            print(f"  {r['run_dir'][:65]}")
            print(f"    peak={r['max_eval']:.0f} @ step {r['max_step']:,}  →  "
                  f"now={r['final_eval']:.1f}  (drop={drop:.0f})")

    # ── cont_prob diagnosis ────────────────────────────────────────────────────
    bad_cont = [r for r in results
                if r["mean_cont"] is not None
                and r["mean_cont"] < CONT_THRESHOLDS.get(r["method"], 0.70)]
    if bad_cont:
        print("\n" + "─" * 60)
        print("  ⚠  Low cont_prob (below horizon-adjusted threshold):")
        print("─" * 60)
        for r in bad_cont:
            thresh = CONT_THRESHOLDS.get(r["method"], 0.70)
            print(f"  {r['run_dir'][:65]}  cont={r['mean_cont']:.2f}  (thresh={thresh})")
    elif any(r["mean_cont"] is not None for r in results):
        good = [r["mean_cont"] for r in results if r["mean_cont"] is not None]
        print(f"\n  ✓ cont_prob looks healthy: mean={np.mean(good):.2f}  "
              f"min={np.min(good):.2f}  (expected >0.90 for H=5, >0.60 for H=20)")

    # ── problem report ─────────────────────────────────────────────────────────
    all_problems = [r for r in results if r["problems"]]
    if all_problems:
        print("\n" + "─" * 60)
        print(f"  ⚠  {len(all_problems)} run(s) with detected problems:")
        print("─" * 60)
        for r in all_problems:
            print(f"  {r['run_dir'][:70]}")
            for prob in r["problems"]:
                print(f"    → {prob}")

    # ── ASCII learning curves ──────────────────────────────────────────────────
    if args.curve:
        print("\n" + "─" * 60)
        print("  Learning curves (eval/return_mean)")
        print("─" * 60)
        for r in results:
            # Reuse series already loaded during summarize_run — no second Reload needed.
            steps = r["_eval_steps"]
            vals  = r["_eval_ret"]
            if len(vals) == 0:
                continue
            n = min(40, len(vals))
            idx = np.linspace(0, len(vals)-1, n, dtype=int)
            v_sub = vals[idx]
            s_sub = steps[idx]
            vmin, vmax = v_sub.min(), v_sub.max()
            scale = max(vmax - vmin, 1.0)
            bar_w = 40
            print(f"\n  {r['run_dir'][:60]}  [{vmin:.0f} … {vmax:.0f}]")
            for s, v in zip(s_sub, v_sub):
                filled = int((v - vmin) / scale * bar_w)
                bar = "█" * filled + "░" * (bar_w - filled)
                step_k = int(s) // 1000
                print(f"    {step_k:>5}k  {bar}  {v:6.1f}")

    # ── CSV export ─────────────────────────────────────────────────────────────
    if args.csv:
        csv_path = Path(args.csv)
        _skip = {"problems", "_eval_steps", "_eval_ret"}
        fieldnames = [k for k in results[0].keys() if k not in _skip]
        fieldnames.append("problems_str")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                row = {k: v for k, v in r.items() if k not in _skip}
                row["problems_str"] = "; ".join(r["problems"])
                w.writerow(row)
        print(f"\nCSV written to {csv_path}")


if __name__ == "__main__":
    main()
