"""
Learning curve comparison across runs.

Usage:
    python scripts/visualize_runs.py --runs_dir runs/ --out plots/

Reads TensorBoard event files and plots smoothed eval/return_mean curves
grouped by method (fixed_h5, fixed_h20, adaptive_5_10_20, …).

Options:
    --runs_dir  Directory containing per-run subdirectories (default: runs/)
    --metric    TensorBoard tag to plot (default: eval/return_mean)
    --smooth    EMA smoothing factor 0-1 (default: 0.6)
    --out       Output directory for plots (default: plots/)
    --show      Show interactive window in addition to saving
"""
from __future__ import annotations

import argparse
import os
import re
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib not installed: pip install matplotlib")

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    raise SystemExit("tensorboard not installed: pip install tensorboard")


# ── helpers ───────────────────────────────────────────────────────────────────

def _smooth(values: List[float], alpha: float) -> List[float]:
    """Exponential moving average."""
    if not values:
        return values
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * out[-1] + (1.0 - alpha) * v)
    return out


def _load_scalar(tb_dir: str, tag: str) -> Tuple[List[int], List[float]]:
    """Read a scalar tag from a TensorBoard events directory."""
    # 2000 points is ample for smooth learning-curve plots; avoids loading all events.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc = EventAccumulator(tb_dir, size_guidance={"scalars": 2000})
        acc.Reload()
    if tag not in acc.Tags()["scalars"]:
        return [], []
    events = acc.Scalars(tag)
    steps  = [int(e.step)  for e in events]
    values = [float(e.value) for e in events]
    return steps, values


def _load_multi(tb_dir: str, tags: List[str]) -> Dict[str, Tuple[List[int], List[float]]]:
    """Load multiple tags in one Reload() call."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc = EventAccumulator(tb_dir, size_guidance={"scalars": 2000})
        acc.Reload()
    available = set(acc.Tags().get("scalars", []))
    result = {}
    for tag in tags:
        if tag in available:
            events = acc.Scalars(tag)
            result[tag] = (
                [int(e.step) for e in events],
                [float(e.value) for e in events],
            )
        else:
            result[tag] = ([], [])
    return result


def _method_label(run_name: str) -> str:
    """Extract a short method label from a run directory name."""
    patterns = [
        (r"fixed_h(\d+)",        lambda m: f"Fixed H={m.group(1)}"),
        (r"adaptive_(\d+_\d+_\d+)", lambda m: f"Adaptive [{m.group(1).replace('_','/')}]"),
    ]
    for pat, fmt in patterns:
        m = re.search(pat, run_name)
        if m:
            return fmt(m)
    return run_name


def _seed_label(run_name: str) -> str:
    m = re.search(r"[_/]s(\d+)", run_name)
    return f"s{m.group(1)}" if m else ""


# ── main ──────────────────────────────────────────────────────────────────────

def load_all_runs(
    runs_dir: str,
    metrics: List[str],
) -> Dict[str, Dict[str, List[Tuple[List[int], List[float]]]]]:
    """
    Scan runs_dir for tb/ subdirectories and load all requested metrics in parallel.

    Returns: {metric: {method_label: [(steps, values), ...]}}
    """
    run_entries = []
    for run_name in sorted(os.listdir(runs_dir)):
        run_path = os.path.join(runs_dir, run_name)
        if not os.path.isdir(run_path):
            continue
        tb_path = os.path.join(run_path, "tb")
        if not os.path.isdir(tb_path):
            continue
        run_entries.append((run_name, tb_path))

    def _load_one(run_name, tb_path):
        return run_name, _load_multi(tb_path, metrics)

    workers = min(len(run_entries), 8)
    results_raw = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_load_one, name, path): name
                   for name, path in run_entries}
        for fut in as_completed(futures):
            run_name, data = fut.result()
            results_raw[run_name] = data

    # Organise into {metric: {method: [(steps, values)]}}
    by_metric: Dict[str, Dict[str, List]] = {m: defaultdict(list) for m in metrics}
    for run_name in sorted(results_raw):          # sorted for deterministic order
        data = results_raw[run_name]
        label = _method_label(run_name)
        for metric in metrics:
            steps, values = data[metric]
            if not steps:
                print(f"  [skip] {run_name}: metric '{metric}' not found")
                continue
            by_metric[metric][label].append((steps, values))

    return by_metric


def plot_curves(
    by_method: Dict[str, List[Tuple[List[int], List[float]]]],
    metric: str,
    smooth: float,
    out_dir: str,
    show: bool,
):
    os.makedirs(out_dir, exist_ok=True)

    # Colour cycle
    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(metric.replace("/", " / "))
    ax.set_title(f"Learning curves — {metric}")

    for idx, (method, runs) in enumerate(sorted(by_method.items())):
        color = COLORS[idx % len(COLORS)]

        # Collect all individual runs (thin lines)
        all_steps_list, all_vals_list = [], []
        for steps, values in runs:
            sv = _smooth(values, smooth)
            ax.plot(steps, sv, color=color, alpha=0.25, linewidth=0.8)
            all_steps_list.append(np.array(steps))
            all_vals_list.append(np.array(values))

        # Mean ± std ribbon if multiple seeds
        if len(runs) > 1:
            # Interpolate to a common step grid
            max_step   = min(s[-1] for s in all_steps_list if len(s))
            grid       = np.linspace(0, max_step, 500)
            interp_vals = np.array([
                np.interp(grid, s, v) for s, v in zip(all_steps_list, all_vals_list)
            ])
            mean_v = _smooth(interp_vals.mean(axis=0).tolist(), smooth)
            std_v  = interp_vals.std(axis=0)

            ax.plot(grid, mean_v, color=color, linewidth=2.0, label=method)
            ax.fill_between(
                grid,
                np.array(mean_v) - std_v,
                np.array(mean_v) + std_v,
                color=color, alpha=0.15,
            )
        else:
            sv = _smooth(runs[0][1], smooth)
            ax.plot(runs[0][0], sv, color=color, linewidth=2.0, label=method)

    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    safe_tag = metric.replace("/", "_")
    out_path = os.path.join(out_dir, f"curves_{safe_tag}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot learning curves from TensorBoard logs.")
    parser.add_argument("--runs_dir", default="runs/",       help="Root runs directory")
    parser.add_argument("--metric",   default="eval/return_mean", help="TensorBoard scalar tag")
    parser.add_argument("--smooth",   default=0.6, type=float, help="EMA smoothing (0=off, 0.9=heavy)")
    parser.add_argument("--out",      default="plots/",      help="Output directory for PNG files")
    parser.add_argument("--show",     action="store_true",   help="Show matplotlib window")
    args = parser.parse_args()

    # Load all metrics in one parallel pass (one Reload per run, not one per metric).
    metrics_to_load = [args.metric]
    if args.metric != "train/return_mean_20":
        metrics_to_load.append("train/return_mean_20")

    print(f"Scanning {args.runs_dir} for {len(metrics_to_load)} metric(s) in parallel ...")
    by_metric = load_all_runs(args.runs_dir, metrics_to_load)

    by_method = by_metric[args.metric]
    if not by_method:
        print("No data found. Check --runs_dir and --metric.")
        return

    print(f"Found {sum(len(v) for v in by_method.values())} run(s) across {len(by_method)} method(s).")
    plot_curves(by_method, args.metric, args.smooth, args.out, args.show)

    if args.metric != "train/return_mean_20":
        by_method2 = by_metric["train/return_mean_20"]
        if by_method2:
            plot_curves(by_method2, "train/return_mean_20", args.smooth, args.out, show=False)


if __name__ == "__main__":
    main()
