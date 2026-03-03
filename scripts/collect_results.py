"""
collect_results.py
==================
Extract TensorBoard metrics from all completed runs, compute
per-group statistics, and save CSVs + plots that feed directly
into the thesis LaTeX tables and figures.

Usage
-----
# After running experiments:
python scripts/collect_results.py

# Only extract a specific group
python scripts/collect_results.py --group main

# Skip re-reading TensorBoard (use cached raw_metrics.csv)
python scripts/collect_results.py --use_cache

Output (written to results/)
------------------------------
  raw_metrics.csv           -- one row per (exp, step): all logged scalars
  main_table.csv            -- final return mean±std per (method, env)
  sample_efficiency_table.csv -- steps to reach threshold per (method, env)
  auc_table.csv             -- area under eval curve per (method, env)
  ablation_metric_table.csv -- ablation: uncertainty metric
  ablation_ensemble_table.csv
  ablation_threshold_table.csv
  horizon_dist.csv          -- fraction of steps per horizon (adaptive only)
  figures/
    learning_curves_hopper.png
    learning_curves_walker2d.png
    horizon_dist_hopper.png
    horizon_dist_walker2d.png
    uncertainty_over_time.png
    wm_loss.png
"""

from __future__ import annotations

import argparse
import csv
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ── optional imports ──────────────────────────────────────────────────────────
try:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
        TENSORS,
        SCALARS,
    )
    HAS_TB = True
except ImportError:
    HAS_TB = False
    warnings.warn("tensorboard not found — cannot read TB logs. Install it first.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    warnings.warn("matplotlib not found — figures will not be generated.")

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "experiments" / "manifest.csv"
RESULTS_DIR   = ROOT / "results"
FIGURES_DIR   = RESULTS_DIR / "figures"

# ── metrics to extract from TensorBoard ────────────────────────────────────────
TB_SCALARS = [
    "eval/return_mean",
    "eval/best_return",
    "train/return_mean_20",
    "train/episode_return",
    "imagine/horizon_used",
    "imagine/uncertainty_mean",
    "loss/world_model",
    "loss/kl",
    "loss/obs",
    "loss/reward",
    "loss/actor",
    "loss/critic",
]

# Friendly display names for methods in plots/tables
METHOD_LABELS = {
    "fixed_h5":        "Fixed-H5",
    "fixed_h15":       "Fixed-H15",
    "fixed_h20":       "Fixed-H20",
    "adaptive_5_10_20": "Adaptive (ours)",
}
METHOD_ORDER = ["fixed_h5", "fixed_h15", "fixed_h20", "adaptive_5_10_20"]
ENV_LABELS   = {"Hopper-v4": "Hopper-v4", "Walker2d-v4": "Walker2d-v4"}

# Performance thresholds for sample-efficiency table
THRESHOLDS = {"Hopper-v4": 1000.0, "Walker2d-v4": 1500.0}

COLORS = {"fixed_h5": "#1f77b4", "fixed_h15": "#ff7f0e",
          "fixed_h20": "#2ca02c", "adaptive_5_10_20": "#d62728"}

# ── helpers ────────────────────────────────────────────────────────────────────

def load_manifest() -> List[dict]:
    if not MANIFEST_PATH.exists():
        print(f"[WARN] manifest not found at {MANIFEST_PATH}")
        return []
    with open(MANIFEST_PATH, newline="") as f:
        return list(csv.DictReader(f))


def find_tb_dir(run_dir: Path) -> Optional[Path]:
    tb_dir = run_dir / "tb"
    if tb_dir.exists():
        return tb_dir
    # fallback: search for any directory containing tfevents
    for p in run_dir.rglob("events.out.tfevents.*"):
        return p.parent
    return None


def read_tb_scalars(tb_dir: Path, tags: List[str]) -> Dict[str, List[tuple]]:
    """
    Returns {tag: [(step, value), ...]} for all requested tags that exist.
    """
    if not HAS_TB:
        return {}
    ea = EventAccumulator(str(tb_dir), size_guidance={SCALARS: 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    result = {}
    for tag in tags:
        if tag in available:
            events = ea.Scalars(tag)
            result[tag] = [(e.step, e.value) for e in events]
    return result


def smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Simple moving average."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def steps_to_threshold(steps: np.ndarray, values: np.ndarray,
                        threshold: float) -> Optional[int]:
    """First step where value >= threshold. None if never reached."""
    idx = np.argwhere(values >= threshold)
    return int(steps[idx[0, 0]]) if len(idx) > 0 else None


def area_under_curve(steps: np.ndarray, values: np.ndarray) -> float:
    """Trapezoidal AUC, normalised by total steps."""
    if len(steps) < 2:
        return 0.0
    return float(np.trapz(values, steps) / (steps[-1] - steps[0] + 1e-8))


# ── main extraction ────────────────────────────────────────────────────────────

def extract_all_metrics(manifest_rows: List[dict],
                        group_filter: Optional[str]) -> Dict[str, dict]:
    """
    Returns {exp_name: {tag: [(step, value), ...], 'meta': {...}}}.
    """
    all_data = {}
    for row in manifest_rows:
        if row.get("status") != "done":
            continue
        if group_filter and row.get("group") != group_filter:
            continue
        run_dir = Path(row["run_dir"])
        tb_dir  = find_tb_dir(run_dir)
        if tb_dir is None:
            print(f"[WARN] no TB dir found in {run_dir}")
            continue
        print(f"  reading {row['exp_name']} ... ", end="", flush=True)
        scalars = read_tb_scalars(tb_dir, TB_SCALARS)
        print(f"{len(scalars)} tags")
        all_data[row["exp_name"]] = {
            "meta": row,
            "scalars": scalars,
        }
    return all_data


def save_raw_csv(all_data: dict, out_path: Path) -> None:
    """Save one row per (exp_name, tag, step, value)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["exp_name", "group", "env_id", "method", "seed",
                         "tag", "step", "value"])
        for exp_name, data in all_data.items():
            meta = data["meta"]
            for tag, events in data["scalars"].items():
                for step, value in events:
                    writer.writerow([exp_name, meta["group"], meta["env_id"],
                                     meta["method"], meta["seed"],
                                     tag, step, value])
    print(f"[OUT] {out_path}")


# ── aggregation helpers ────────────────────────────────────────────────────────

def group_by(all_data: dict, keys: List[str]) -> Dict[tuple, List[dict]]:
    """Group all_data entries by the given meta-key tuple."""
    groups = defaultdict(list)
    for exp_name, data in all_data.items():
        key = tuple(data["meta"].get(k, "") for k in keys)
        groups[key].append(data)
    return groups


def compute_final_return(data_list: List[dict],
                         tag: str = "eval/return_mean",
                         last_n: int = 3) -> tuple[float, float]:
    """Mean±std of the last `last_n` eval points, averaged across seeds."""
    seed_finals = []
    for data in data_list:
        events = data["scalars"].get(tag, [])
        if not events:
            continue
        vals = [v for _, v in events]
        final = float(np.mean(vals[-last_n:])) if len(vals) >= last_n else float(np.mean(vals))
        seed_finals.append(final)
    if not seed_finals:
        return float("nan"), float("nan")
    return float(np.mean(seed_finals)), float(np.std(seed_finals))


def compute_auc(data_list: List[dict],
                tag: str = "eval/return_mean") -> tuple[float, float]:
    seed_aucs = []
    for data in data_list:
        events = data["scalars"].get(tag, [])
        if len(events) < 2:
            continue
        steps  = np.array([s for s, _ in events])
        values = np.array([v for _, v in events])
        seed_aucs.append(area_under_curve(steps, values))
    if not seed_aucs:
        return float("nan"), float("nan")
    return float(np.mean(seed_aucs)), float(np.std(seed_aucs))


def compute_steps_to_threshold(data_list: List[dict],
                                threshold: float,
                                tag: str = "eval/return_mean") -> str:
    seed_steps = []
    for data in data_list:
        events = data["scalars"].get(tag, [])
        if not events:
            continue
        steps  = np.array([s for s, _ in events])
        values = np.array([v for _, v in events])
        s = steps_to_threshold(steps, values, threshold)
        seed_steps.append(s)
    reached = [s for s in seed_steps if s is not None]
    if not reached:
        return "NR"
    mean_k = int(np.mean(reached)) // 1000
    return f"{mean_k}k"


# ── table builders ─────────────────────────────────────────────────────────────

def build_main_table(all_data: dict, out_path: Path) -> None:
    groups = group_by(all_data, ["env_id", "method"])
    rows = []
    for env in ["Hopper-v4", "Walker2d-v4"]:
        for method in METHOD_ORDER:
            key = (env, method)
            data_list = groups.get(key, [])
            mean, std = compute_final_return(data_list)
            auc_mean, auc_std = compute_auc(data_list)
            rows.append({
                "env_id": env, "method": method,
                "label": METHOD_LABELS.get(method, method),
                "n_seeds": len(data_list),
                "mean_return": f"{mean:.1f}" if not np.isnan(mean) else "NaN",
                "std_return":  f"{std:.1f}"  if not np.isnan(std)  else "NaN",
                "auc_mean":    f"{auc_mean:.1f}" if not np.isnan(auc_mean) else "NaN",
                "auc_std":     f"{auc_std:.1f}"  if not np.isnan(auc_std)  else "NaN",
            })
    _save_csv(rows, out_path)


def build_sample_efficiency_table(all_data: dict, out_path: Path) -> None:
    groups = group_by(all_data, ["env_id", "method"])
    rows = []
    for env in ["Hopper-v4", "Walker2d-v4"]:
        thresh = THRESHOLDS.get(env, 1000.0)
        for method in METHOD_ORDER:
            key = (env, method)
            data_list = groups.get(key, [])
            steps_str = compute_steps_to_threshold(data_list, thresh)
            rows.append({
                "env_id": env, "method": method,
                "label": METHOD_LABELS.get(method, method),
                "threshold": thresh,
                "steps_to_threshold": steps_str,
            })
    _save_csv(rows, out_path)


def build_ablation_table(all_data: dict, out_path: Path,
                         group_keys: List[str],
                         label_fn) -> None:
    groups = group_by(all_data, group_keys)
    rows = []
    for key, data_list in sorted(groups.items()):
        mean, std = compute_final_return(data_list)
        auc_mean, auc_std = compute_auc(data_list)
        row = {k: v for k, v in zip(group_keys, key)}
        row.update({
            "label": label_fn(key),
            "n_seeds": len(data_list),
            "mean_return": f"{mean:.1f}" if not np.isnan(mean) else "NaN",
            "std_return":  f"{std:.1f}"  if not np.isnan(std)  else "NaN",
            "auc_mean":    f"{auc_mean:.1f}" if not np.isnan(auc_mean) else "NaN",
            "auc_std":     f"{auc_std:.1f}"  if not np.isnan(auc_std)  else "NaN",
        })
        rows.append(row)
    _save_csv(rows, out_path)


def build_horizon_dist(all_data: dict, out_path: Path) -> None:
    """Fraction of updates selecting each horizon, over training."""
    rows = []
    for exp_name, data in all_data.items():
        if data["meta"]["method"] != "adaptive_5_10_20":
            continue
        events = data["scalars"].get("imagine/horizon_used", [])
        if not events:
            continue
        for step, h in events:
            rows.append({
                "exp_name": exp_name,
                "env_id": data["meta"]["env_id"],
                "seed": data["meta"]["seed"],
                "step": int(step),
                "horizon": int(h),
            })
    _save_csv(rows, out_path)


def _save_csv(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"[WARN] no data for {out_path.name}")
        return
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OUT] {out_path}")


# ── figure builders ────────────────────────────────────────────────────────────

def _aligned_mean_std(data_list: List[dict], tag: str,
                      n_points: int = 200
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate each seed's curve onto a common step grid,
    then return (steps, mean, std).
    """
    all_curves = []
    all_steps  = []
    for data in data_list:
        events = data["scalars"].get(tag, [])
        if len(events) < 2:
            continue
        steps  = np.array([s for s, _ in events], dtype=float)
        values = np.array([v for _, v in events], dtype=float)
        all_curves.append((steps, values))
        all_steps.append(steps)

    if not all_curves:
        return np.array([]), np.array([]), np.array([])

    # Common grid
    min_step = max(c[0][0]  for c in all_curves)
    max_step = min(c[0][-1] for c in all_curves)
    if min_step >= max_step:
        max_step = max(c[0][-1] for c in all_curves)
    grid = np.linspace(min_step, max_step, n_points)

    interp = [np.interp(grid, c[0], c[1]) for c in all_curves]
    interp = np.stack(interp, axis=0)   # [n_seeds, n_points]
    return grid, interp.mean(axis=0), interp.std(axis=0)


def plot_learning_curves(all_data: dict, env_id: str, out_path: Path) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    groups = group_by(all_data, ["env_id", "method"])

    for method in METHOD_ORDER:
        data_list = groups.get((env_id, method), [])
        # also accept data from main group only to avoid mixing ablations
        data_list = [d for d in data_list if d["meta"]["group"] == "main"]
        if not data_list:
            continue
        steps, mean, std = _aligned_mean_std(data_list, "eval/return_mean")
        if len(steps) == 0:
            continue
        label = METHOD_LABELS.get(method, method)
        color = COLORS.get(method, None)
        ax.plot(steps / 1e3, mean, label=label, color=color, linewidth=2)
        ax.fill_between(steps / 1e3, mean - std, mean + std,
                        alpha=0.2, color=color)

    ax.set_xlabel("Environment steps (×1k)")
    ax.set_ylabel("Evaluation return")
    ax.set_title(f"Learning curves — {env_id}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[FIG] {out_path}")


def plot_horizon_distribution(all_data: dict, env_id: str, out_path: Path) -> None:
    """Stacked area chart showing fraction of steps each horizon is selected."""
    if not HAS_MPL:
        return
    data_list = [d for d in all_data.values()
                 if d["meta"]["method"] == "adaptive_5_10_20"
                 and d["meta"]["env_id"] == env_id
                 and d["meta"]["group"] == "main"]
    if not data_list:
        return

    horizon_vals = [5, 10, 20]
    fig, axes = plt.subplots(1, len(data_list), figsize=(5 * len(data_list), 4),
                              sharey=True)
    if len(data_list) == 1:
        axes = [axes]

    for ax, data in zip(axes, data_list):
        events = data["scalars"].get("imagine/horizon_used", [])
        if not events:
            continue
        steps  = np.array([s for s, _ in events])
        vals   = np.array([v for _, v in events])

        # Rolling window fraction per horizon
        window = max(1, len(steps) // 50)
        fracs = {}
        for h in horizon_vals:
            is_h = (vals == h).astype(float)
            fracs[h] = np.convolve(is_h, np.ones(window) / window, mode="same")

        colors_h = ["#e41a1c", "#ff7f00", "#4daf4a"]
        bottoms  = np.zeros(len(steps))
        for h, color in zip(horizon_vals, colors_h):
            ax.fill_between(steps / 1e3, bottoms, bottoms + fracs[h],
                            alpha=0.7, color=color, label=f"H={h}")
            bottoms += fracs[h]

        ax.set_xlabel("Steps (×1k)")
        ax.set_title(f"Seed {data['meta']['seed']}")
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Fraction of updates")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Horizon distribution — {env_id} (Adaptive)", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] {out_path}")


def plot_uncertainty_over_time(all_data: dict, out_path: Path) -> None:
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for env_id in ["Hopper-v4", "Walker2d-v4"]:
        data_list = [d for d in all_data.values()
                     if d["meta"]["method"] == "adaptive_5_10_20"
                     and d["meta"]["env_id"] == env_id
                     and d["meta"]["group"] == "main"]
        if not data_list:
            continue
        steps, mean, std = _aligned_mean_std(data_list, "imagine/uncertainty_mean")
        if len(steps) == 0:
            continue
        ax.plot(steps / 1e3, mean, label=env_id, linewidth=2)
        ax.fill_between(steps / 1e3, mean - std, mean + std, alpha=0.2)
        plotted = True

    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Environment steps (×1k)")
    ax.set_ylabel("Ensemble disagreement")
    ax.set_title("Uncertainty over training (Adaptive agent)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[FIG] {out_path}")


def plot_wm_loss(all_data: dict, env_id: str, out_path: Path) -> None:
    if not HAS_MPL:
        return
    tags = ["loss/world_model", "loss/obs", "loss/reward", "loss/kl"]
    colors_t = ["black", "#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    groups = group_by(all_data, ["env_id", "method"])

    for method in METHOD_ORDER:
        data_list = groups.get((env_id, method), [])
        data_list = [d for d in data_list if d["meta"]["group"] == "main"]
        for tag, tc in zip(tags[:1], colors_t[:1]):  # just total loss
            steps, mean, std = _aligned_mean_std(data_list, tag)
            if not len(steps):
                continue
            label = f"{METHOD_LABELS.get(method, method)}"
            color = COLORS.get(method)
            ax.plot(steps / 1e3, smooth(mean), label=label, color=color, linewidth=1.5)

    ax.set_xlabel("Environment steps (×1k)")
    ax.set_ylabel("World model loss")
    ax.set_title(f"World model training loss — {env_id}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[FIG] {out_path}")


# ── entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ThesisWM results extractor")
    p.add_argument("--group", default=None,
                   help="Only process experiments in this group.")
    p.add_argument("--use_cache", action="store_true",
                   help="Skip TB reading; use existing raw_metrics.csv.")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load manifest
    manifest_rows = load_manifest()
    done = [r for r in manifest_rows if r.get("status") == "done"]
    print(f"\nManifest: {len(manifest_rows)} total, {len(done)} done")

    # 2. Extract TB data (or load cached)
    raw_csv = RESULTS_DIR / "raw_metrics.csv"
    if args.use_cache and raw_csv.exists():
        print(f"[CACHE] loading {raw_csv}")
        all_data: dict = {}
        with open(raw_csv, newline="") as f:
            for row in csv.DictReader(f):
                en = row["exp_name"]
                if en not in all_data:
                    all_data[en] = {
                        "meta": {k: row[k] for k in
                                 ["exp_name", "group", "env_id", "method", "seed"]},
                        "scalars": defaultdict(list),
                    }
                all_data[en]["scalars"][row["tag"]].append(
                    (int(row["step"]), float(row["value"]))
                )
    else:
        if not HAS_TB:
            print("[ERROR] tensorboard is required to read logs. "
                  "Install with: pip install tensorboard")
            return
        print("\nReading TensorBoard logs ...")
        all_data = extract_all_metrics(manifest_rows, args.group)
        save_raw_csv(all_data, raw_csv)

    if not all_data:
        print("[WARN] no data found. Have you run experiments yet?")
        print("       Run: python scripts/run_experiments.py --group main")
        return

    # 3. Build summary CSVs
    print("\nBuilding tables ...")
    build_main_table(all_data, RESULTS_DIR / "main_table.csv")
    build_sample_efficiency_table(all_data, RESULTS_DIR / "sample_efficiency_table.csv")

    main_data = {k: v for k, v in all_data.items() if v["meta"]["group"] == "main"}

    # Ablations (only if data available)
    abl_metric = {k: v for k, v in all_data.items() if v["meta"]["group"] == "ablation_metric"}
    if abl_metric:
        build_ablation_table(
            abl_metric,
            RESULTS_DIR / "ablation_metric_table.csv",
            group_keys=["env_id", "method", "uncertainty_metric"],
            label_fn=lambda key: key[2],  # metric name
        )

    abl_ens = {k: v for k, v in all_data.items() if v["meta"]["group"] == "ablation_ensemble"}
    if abl_ens:
        build_ablation_table(
            abl_ens,
            RESULTS_DIR / "ablation_ensemble_table.csv",
            group_keys=["env_id", "method", "world_model_ensemble_size"],
            label_fn=lambda key: f"N={key[2]}, {key[1]}",
        )

    abl_thresh = {k: v for k, v in all_data.items() if v["meta"]["group"] == "ablation_threshold"}
    if abl_thresh:
        build_ablation_table(
            abl_thresh,
            RESULTS_DIR / "ablation_threshold_table.csv",
            group_keys=["env_id", "method", "method_thresh_high", "method_thresh_mid"],
            label_fn=lambda key: f"τ_h={key[2]}, τ_m={key[3]}",
        )

    build_horizon_dist(all_data, RESULTS_DIR / "horizon_dist.csv")

    # 4. Build figures
    print("\nBuilding figures ...")
    plot_learning_curves(main_data, "Hopper-v4",   FIGURES_DIR / "learning_curves_hopper.png")
    plot_learning_curves(main_data, "Walker2d-v4", FIGURES_DIR / "learning_curves_walker2d.png")
    plot_horizon_distribution(all_data, "Hopper-v4",   FIGURES_DIR / "horizon_dist_hopper.png")
    plot_horizon_distribution(all_data, "Walker2d-v4", FIGURES_DIR / "horizon_dist_walker2d.png")
    plot_uncertainty_over_time(all_data, FIGURES_DIR / "uncertainty_over_time.png")
    plot_wm_loss(main_data, "Hopper-v4", FIGURES_DIR / "wm_loss.png")

    print(f"\n{'='*60}")
    print(f"All outputs written to: {RESULTS_DIR}")
    print("Next step: open thesis/thesis_skeleton.tex and fill in \\RESULT{{}} markers.")


if __name__ == "__main__":
    main()
