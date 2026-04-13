"""
make_thesis_figures.py
======================
Generate all thesis figures and tables from TensorBoard logs.

Outputs (all written to thesis/figures/ and thesis/tables/):
  figures/
    training_curves_<env>.pdf   — eval_return vs steps, mean±std per method
    horizon_dist.pdf             — horizon usage distribution for adaptive runs
    uncertainty_hist.pdf         — ensemble uncertainty histogram per env
    rscale_vs_return.pdf         — return_scale vs eval_return scatter
  tables/
    main_results.tex             — LaTeX tabular: method × env mean±std
    fix_timeline.tex             — LaTeX tabular: improvement per fix
    horizon_stats.tex            — horizon % usage per adaptive run

Usage:
    python scripts/make_thesis_figures.py --runs_dir runs/
    python scripts/make_thesis_figures.py --runs_dir runs/ --filter hopper
    python scripts/make_thesis_figures.py --runs_dir runs/ --no_save   # preview only
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Optional imports ───────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed — skipping figure generation (tables still written)")

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kwargs):
        return it

# Import caching helpers from scan_logs (no re-parsing needed if cache exists)
from scripts.scan_logs import (
    _load_cache_arrays,
    _cache_valid,
    _stream_scalars,
    _save_cache,
    _parse_name,
)

# ── Constants ─────────────────────────────────────────────────────────────────

METHOD_ORDER  = ["fixed_h5", "fixed_h15", "fixed_h20", "adaptive"]
METHOD_LABELS = {
    "fixed_h5":  "Fixed H=5",
    "fixed_h15": "Fixed H=15",
    "fixed_h20": "Fixed H=20",
    "adaptive":  "Adaptive (ours)",
}
METHOD_COLORS = {
    "fixed_h5":  "#9e9e9e",
    "fixed_h15": "#ff7f0e",
    "fixed_h20": "#1f77b4",
    "adaptive":  "#2ca02c",
}
METHOD_LS = {
    "fixed_h5":  ":",
    "fixed_h15": "--",
    "fixed_h20": "-.",
    "adaptive":  "-",
}
ENV_ORDER = ["Hopper", "Walker2d"]

# ── Data loading ───────────────────────────────────────────────────────────────

def _load_run_arrays(run_dir: Path) -> Optional[dict]:
    """Load TB arrays from cache (or re-parse if needed). Returns {tag: (steps, vals)}."""
    tb_dir = run_dir / "tb"
    if not tb_dir.exists():
        return None
    if _cache_valid(run_dir, tb_dir):
        arrays = _load_cache_arrays(run_dir)
        if arrays is not None:
            return arrays
    arrays = _stream_scalars(tb_dir)
    if arrays:
        _save_cache(run_dir, tb_dir, arrays)
    return arrays


def _load_worker(run_dir_str: str) -> Tuple[str, Optional[dict]]:
    run_dir = Path(run_dir_str)
    arrays = _load_run_arrays(run_dir)
    return str(run_dir), arrays


def load_all_runs(runs_dir: Path, filter_str: str = "", workers: int = 0
                  ) -> Dict[str, dict]:
    """
    Returns {run_dir_str: {"env": str, "method": str, "seed": int, "arrays": dict}}.
    """
    run_dirs = sorted(
        p for p in runs_dir.iterdir()
        if p.is_dir() and (p / "tb").exists()
        and (filter_str.lower() in p.name.lower() if filter_str else True)
    )
    if not run_dirs:
        print(f"[WARN] No runs found in {runs_dir} (filter={filter_str!r})")
        return {}

    n_workers = workers or min(os.cpu_count() or 4, len(run_dirs))
    results: Dict[str, dict] = {}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_load_worker, str(d)): d for d in run_dirs}
        for fut in _tqdm(as_completed(futures), total=len(futures), desc="Loading runs"):
            run_dir_str, arrays = fut.result()
            if arrays is None:
                continue
            run_dir = Path(run_dir_str)
            env, method, seed = _parse_name(run_dir.name)
            if env == "?" or method == "?":
                continue
            results[run_dir_str] = {
                "env":     env,
                "method":  method,
                "seed":    seed,
                "name":    run_dir.name,
                "arrays":  arrays,
            }
    print(f"Loaded {len(results)} runs.")
    return results


# ── Interpolation helpers ──────────────────────────────────────────────────────

def _interp_to_grid(steps: np.ndarray, vals: np.ndarray,
                    grid: np.ndarray) -> np.ndarray:
    """Linearly interpolate (steps, vals) onto grid. Outside range → NaN."""
    if len(steps) < 2:
        return np.full(len(grid), np.nan)
    out = np.interp(grid, steps, vals, left=np.nan, right=np.nan)
    return out


def build_curves(runs: Dict[str, dict],
                 tag: str = "eval/return_mean",
                 n_grid: int = 200,
                 max_steps: Optional[int] = None,
                 ) -> Dict[str, Dict[str, dict]]:
    """
    Build {env: {method: {"mean": arr, "std": arr, "steps": arr, "seeds": [arrs]}}}
    across seeds, on a common step grid.
    """
    # Collect (steps, vals) per (env, method)
    raw: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    all_max_steps = []

    for info in runs.values():
        steps, vals = info["arrays"].get(tag, (np.array([]), np.array([])))
        if len(steps) == 0:
            continue
        raw[info["env"]][info["method"]].append((steps, vals))
        all_max_steps.append(float(steps[-1]))

    if not all_max_steps:
        return {}

    cap = max_steps or float(np.percentile(all_max_steps, 80))
    grid = np.linspace(0, cap, n_grid)

    curves: Dict[str, Dict[str, dict]] = {}
    for env, methods in raw.items():
        curves[env] = {}
        for method, seed_list in methods.items():
            interped = [_interp_to_grid(s, v, grid) for s, v in seed_list]
            mat = np.vstack(interped)  # [n_seeds, n_grid]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(mat, axis=0)
                std  = np.nanstd(mat,  axis=0)
            curves[env][method] = {"steps": grid, "mean": mean, "std": std,
                                   "seeds": interped, "n": len(seed_list)}
    return curves


# ── Aggregate stats ────────────────────────────────────────────────────────────

def compute_final_stats(runs: Dict[str, dict],
                        tag: str = "eval/return_mean",
                        last_frac: float = 0.25,
                        ) -> Dict[str, Dict[str, dict]]:
    """
    {env: {method: {"mean": float, "std": float, "max": float, "n": int, "seeds": [float]}}}
    Using last `last_frac` of each run's steps.
    """
    raw: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for info in runs.values():
        steps, vals = info["arrays"].get(tag, (np.array([]), np.array([])))
        if len(vals) == 0:
            continue
        n = max(1, int(len(vals) * last_frac))
        final_val = float(np.mean(vals[-n:]))
        raw[info["env"]][info["method"]].append(final_val)

    stats: Dict[str, Dict[str, dict]] = {}
    for env, methods in raw.items():
        stats[env] = {}
        for method, seed_vals in methods.items():
            arr = np.array(seed_vals)
            stats[env][method] = {
                "mean":  float(np.mean(arr)),
                "std":   float(np.std(arr)),
                "max":   float(np.max(arr)),
                "n":     len(arr),
                "seeds": seed_vals,
            }
    return stats


def compute_horizon_dist(runs: Dict[str, dict]) -> Dict[str, Dict[str, dict]]:
    """
    For adaptive runs: {env: {seed_name: {"H=5": pct, "H=10": pct, "H=20": pct, "uncertainty": float}}}
    """
    dist: Dict[str, Dict[str, dict]] = defaultdict(dict)

    for name, info in runs.items():
        if info["method"] != "adaptive":
            continue
        env = info["env"]
        seed = info["seed"]
        arrays = info["arrays"]

        h_steps, h_vals = arrays.get("imagine/horizon_used", (np.array([]), np.array([])))
        u_steps, u_vals = arrays.get("imagine/uncertainty_mean", (np.array([]), np.array([])))

        if len(h_vals) == 0:
            continue

        total = len(h_vals)
        h_dist = {}
        for h in [5, 10, 20]:
            h_dist[f"H={h}"] = float(np.sum(h_vals == h) / total * 100)

        unc = float(np.median(u_vals)) if len(u_vals) > 0 else float("nan")
        dist[env][f"s{seed}"] = {**h_dist, "uncertainty_med": unc,
                                  "h_avg": float(np.mean(h_vals))}
    return dict(dist)


# ── Figure generation ──────────────────────────────────────────────────────────

def plot_training_curves(curves: Dict[str, Dict[str, dict]],
                         fig_dir: Path, save: bool = True) -> None:
    """One subplot per environment, methods overlaid with ±1 std band."""
    if not HAS_MPL:
        return

    envs = [e for e in ENV_ORDER if e in curves]
    if not envs:
        return

    fig, axes = plt.subplots(1, len(envs), figsize=(5.5 * len(envs), 4.5))
    if len(envs) == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        env_data = curves[env]
        for method in METHOD_ORDER:
            if method not in env_data:
                continue
            c = env_data[method]
            color = METHOD_COLORS[method]
            ls    = METHOD_LS[method]
            label = METHOD_LABELS[method]
            steps_k = c["steps"] / 1_000

            ax.plot(steps_k, c["mean"], color=color, linestyle=ls,
                    linewidth=2.0, label=f"{label} (n={c['n']})")
            ax.fill_between(steps_k,
                            c["mean"] - c["std"],
                            c["mean"] + c["std"],
                            alpha=0.15, color=color)

        ax.set_title(env, fontsize=13)
        ax.set_xlabel("Environment steps (×10³)", fontsize=11)
        ax.set_ylabel("Eval return (mean ± std)", fontsize=11)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)

    plt.suptitle("Evaluation Return — ThesisWM Experiment Comparison", fontsize=13, y=1.01)
    plt.tight_layout()

    out = fig_dir / "training_curves.pdf"
    if save:
        plt.savefig(out, bbox_inches="tight")
        print(f"  Saved {out}")
    else:
        plt.show()
    plt.close()


def plot_horizon_dist(dist: Dict[str, Dict[str, dict]],
                      fig_dir: Path, save: bool = True) -> None:
    """Bar chart: horizon usage (H=5/10/20) for each adaptive run."""
    if not HAS_MPL or not dist:
        return

    all_runs = [(env, seed_key, info)
                for env, seeds in dist.items()
                for seed_key, info in seeds.items()]

    if not all_runs:
        return

    labels = [f"{env}/{sk}" for env, sk, _ in all_runs]
    h5_pcts  = [info["H=5"]  for _, _, info in all_runs]
    h10_pcts = [info["H=10"] for _, _, info in all_runs]
    h20_pcts = [info["H=20"] for _, _, info in all_runs]

    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.1), 4))
    b5  = ax.bar(x, h5_pcts,  width, label="H=5",  color="#d62728")
    b10 = ax.bar(x, h10_pcts, width, bottom=h5_pcts, label="H=10", color="#ff7f0e")
    b20 = ax.bar(x, h20_pcts, width,
                 bottom=np.array(h5_pcts) + np.array(h10_pcts),
                 label="H=20", color="#1f77b4")

    # Annotate with median uncertainty
    for i, (_, _, info) in enumerate(all_runs):
        u = info.get("uncertainty_med", float("nan"))
        if np.isfinite(u):
            ax.text(i, 103, f"ū={u:.2f}", ha="center", va="bottom", fontsize=8, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% of actor-critic updates", fontsize=11)
    ax.set_ylim(0, 120)
    ax.set_title("Adaptive Horizon Distribution per Run", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = fig_dir / "horizon_dist.pdf"
    if save:
        plt.savefig(out, bbox_inches="tight")
        print(f"  Saved {out}")
    else:
        plt.show()
    plt.close()


def plot_rscale_vs_return(runs: Dict[str, dict],
                          fig_dir: Path, save: bool = True) -> None:
    """Scatter: final eval_return vs median return_scale, coloured by method."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, len(ENV_ORDER), figsize=(10, 4))
    for ax, env in zip(axes, ENV_ORDER):
        for method in METHOD_ORDER:
            xs, ys = [], []
            for info in runs.values():
                if info["env"] != env or info["method"] != method:
                    continue
                _, ret = info["arrays"].get("eval/return_mean",
                                            (np.array([]), np.array([])))
                _, rs  = info["arrays"].get("value/return_scale",
                                            (np.array([]), np.array([])))
                if len(ret) == 0 or len(rs) == 0:
                    continue
                xs.append(float(np.median(rs[-50:])))
                ys.append(float(np.mean(ret[-20:])))
            if xs:
                ax.scatter(xs, ys, color=METHOD_COLORS[method],
                           label=METHOD_LABELS[method], s=70, alpha=0.8,
                           marker={"fixed_h5": "x", "fixed_h15": "s",
                                   "fixed_h20": "^", "adaptive": "o"}[method])

        ax.axvline(100, color="red", linestyle="--", linewidth=1, alpha=0.6,
                   label="rscale cap (100)")
        ax.set_xlabel("Median return_scale", fontsize=11)
        ax.set_ylabel("Final eval return (mean last 20)", fontsize=11)
        ax.set_title(env, fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Return Scale vs Final Performance", fontsize=12)
    plt.tight_layout()
    out = fig_dir / "rscale_vs_return.pdf"
    if save:
        plt.savefig(out, bbox_inches="tight")
        print(f"  Saved {out}")
    else:
        plt.show()
    plt.close()


# ── Table generation ───────────────────────────────────────────────────────────

def _pm(mean: float, std: float, n: int) -> str:
    """Format as mean±std, bold if max in its column."""
    if np.isnan(mean):
        return "---"
    return f"{mean:.1f} \\pm {std:.1f}"


def write_main_results_table(stats: Dict[str, Dict[str, dict]],
                             out_path: Path) -> None:
    """
    LaTeX table: rows = methods, column pairs = (env mean±std, env max).
    """
    envs = [e for e in ENV_ORDER if e in stats]
    methods = [m for m in METHOD_ORDER if any(m in stats.get(e, {}) for e in envs)]

    lines = [
        "% Auto-generated by make_thesis_figures.py",
        "% \\input{tables/main_results.tex}",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Final evaluation return (mean $\\pm$ std over seeds, last 25\\%% of training).}",
        "  \\label{tab:main_results}",
        "  \\begin{tabular}{l" + "".join(["cc"] * len(envs)) + "}",
        "    \\toprule",
    ]

    # Header
    env_headers = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{\\textbf{{{e}}}}}" for e in envs
    )
    lines.append(f"    \\textbf{{Method}} & {env_headers} \\\\")
    sub_headers = " & ".join(["Final Ret. & Peak Ret."] * len(envs))
    lines.append(f"    & {sub_headers} \\\\")
    lines.append("    \\midrule")

    # Identify best final return per env for bolding
    best: Dict[str, float] = {}
    for env in envs:
        vals = [stats[env][m]["mean"] for m in methods if m in stats.get(env, {})]
        best[env] = max(vals) if vals else float("nan")

    for method in methods:
        label = METHOD_LABELS[method]
        cells = []
        for env in envs:
            info = stats.get(env, {}).get(method)
            if info is None:
                cells.extend(["---", "---"])
            else:
                m_str = _pm(info["mean"], info["std"], info["n"])
                if abs(info["mean"] - best.get(env, float("nan"))) < 0.5:
                    m_str = f"\\textbf{{{m_str}}}"
                max_str = f"{info['max']:.1f}"
                cells.extend([m_str, max_str])
        lines.append(f"    {label} & {' & '.join(cells)} \\\\")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]

    out_path.write_text("\n".join(lines))
    print(f"  Saved {out_path}")


def write_horizon_table(dist: Dict[str, Dict[str, dict]],
                        out_path: Path) -> None:
    """LaTeX table: adaptive runs with horizon distribution %."""
    if not dist:
        out_path.write_text("% No adaptive runs found.\n")
        return

    lines = [
        "% Auto-generated by make_thesis_figures.py",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Horizon selection distribution for adaptive runs.}",
        "  \\label{tab:horizon_dist}",
        "  \\begin{tabular}{llccccc}",
        "    \\toprule",
        "    Env & Seed & H=5 (\\%) & H=10 (\\%) & H=20 (\\%) & "
        "$\\bar{H}$ & $\\bar{u}$ (med) \\\\",
        "    \\midrule",
    ]

    for env in ENV_ORDER:
        if env not in dist:
            continue
        for seed_key in sorted(dist[env].keys()):
            info = dist[env][seed_key]
            h5  = info.get("H=5",  0.0)
            h10 = info.get("H=10", 0.0)
            h20 = info.get("H=20", 0.0)
            havg = info.get("h_avg", float("nan"))
            unc  = info.get("uncertainty_med", float("nan"))
            lines.append(
                f"    {env} & {seed_key} & {h5:.1f} & {h10:.1f} & {h20:.1f} & "
                f"{havg:.1f} & {unc:.3f} \\\\"
            )
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]

    out_path.write_text("\n".join(lines))
    print(f"  Saved {out_path}")


def write_fix_timeline_table(out_path: Path,
                             stats_before: Optional[Dict] = None,
                             stats_after: Optional[Dict] = None) -> None:
    """
    Hardcoded + optionally data-driven table of fix effectiveness.
    Uses data from the analysis reports when no before/after snapshots are available.
    """
    # Hardcoded from report_20260412 (after fix1) and report_20260413 (after fix1, before fix3)
    # Format: (fix, problem, evidence_before, evidence_after, status)
    fixes = [
        (
            "Fix 0 (Apr 10): Threshold calibration",
            "Adaptive stuck at H=5 100\\% (untrained-ensemble uncertainty ≈11)",
            "thresh\\_high=3.0 → unc≫thresh: always H=5",
            "thresh\\_high=0.70 → unc∈[0.44,0.75]: uses H=10/H=20",
            "Resolved for Hopper",
        ),
        (
            "Fix 1 (Apr 11): symlog\\_clamp + return\\_scale",
            "ACTOR\\_GRAD\\_TINY in 18/23 runs; rscale 200–1000",
            "rscale med=200–800; actor\\_grad=0.001–0.005",
            "rscale med=30–100; actor\\_grad=0.01–0.07 (best runs)",
            "Resolved for H=15/H=20; H=5 still capped",
        ),
        (
            "Fix 2 (Apr 12): Walker2d threshold recalib.",
            "Walker2d adaptive stuck at H=5 (unc range 0.62–1.79≫Hopper 0.44–0.75)",
            "thresh\\_high=0.70: unc>0.70 → H=5 100\\%",
            "thresh\\_high=1.50: s1 (unc=0.97)→H=10, s2 (unc=1.07)→H=10",
            "Resolved",
        ),
        (
            "Fix 3 (Apr 13): entropy\\_coef + critic\\_lr",
            "SPIKE\\_THEN\\_DROP in 13/24 runs; entropy dominated actor loss",
            "entropy\\_coef=1e-2: policy\\_std frozen at 1.0 until sudden spike",
            "entropy\\_coef=3e-4: expected smooth learning from step 1",
            "Applied — verification pending",
        ),
    ]

    lines = [
        "% Auto-generated by make_thesis_figures.py",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Chronological fixes applied to stabilise training.}",
        "  \\label{tab:fix_timeline}",
        "  \\small",
        "  \\begin{tabular}{p{3.5cm}p{3.5cm}p{2.5cm}p{2.5cm}p{2.0cm}}",
        "    \\toprule",
        "    \\textbf{Fix} & \\textbf{Problem} & \\textbf{Before} & "
        "\\textbf{After} & \\textbf{Status} \\\\",
        "    \\midrule",
    ]

    for fix, prob, before, after, status in fixes:
        lines.append(f"    {fix} & {prob} & {before} & {after} & {status} \\\\[4pt]")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]

    out_path.write_text("\n".join(lines))
    print(f"  Saved {out_path}")


def write_text_summary(stats: Dict[str, Dict[str, dict]],
                       dist: Dict[str, Dict[str, dict]],
                       out_path: Path) -> None:
    """Human-readable summary of all results."""
    lines = ["=" * 70, "  ThesisWM Results Summary", "=" * 70, ""]

    for env in ENV_ORDER:
        if env not in stats:
            continue
        lines.append(f"  {env}")
        lines.append("  " + "-" * 40)
        lines.append(f"  {'Method':<18} {'Mean':>8} {'Std':>8} {'Max':>8} {'n':>4}")
        for method in METHOD_ORDER:
            info = stats[env].get(method)
            if info is None:
                continue
            lines.append(
                f"  {METHOD_LABELS[method]:<18} "
                f"{info['mean']:>8.1f} {info['std']:>8.1f} {info['max']:>8.1f} "
                f"{info['n']:>4d}"
            )
        lines.append("")

    if dist:
        lines += ["  Adaptive Horizon Distribution", "  " + "-" * 40]
        for env in ENV_ORDER:
            if env not in dist:
                continue
            for sk, info in sorted(dist[env].items()):
                lines.append(
                    f"  {env}/{sk}: H=5:{info['H=5']:.0f}%  H=10:{info['H=10']:.0f}%  "
                    f"H=20:{info['H=20']:.0f}%  ū={info['uncertainty_med']:.3f}  "
                    f"H̄={info['h_avg']:.1f}"
                )
        lines.append("")

    # Did adaptive help?
    lines.append("  Adaptive vs. Best Fixed Baseline")
    lines.append("  " + "-" * 40)
    for env in ENV_ORDER:
        if env not in stats:
            continue
        fixed_means = {m: stats[env][m]["mean"]
                       for m in ["fixed_h15", "fixed_h20"]
                       if m in stats[env]}
        if not fixed_means or "adaptive" not in stats[env]:
            continue
        best_fixed_method = max(fixed_means, key=lambda k: fixed_means[k])
        best_fixed = fixed_means[best_fixed_method]
        adap = stats[env]["adaptive"]["mean"]
        delta = adap - best_fixed
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"  {env}: adaptive={adap:.1f}  best_fixed={best_fixed:.1f} "
            f"({METHOD_LABELS[best_fixed_method]})  Δ={sign}{delta:.1f}"
        )
    lines.append("")
    lines.append("=" * 70)

    text = "\n".join(lines)
    out_path.write_text(text)
    print(f"  Saved {out_path}")
    print()
    print(text)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate thesis figures and tables.")
    ap.add_argument("--runs_dir",  default="runs", help="Root runs directory")
    ap.add_argument("--filter",    default="",     help="Filter run names (substring)")
    ap.add_argument("--out_dir",   default="thesis", help="Output root (figures/ and tables/ created here)")
    ap.add_argument("--no_save",   action="store_true", help="Preview figures instead of saving")
    ap.add_argument("--workers",   type=int, default=0, help="Parallel workers (0=auto)")
    ap.add_argument("--max_steps", type=int, default=0, help="X-axis cap for curves (0=auto)")
    args = ap.parse_args()

    runs_dir = (ROOT / args.runs_dir).resolve()
    out_root = (ROOT / args.out_dir).resolve()
    fig_dir  = out_root / "figures"
    tab_dir  = out_root / "tables"

    if not args.no_save:
        fig_dir.mkdir(parents=True, exist_ok=True)
        tab_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print(f"\nLoading runs from {runs_dir} …")
    runs = load_all_runs(runs_dir, args.filter, args.workers)
    if not runs:
        print("[ERROR] No runs loaded. Check --runs_dir path.")
        return

    # 2. Build curves and aggregate stats
    print("\nBuilding training curves …")
    max_steps = args.max_steps or None
    curves = build_curves(runs, tag="eval/return_mean", max_steps=max_steps)
    stats  = compute_final_stats(runs, tag="eval/return_mean")
    dist   = compute_horizon_dist(runs)

    # 3. Figures
    print("\nGenerating figures …")
    save = not args.no_save
    plot_training_curves(curves, fig_dir, save=save)
    plot_horizon_dist(dist, fig_dir, save=save)
    plot_rscale_vs_return(runs, fig_dir, save=save)

    # 4. Tables
    print("\nGenerating tables …")
    if not args.no_save:
        write_main_results_table(stats, tab_dir / "main_results.tex")
        write_horizon_table(dist, tab_dir / "horizon_dist.tex")
        write_fix_timeline_table(tab_dir / "fix_timeline.tex")
        write_text_summary(stats, dist, tab_dir / "summary.txt")
    else:
        write_text_summary(stats, dist, Path("/dev/null"))

    print("\nDone.")


if __name__ == "__main__":
    main()
