"""
run_experiments.py
==================
Master experiment orchestrator for ThesisWM.

Usage
-----
# Dry-run: print commands without executing
python scripts/run_experiments.py --group main --dry_run

# Run the main comparison group (sequentially)
python scripts/run_experiments.py --group main

# Run a specific ablation group
python scripts/run_experiments.py --group ablation_metric
python scripts/run_experiments.py --group ablation_ensemble
python scripts/run_experiments.py --group ablation_threshold

# Run everything
python scripts/run_experiments.py --group all

# Skip experiments already completed (manifest says 'done')
python scripts/run_experiments.py --group main --skip_done

Experiment Groups
-----------------
  main              -- Hopper + Walker2d × {fixed_h5, fixed_h15, fixed_h20, adaptive} × 3 seeds
  ablation_metric   -- Hopper × 3 uncertainty metrics (adaptive only) × 3 seeds
  ablation_ensemble -- Hopper × ensemble size {1→fixed, 2→adaptive, 3→adaptive} × 3 seeds
  ablation_threshold-- Hopper × 4 threshold pairs (adaptive) × 3 seeds

Each completed run is recorded in experiments/manifest.csv so that
collect_results.py knows where to look for TensorBoard logs.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, fields
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # repo root
MANIFEST_PATH = ROOT / "experiments" / "manifest.csv"

# ── experiment definitions ─────────────────────────────────────────────────────

@dataclass
class Experiment:
    group: str            # which ablation group this belongs to
    env_id: str
    method: str           # fixed_h5 | fixed_h15 | fixed_h20 | adaptive_5_10_20
    seed: int
    extra_overrides: List[str]  # additional Hydra overrides
    base_config: str = "config"

    @property
    def exp_name(self) -> str:
        """Unique name used for run_dir and manifest."""
        env_tag = self.env_id.replace("-", "").lower()
        overrides_tag = "_".join(
            o.replace("=", "-").replace(".", "_").replace("/", "-")
            for o in self.extra_overrides
            if any(k in o for k in ["metric", "ensemble", "thresh", "horizon_fixed"])
        )
        tag = f"{env_tag}_{self.method}"
        if overrides_tag:
            tag += f"_{overrides_tag}"
        tag += f"_seed{self.seed}"
        return tag

    def build_cmd(self) -> List[str]:
        overrides = [
            f"exp_name={self.exp_name}",
            f"env.id={self.env_id}",
            f"method.name={self.method}",
            f"seed={self.seed}",
            "training.resume=true",
        ] + self.extra_overrides
        return ["python", "train.py", "--config", self.base_config] + overrides

    def run_dir(self) -> Path:
        return ROOT / "runs" / self.exp_name


# ── MAIN COMPARISON ────────────────────────────────────────────────────────────
ENVS_MAIN = ["Hopper-v4", "Walker2d-v4"]
SEEDS = [0, 1, 2]

FIXED_VARIANTS = {
    "fixed_h5":  ["imagination.horizon_fixed=5"],
    "fixed_h15": ["imagination.horizon_fixed=15"],
    "fixed_h20": ["imagination.horizon_fixed=20"],
}

ADAPTIVE_OVERRIDES_PER_ENV = {
    "Hopper-v4":   ["method.uncertainty_metric=next_obs_mean_l2",
                    "method.thresh_high=0.30", "method.thresh_mid=0.12",
                    "imagination.horizons=[5,10,20]"],
    "Walker2d-v4": ["method.uncertainty_metric=next_obs_mean_l2",
                    "method.thresh_high=0.35", "method.thresh_mid=0.15",
                    "imagination.horizons=[5,10,20]"],
}

SHARED_TRAINING_OVERRIDES = [
    "training.total_steps=1000000",
    "training.steps_per_chunk=200000",
    "training.start_learning=5000",
    "training.checkpoint_every_steps=10000",
    "training.log_every_steps=2000",
    "training.eval_every_steps=2000",
    "training.eval_episodes=5",
]


def make_main_experiments() -> List[Experiment]:
    exps = []
    for env in ENVS_MAIN:
        for method, method_overrides in FIXED_VARIANTS.items():
            for seed in SEEDS:
                exps.append(Experiment(
                    group="main",
                    env_id=env,
                    method=method,
                    seed=seed,
                    extra_overrides=method_overrides + SHARED_TRAINING_OVERRIDES,
                ))
        # Adaptive
        for seed in SEEDS:
            exps.append(Experiment(
                group="main",
                env_id=env,
                method="adaptive_5_10_20",
                seed=seed,
                extra_overrides=ADAPTIVE_OVERRIDES_PER_ENV[env] + SHARED_TRAINING_OVERRIDES,
            ))
    return exps


# ── ABLATION: UNCERTAINTY METRIC ───────────────────────────────────────────────

UNCERTAINTY_METRICS = ["latent_mean_l2", "next_obs_mean_l2", "gaussian_kl"]

def make_ablation_metric_experiments() -> List[Experiment]:
    exps = []
    for metric in UNCERTAINTY_METRICS:
        for seed in SEEDS:
            exps.append(Experiment(
                group="ablation_metric",
                env_id="Hopper-v4",
                method="adaptive_5_10_20",
                seed=seed,
                extra_overrides=[
                    f"method.uncertainty_metric={metric}",
                    "method.thresh_high=0.30",
                    "method.thresh_mid=0.12",
                    "imagination.horizons=[5,10,20]",
                ] + SHARED_TRAINING_OVERRIDES,
            ))
    return exps


# ── ABLATION: ENSEMBLE SIZE ────────────────────────────────────────────────────
# N=1 with adaptive is impossible (no disagreement), so we test:
#   - N=1, Fixed-H15  (no ensemble benefit)
#   - N=2, Fixed-H15  (ensemble WM but fixed horizon)
#   - N=2, Adaptive   (our method)
#   - N=3, Adaptive   (larger ensemble)

def make_ablation_ensemble_experiments() -> List[Experiment]:
    configs = [
        (1, "fixed_h15",      ["imagination.horizon_fixed=15", "world_model.ensemble_size=1"]),
        (2, "fixed_h15",      ["imagination.horizon_fixed=15", "world_model.ensemble_size=2"]),
        (2, "adaptive_5_10_20", ["method.uncertainty_metric=next_obs_mean_l2",
                                  "method.thresh_high=0.30", "method.thresh_mid=0.12",
                                  "imagination.horizons=[5,10,20]", "world_model.ensemble_size=2"]),
        (3, "adaptive_5_10_20", ["method.uncertainty_metric=next_obs_mean_l2",
                                  "method.thresh_high=0.30", "method.thresh_mid=0.12",
                                  "imagination.horizons=[5,10,20]", "world_model.ensemble_size=3"]),
    ]
    exps = []
    for n, method, overrides in configs:
        for seed in SEEDS:
            exps.append(Experiment(
                group="ablation_ensemble",
                env_id="Hopper-v4",
                method=method,
                seed=seed,
                extra_overrides=overrides + SHARED_TRAINING_OVERRIDES,
            ))
    return exps


# ── ABLATION: THRESHOLD SENSITIVITY ───────────────────────────────────────────
# Vary (thresh_high, thresh_mid) for Adaptive on Hopper
THRESHOLD_PAIRS = [
    (0.20, 0.08),
    (0.30, 0.12),   # default
    (0.40, 0.16),
    (0.50, 0.20),
]

def make_ablation_threshold_experiments() -> List[Experiment]:
    exps = []
    for (th, tm) in THRESHOLD_PAIRS:
        for seed in SEEDS:
            exps.append(Experiment(
                group="ablation_threshold",
                env_id="Hopper-v4",
                method="adaptive_5_10_20",
                seed=seed,
                extra_overrides=[
                    "method.uncertainty_metric=next_obs_mean_l2",
                    f"method.thresh_high={th}",
                    f"method.thresh_mid={tm}",
                    "imagination.horizons=[5,10,20]",
                ] + SHARED_TRAINING_OVERRIDES,
            ))
    return exps


# ── REGISTRY ───────────────────────────────────────────────────────────────────

EXPERIMENT_GROUPS = {
    "main":               make_main_experiments,
    "ablation_metric":    make_ablation_metric_experiments,
    "ablation_ensemble":  make_ablation_ensemble_experiments,
    "ablation_threshold": make_ablation_threshold_experiments,
}


# ── MANIFEST ───────────────────────────────────────────────────────────────────

MANIFEST_FIELDS = [
    "exp_name", "group", "env_id", "method", "seed",
    "status",           # pending | running | done | failed
    "run_dir",
    "start_time", "end_time",
    "final_eval_return",
    "cmd",
]


def load_manifest() -> dict[str, dict]:
    """Returns {exp_name: row_dict}."""
    if not MANIFEST_PATH.exists():
        return {}
    rows = {}
    with open(MANIFEST_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["exp_name"]] = row
    return rows


def save_manifest(rows: dict[str, dict]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)


def upsert_manifest(rows: dict[str, dict], exp: Experiment, **kwargs) -> None:
    row = rows.get(exp.exp_name, {
        "exp_name": exp.exp_name,
        "group": exp.group,
        "env_id": exp.env_id,
        "method": exp.method,
        "seed": exp.seed,
        "status": "pending",
        "run_dir": str(exp.run_dir()),
        "start_time": "",
        "end_time": "",
        "final_eval_return": "",
        "cmd": " ".join(exp.build_cmd()),
    })
    row.update(kwargs)
    rows[exp.exp_name] = row


# ── RUNNER ─────────────────────────────────────────────────────────────────────

def run_experiment(exp: Experiment, dry_run: bool, manifest: dict[str, dict]) -> bool:
    """Run a single experiment. Returns True if succeeded."""
    cmd = exp.build_cmd()
    print("\n" + "=" * 72)
    print(f"[RUN] {exp.exp_name}")
    print(f"  Group  : {exp.group}")
    print(f"  Env    : {exp.env_id}")
    print(f"  Method : {exp.method}")
    print(f"  Seed   : {exp.seed}")
    print(f"  Cmd    : {' '.join(cmd)}")
    print("=" * 72)

    if dry_run:
        print("  [DRY RUN — skipping]")
        return True

    upsert_manifest(manifest, exp, status="running",
                    start_time=datetime.now().isoformat())
    save_manifest(manifest)

    t0 = time.time()
    try:
        result = subprocess.run(cmd, check=True, cwd=str(ROOT))
        elapsed = time.time() - t0
        upsert_manifest(manifest, exp, status="done",
                        end_time=datetime.now().isoformat())
        save_manifest(manifest)
        print(f"[DONE] {exp.exp_name} in {elapsed/60:.1f} min")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t0
        upsert_manifest(manifest, exp, status="failed",
                        end_time=datetime.now().isoformat())
        save_manifest(manifest)
        print(f"[FAILED] {exp.exp_name} after {elapsed/60:.1f} min — returncode {e.returncode}")
        return False


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ThesisWM experiment orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--group",
        choices=list(EXPERIMENT_GROUPS.keys()) + ["all"],
        default="main",
        help="Which experiment group to run.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without running them.",
    )
    p.add_argument(
        "--skip_done", action="store_true",
        help="Skip experiments that are already marked 'done' in the manifest.",
    )
    p.add_argument(
        "--list", action="store_true",
        help="List all experiments in the selected group and exit.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Collect experiments
    if args.group == "all":
        experiments = []
        for fn in EXPERIMENT_GROUPS.values():
            experiments.extend(fn())
    else:
        experiments = EXPERIMENT_GROUPS[args.group]()

    if args.list:
        print(f"\n{'EXP NAME':<70} {'GROUP':<22} {'METHOD':<22} {'SEED'}")
        print("-" * 130)
        for exp in experiments:
            print(f"{exp.exp_name:<70} {exp.group:<22} {exp.method:<22} {exp.seed}")
        print(f"\nTotal: {len(experiments)} experiments")
        return

    manifest = load_manifest()

    # Register all experiments in the manifest (status=pending if new)
    for exp in experiments:
        if exp.exp_name not in manifest:
            upsert_manifest(manifest, exp, status="pending")
    save_manifest(manifest)

    print(f"\n==> Running {len(experiments)} experiments (group={args.group})")

    n_success = 0
    n_skip = 0
    n_fail = 0

    for exp in experiments:
        row = manifest.get(exp.exp_name, {})
        if args.skip_done and row.get("status") == "done":
            print(f"[SKIP] {exp.exp_name} (already done)")
            n_skip += 1
            continue

        ok = run_experiment(exp, dry_run=args.dry_run, manifest=manifest)
        if ok:
            n_success += 1
        else:
            n_fail += 1

    print(f"\n{'='*72}")
    print(f"Finished: {n_success} ok / {n_skip} skipped / {n_fail} failed")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
