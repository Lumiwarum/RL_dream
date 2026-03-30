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
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    # ── Steps ───────────────────────────────────────────────────────────────────
    # 300k pilot: verify learning curves before committing to longer runs.
    # Hopper converges by ~250k; Walker2d shows a clear trend by 300k.
    # Bump to 500k / 1M after reviewing pilot results.
    "training.total_steps=300000",
    "training.steps_per_chunk=300000",  # must match total_steps (single-subprocess design)

    # ── Env collection ──────────────────────────────────────────────────────────
    "env.num_envs=8",

    # ── Learning schedule ───────────────────────────────────────────────────────
    "training.start_learning=5000",
    "training.checkpoint_every_steps=10000",
    "training.log_every_steps=2000",

    # ── Eval ────────────────────────────────────────────────────────────────────
    # Less frequent eval reduces the growing overhead as episodes get longer.
    # At 300k steps and ~80 FPS: eval fires ~30 times total (every 10k steps).
    "training.eval_every_steps=10000",
    "training.eval_episodes=5",
    "training.eval_max_episode_steps=1000",

    # ── Update ratio ────────────────────────────────────────────────────────────
    # Keep updates_per_step FIXED regardless of num_envs.
    # Scaling it with N gives the same FPS (GPU stays bottleneck); keeping it fixed
    # means N=8 collects 2x transitions per GPU update → ~2x FPS vs N=4.
    "training.updates_per_step=4",

    # ── Network size ────────────────────────────────────────────────────────────
    # 512/256 are DreamerV3 image-scale sizes. For state-vector tasks (Hopper=11-dim,
    # Walker2d=17-dim), 256/128 gives ~3x faster updates with no quality loss.
    "world_model.hidden_dim=256",
    "world_model.deter_dim=128",
    "agent.actor_hidden=256",
    "agent.critic_hidden=256",

    # ── World model loss ─────────────────────────────────────────────────────────
    # free_nats=1.0 is a TOTAL KL floor (max(KL_sum, 1)), NOT per-dimension.
    # kl_beta=0.5 gives obs reconstruction more weight relative to KL.
    # sigreg_weight=0.01 prevents SIGReg from fighting the posterior encoding info.
    "world_model.free_nats=1.0",
    "world_model.kl_beta=0.5",
    "world_model.sigreg_weight=0.01",
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
    for _n, method, overrides in configs:
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


# ── LOG TAILER ─────────────────────────────────────────────────────────────────

# Lines from train.py that are worth showing in the terminal.
_INTERESTING = ("[PROGRESS]", "[DONE]", "[ROLLBACK]", "[RESUME]", "[FAILED]")

def _log_tailer(log_path: Path, tag: str, stop: threading.Event) -> None:
    """Background thread: poll log_path and forward structured lines to stdout."""
    pos = 0
    while not stop.wait(timeout=5.0):   # check every 5 s; exits when stop is set
        try:
            text = log_path.read_text(errors="replace")
        except OSError:
            continue
        chunk = text[pos:]
        pos = len(text)
        for line in chunk.splitlines():
            if any(line.startswith(kw) for kw in _INTERESTING):
                print(f"  [{tag}] {line}", flush=True)
    # One final drain after the subprocess exits
    try:
        text = log_path.read_text(errors="replace")
        for line in text[pos:].splitlines():
            if any(line.startswith(kw) for kw in _INTERESTING):
                print(f"  [{tag}] {line}", flush=True)
    except OSError:
        pass


# ── RUNNER ─────────────────────────────────────────────────────────────────────

def _tag(exp: Experiment, gpu_id: Optional[int]) -> str:
    """Short one-line identifier: env | method | seed [GPU:N]."""
    gpu_str = f" [GPU:{gpu_id}]" if gpu_id is not None else ""
    return f"{exp.env_id} | {exp.method} | seed={exp.seed}{gpu_str}"


def run_experiment(exp: Experiment, dry_run: bool, manifest: dict[str, dict],
                   gpu_id: Optional[int] = None) -> bool:
    """Run a single experiment. Returns True if succeeded."""
    cmd = exp.build_cmd()
    tag = _tag(exp, gpu_id)

    if dry_run:
        print(f"[DRY RUN]  {tag}")
        print(f"           cmd: {' '.join(cmd)}")
        return True

    # Log file: runs/<exp_name>/train.log  (trainer creates the dir; we pre-create it here)
    log_dir = ROOT / "runs" / exp.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"

    upsert_manifest(manifest, exp, status="running",
                    start_time=datetime.now().isoformat())
    save_manifest(manifest)

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[START]  {tag}  →  log: runs/{exp.exp_name}/train.log")

    # Short tag for tailer prefix: "Hopper-v4|fixed_h5|s0"
    short = f"{exp.env_id}|{exp.method}|s{exp.seed}"
    stop_tailer = threading.Event()
    tailer = threading.Thread(
        target=_log_tailer, args=(log_path, short, stop_tailer), daemon=True
    )

    t0 = time.time()
    with open(log_path, "a") as log_f:
        tailer.start()
        try:
            subprocess.run(cmd, check=True, cwd=str(ROOT), env=env,
                           stdout=log_f, stderr=log_f)
            elapsed = time.time() - t0
            upsert_manifest(manifest, exp, status="done",
                            end_time=datetime.now().isoformat())
            save_manifest(manifest)
            stop_tailer.set()
            tailer.join(timeout=3)
            print(f"[DONE]   {tag}  ({elapsed/60:.1f} min)")
            return True
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - t0
            upsert_manifest(manifest, exp, status="failed",
                            end_time=datetime.now().isoformat())
            save_manifest(manifest)
            stop_tailer.set()
            tailer.join(timeout=3)
            print(f"[FAILED] {tag}  ({elapsed/60:.1f} min, rc={e.returncode})")
            try:
                lines = log_path.read_text().splitlines()
                tail = lines[-20:] if len(lines) > 20 else lines
                print(f"  --- last {len(tail)} lines of {log_path.name} ---")
                for line in tail:
                    print(f"  {line}")
            except Exception:
                pass
            return False
        except KeyboardInterrupt:
            # User pressed Ctrl+C — mark as pending so next run retries it
            elapsed = time.time() - t0
            upsert_manifest(manifest, exp, status="pending",
                            end_time=datetime.now().isoformat())
            save_manifest(manifest)
            stop_tailer.set()
            tailer.join(timeout=3)
            print(f"[INTERRUPTED] {tag}  ({elapsed/60:.1f} min) — marked pending for retry")
            raise  # propagate so the outer loop / pool can stop cleanly


def _worker(args):
    """Top-level picklable wrapper for ProcessPoolExecutor."""
    exp, dry_run, gpu_id = args
    # Each worker reloads the manifest from disk (avoids cross-process dict sharing)
    manifest = load_manifest()
    return run_experiment(exp, dry_run=dry_run, manifest=manifest, gpu_id=gpu_id)


def run_experiments_parallel(experiments: List[Experiment], dry_run: bool,
                              n_workers: int, n_gpus: int) -> tuple[int, int]:
    """Run experiments in parallel across n_workers processes, cycling over n_gpus GPUs."""
    tasks = [
        (exp, dry_run, (i % n_gpus) if n_gpus > 0 else None)
        for i, exp in enumerate(experiments)
    ]
    n_success = 0
    n_fail = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            exp = futures[fut]
            try:
                ok = fut.result()
            except Exception as exc:
                print(f"[ERROR]  {_tag(exp, None)}  —  {exc}")
                ok = False
            if ok:
                n_success += 1
            else:
                n_fail += 1
    return n_success, n_fail


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
        "--reset", action="store_true",
        help="Reset all experiments in the selected group back to 'pending' in the manifest, "
             "then run them. Use this when you want to re-run from scratch after deleting runs. "
             "Experiments outside the selected group are NOT affected.",
    )
    p.add_argument(
        "--list", action="store_true",
        help="List all experiments in the selected group and exit.",
    )
    p.add_argument(
        "--parallel", type=int, default=1, metavar="N",
        help="Run up to N experiments simultaneously (default: 1 = sequential). "
             "Use --gpus to specify how many GPUs to cycle across.",
    )
    p.add_argument(
        "--gpus", type=int, default=1, metavar="N",
        help="Number of GPUs to cycle CUDA_VISIBLE_DEVICES across when --parallel > 1 "
             "(default: 1). Set 0 to leave CUDA_VISIBLE_DEVICES unset.",
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

    # --reset: mark every experiment in the current group back to pending so they re-run.
    # Experiments from other groups remain untouched.
    if args.reset:
        reset_names = [exp.exp_name for exp in experiments]
        n_reset = sum(1 for n in reset_names if manifest.get(n, {}).get("status") in ("done", "failed"))
        if n_reset:
            print(f"[RESET] resetting {n_reset} done/failed experiment(s) to pending "
                  f"for group '{args.group}'.")
            for name in reset_names:
                if name in manifest:
                    manifest[name]["status"] = "pending"
        else:
            print(f"[RESET] no done/failed experiments found for group '{args.group}'; nothing to reset.")

    # Reset any "running" entries to "pending" — they were interrupted (crash / Ctrl+C)
    # and must be retried. "done" entries are never touched (unless --reset was passed above).
    stale = [name for name, row in manifest.items() if row.get("status") == "running"]
    if stale:
        print(f"[CLEANUP] {len(stale)} interrupted run(s) reset to pending: {', '.join(stale)}")
        for name in stale:
            manifest[name]["status"] = "pending"

    # Register all experiments in the manifest (status=pending if new)
    for exp in experiments:
        if exp.exp_name not in manifest:
            upsert_manifest(manifest, exp, status="pending")
    save_manifest(manifest)

    print(f"\n==> Running {len(experiments)} experiments (group={args.group}, "
          f"parallel={args.parallel}, gpus={args.gpus})")

    # Filter skip_done before dispatching
    pending = []
    n_skip = 0
    for exp in experiments:
        row = manifest.get(exp.exp_name, {})
        if args.skip_done and row.get("status") == "done":
            print(f"[SKIP]   {_tag(exp, None)}")
            n_skip += 1
        else:
            pending.append(exp)

    n_success = 0
    n_fail = 0
    try:
        if args.parallel > 1:
            n_success, n_fail = run_experiments_parallel(
                pending, dry_run=args.dry_run,
                n_workers=args.parallel, n_gpus=args.gpus,
            )
        else:
            for exp in pending:
                ok = run_experiment(exp, dry_run=args.dry_run, manifest=manifest)
                if ok:
                    n_success += 1
                else:
                    n_fail += 1
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Ctrl+C received — stopping. Interrupted runs were marked pending.")

    print(f"\n{'='*72}")
    print(f"Finished: {n_success} ok / {n_skip} skipped / {n_fail} failed"
          + (f" / {len(pending) - n_success - n_fail} pending (interrupted)"
             if n_success + n_fail < len(pending) else ""))
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
