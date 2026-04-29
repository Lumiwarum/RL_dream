"""env_scan.py — scan MuJoCo environments to find which converge well.

Usage
-----
# Dry-run: print commands without executing
python scripts/env_scan.py --group env_scan --dry_run

# List experiments in a group
python scripts/env_scan.py --group env_scan_1M --list

# Run the 500k scan sequentially
python scripts/env_scan.py --group env_scan

# Run the 1M scan in parallel across 2 GPUs
python scripts/env_scan.py --group env_scan_1M --parallel 2 --gpus 2

# Skip experiments already marked done
python scripts/env_scan.py --group env_scan --skip_done

Groups
------
  env_scan      -- 9 MuJoCo envs × baseline fixed_h15 × seed=0 × 500k steps
  env_scan_1M   -- harder 5 envs × baseline fixed_h15 × seed=0 × 1M steps
  env_scan_full -- 9 envs × fixed_h15 × seeds 0,1 × 500k (repeat with stats)
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent   # dreamer_impl/
MANIFEST = ROOT / "experiments" / "env_scan_manifest.csv"
FIELDS = ["exp_name", "group", "env_id", "method", "seed",
          "status", "start_time", "end_time", "cmd"]


@dataclass
class Experiment:
    group: str
    env_id: str
    method: str
    seed: int
    overrides: List[str] = field(default_factory=list)

    @property
    def exp_name(self) -> str:
        env = self.env_id.replace("-", "").lower()
        return f"scan_{env}_{self.method}_s{self.seed}"

    def cmd(self) -> List[str]:
        return [
            "python", "train.py",
            f"exp_name={self.exp_name}",
            f"env.id={self.env_id}",
            f"seed={self.seed}",
        ] + self.overrides


# ── Shared overrides ──────────────────────────────────────────────────────────
_BASE = [
    "env.num_envs=8",
    "compile=false",
]

_FIXED_H15 = [
    "adaptive.enabled=false",
    "model.imag_horizon=15",
    "model.ensemble_size=1",
    "model.ensemble_avg_rewards=false",
]

# ── Environment list ──────────────────────────────────────────────────────────
# Ordered roughly by expected difficulty (easiest first).
# InvertedPendulum, InvertedDoublePendulum, Reacher: very fast convergence.
# Swimmer: no termination → easy horizon test.
# Hopper, Walker2d: standard DreamerV3 benchmarks.
# HalfCheetah: dense reward but high-dim; usually converges well.
# Ant: high-dim obs+act; needs more steps.
# Humanoid: very high-dim; 1M is borderline.
ENVS_SIMPLE = [
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v4",
    "Reacher-v4",
    "Swimmer-v4",
]

ENVS_MEDIUM = [
    "Hopper-v4",
    "Walker2d-v4",
    "HalfCheetah-v4",
]

ENVS_HARD = [
    "Ant-v4",
    "Humanoid-v4",
]

ENVS_ALL = ENVS_SIMPLE + ENVS_MEDIUM + ENVS_HARD


def _make(group, env_id, seed, method, overrides, extra=None):
    return Experiment(
        group=group,
        env_id=env_id,
        method=method,
        seed=seed,
        overrides=_BASE + overrides + (extra or []),
    )


# ── Experiment groups ─────────────────────────────────────────────────────────
EXPERIMENT_GROUPS: dict[str, list[Experiment]] = {

    # 500k steps, all 9 envs, baseline fixed_h15, single seed
    "env_scan": [
        _make("env_scan", env, 0, "fixed_h15", _FIXED_H15,
              ["trainer.steps=500000"])
        for env in ENVS_ALL
    ],

    # 1M steps, harder envs only — run after env_scan to confirm early trends
    "env_scan_1M": [
        _make("env_scan_1M", env, 0, "fixed_h15", _FIXED_H15,
              ["trainer.steps=1000000"])
        for env in ENVS_MEDIUM + ENVS_HARD
    ],

    # 2 seeds × all envs × 500k — after identifying promising envs, get statistics
    "env_scan_full": [
        _make("env_scan_full", env, seed, "fixed_h15", _FIXED_H15,
              ["trainer.steps=500000"])
        for env in ENVS_ALL
        for seed in [0, 1]
    ],
}


# ── Manifest helpers ──────────────────────────────────────────────────────────

def load_manifest() -> dict:
    if not MANIFEST.exists():
        return {}
    with open(MANIFEST, newline="") as f:
        return {r["exp_name"]: r for r in csv.DictReader(f)}


def save_manifest(rows: dict):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows.values():
            w.writerow(r)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_one(exp: Experiment, dry_run: bool, gpu_id: int | None = None) -> bool:
    cmd = exp.cmd()
    if dry_run:
        print(f"[DRY RUN] {' '.join(cmd)}")
        return True
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log = ROOT / "runs" / exp.exp_name / "train.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[START] {exp.exp_name}  (GPU {gpu_id})")
    t0 = datetime.now()
    with open(log, "a") as f:
        ret = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=f)
    elapsed = (datetime.now() - t0).seconds / 60
    ok = ret.returncode == 0
    status = "[DONE]" if ok else "[FAIL]"
    print(f"{status} {exp.exp_name} ({elapsed:.1f} min)")
    return ok


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Scan MuJoCo environments with baseline DreamerV3.")
    p.add_argument("--group", choices=list(EXPERIMENT_GROUPS), default="env_scan",
                   help="Which experiment group to run (default: env_scan)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--skip_done", action="store_true",
                   help="Skip experiments already marked done in manifest")
    p.add_argument("--parallel", type=int, default=1,
                   help="Number of experiments to run in parallel")
    p.add_argument("--gpus", type=int, default=1,
                   help="Number of GPUs to round-robin across (used with --parallel)")
    p.add_argument("--list", action="store_true",
                   help="List experiment names in the group and exit")
    args = p.parse_args()

    experiments = EXPERIMENT_GROUPS[args.group]

    if args.list:
        print(f"\nGroup '{args.group}' — {len(experiments)} experiments:\n")
        for e in experiments:
            print(f"  {e.exp_name:<55} {e.env_id}")
        return

    manifest = load_manifest()

    # Register new experiments in manifest
    for e in experiments:
        if e.exp_name not in manifest:
            manifest[e.exp_name] = {
                "exp_name": e.exp_name, "group": e.group,
                "env_id": e.env_id, "method": e.method, "seed": e.seed,
                "status": "pending", "start_time": "", "end_time": "",
                "cmd": " ".join(e.cmd()),
            }
    save_manifest(manifest)

    if args.skip_done:
        experiments = [e for e in experiments
                       if manifest.get(e.exp_name, {}).get("status") != "done"]
        print(f"Skipping done experiments — {len(experiments)} remaining.")

    if not experiments:
        print("Nothing to run.")
        return

    if args.parallel > 1:
        tasks = [(e, args.dry_run, i % args.gpus) for i, e in enumerate(experiments)]
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futs = {pool.submit(run_one, *t): t[0] for t in tasks}
            for fut in as_completed(futs):
                e = futs[fut]
                ok = fut.result()
                manifest[e.exp_name]["status"] = "done" if ok else "failed"
                manifest[e.exp_name]["end_time"] = datetime.now().isoformat()
                save_manifest(manifest)
    else:
        for e in experiments:
            manifest[e.exp_name]["status"] = "running"
            manifest[e.exp_name]["start_time"] = datetime.now().isoformat()
            save_manifest(manifest)
            ok = run_one(e, args.dry_run)
            manifest[e.exp_name]["status"] = "done" if ok else "failed"
            manifest[e.exp_name]["end_time"] = datetime.now().isoformat()
            save_manifest(manifest)


if __name__ == "__main__":
    main()
