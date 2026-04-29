"""Experiment launcher.

Usage:
    python scripts/run_experiments.py --group smoke --dry_run
    python scripts/run_experiments.py --group main --parallel 4 --gpus 4
    python scripts/run_experiments.py --group control --skip_done
    python scripts/run_experiments.py --group overnight_three_envs --parallel 6 --gpus 2
    python scripts/run_experiments.py --group smoke --list
"""
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
MANIFEST = ROOT / "experiments" / "manifest.csv"


@dataclass
class Experiment:
    group: str
    env_id: str
    method: str
    seed: int
    overrides: List[str] = field(default_factory=list)

    @property
    def exp_name(self):
        env = self.env_id.split("-")[0].lower()
        return f"{env}_{self.method}_s{self.seed}"

    def cmd(self):
        overrides = []
        seen = set()
        for item in reversed(self.overrides):
            key = item.split("=", 1)[0]
            if key in seen:
                continue
            seen.add(key)
            overrides.append(item)
        overrides.reverse()
        return [
            "python", "train.py",
            f"exp_name={self.exp_name}",
            f"env.id={self.env_id}",
            f"seed={self.seed}",
        ] + overrides


# ── Shared overrides ──────────────────────────────────────────────────────────
BASE = [
    "trainer.steps=500000",
    "env.num_envs=8",
]

FIXED_H10      = ["adaptive.enabled=false", "model.imag_horizon=10",  "model.ensemble_size=1"]
FIXED_H15      = ["adaptive.enabled=false", "model.imag_horizon=15",  "model.ensemble_size=1"]
FIXED_H15_ENS  = ["adaptive.enabled=false", "model.imag_horizon=15",  "model.ensemble_size=2", "model.ensemble_avg_rewards=true"]
FIXED_H20      = ["adaptive.enabled=false", "model.imag_horizon=20",  "model.ensemble_size=1"]
FIXED_H20_ENS  = ["adaptive.enabled=false", "model.imag_horizon=20",  "model.ensemble_size=2", "model.ensemble_avg_rewards=true"]

ADAPTIVE_N1    = ["adaptive.enabled=true",  "model.ensemble_size=1"]
ADAPTIVE_N2    = ["adaptive.enabled=true",  "model.ensemble_size=2", "model.ensemble_avg_rewards=true"]

HOPPER_THRESH   = ["adaptive.thresh_high=0.27", "adaptive.thresh_mid=0.20"]
WALKER_THRESH   = ["adaptive.thresh_high=1.80", "adaptive.thresh_mid=1.30"]
PENDULUM_THRESH = ["adaptive.thresh_high=0.30", "adaptive.thresh_mid=0.08"]
DOUBLEPEND_THRESH = ["adaptive.thresh_high=0.30", "adaptive.thresh_mid=0.08"]
HALFCHEETAH_THRESH = ["adaptive.thresh_high=1.00", "adaptive.thresh_mid=0.50"]
ANT_THRESH = ["adaptive.thresh_high=2.50", "adaptive.thresh_mid=1.50"]

_ENV_THRESH = {
    "Hopper-v4":                   HOPPER_THRESH,
    "Walker2d-v4":                 WALKER_THRESH,
    "InvertedPendulum-v5":         PENDULUM_THRESH,
    "InvertedDoublePendulum-v4":   DOUBLEPEND_THRESH,
    "HalfCheetah-v4":              HALFCHEETAH_THRESH,
    "Ant-v4":                      ANT_THRESH,
}


def make_group(group, env_id, seeds, method, base_overrides, extra=None):
    thresholds = _ENV_THRESH.get(env_id, [])
    overrides = BASE + base_overrides + thresholds + (extra or [])
    return [
        Experiment(group=group, env_id=env_id, method=method, seed=s, overrides=overrides)
        for s in seeds
    ]


SEEDS_FULL  = [0, 1, 2]
SEEDS_SMOKE = [0, 1]
THREE_ENVS = ["InvertedDoublePendulum-v4", "Ant-v4", "HalfCheetah-v4"]


def make_control_suite(group, envs, seeds, steps=500000):
    """2x2 thesis comparison: fixed/adaptive horizon x single/ensemble WM."""
    experiments = []
    for env_id in envs:
        extra = ["trainer.steps=300000"] if "InvertedDoublePendulum" in env_id else [f"trainer.steps={steps}"]
        experiments += make_group(group, env_id, seeds, "fixed_h15_n1", FIXED_H15, extra)
        experiments += make_group(group, env_id, seeds, "fixed_h15_ens", FIXED_H15_ENS, extra)
        experiments += make_group(group, env_id, seeds, "adaptive_n1", ADAPTIVE_N1, extra)
        experiments += make_group(group, env_id, seeds, "adaptive_ens", ADAPTIVE_N2, extra)
    return experiments

EXPERIMENT_GROUPS = {
    "smoke": (
        make_group("smoke", "InvertedPendulum-v5", [0], "fixed_h15", FIXED_H15,
                   ["trainer.steps=100000"]) +
        make_group("smoke", "InvertedPendulum-v5", [0], "adaptive_n1", ADAPTIVE_N1,
                   ["trainer.steps=100000"])
    ),
    "main": (
        make_group("main", "Hopper-v4", SEEDS_FULL, "fixed_h10",   FIXED_H10) +
        make_group("main", "Hopper-v4", SEEDS_FULL, "fixed_h15",   FIXED_H15) +
        make_group("main", "Hopper-v4", SEEDS_FULL, "fixed_h20",   FIXED_H20) +
        make_group("main", "Hopper-v4", SEEDS_FULL, "adaptive_n1", ADAPTIVE_N1)
    ),
    "control": (
        make_group("control", "Hopper-v4", SEEDS_SMOKE, "fixed_h20_n1",  FIXED_H20) +
        make_group("control", "Hopper-v4", SEEDS_SMOKE, "fixed_h20_ens", FIXED_H20_ENS) +
        make_group("control", "Hopper-v4", SEEDS_SMOKE, "adaptive_n1",   ADAPTIVE_N1) +
        make_group("control", "Hopper-v4", SEEDS_SMOKE, "adaptive_ens",  ADAPTIVE_N2)
    ),
    "control_full": (
        make_group("control_full", "Hopper-v4", SEEDS_FULL, "fixed_h20_n1",  FIXED_H20) +
        make_group("control_full", "Hopper-v4", SEEDS_FULL, "fixed_h20_ens", FIXED_H20_ENS) +
        make_group("control_full", "Hopper-v4", SEEDS_FULL, "adaptive_n1",   ADAPTIVE_N1) +
        make_group("control_full", "Hopper-v4", SEEDS_FULL, "adaptive_ens",  ADAPTIVE_N2) +
        make_group("control_full", "InvertedPendulum-v5", SEEDS_SMOKE, "fixed_h20_n1", FIXED_H20,
                   ["trainer.steps=300000"]) +
        make_group("control_full", "InvertedPendulum-v5", SEEDS_SMOKE, "adaptive_ens", ADAPTIVE_N2,
                   ["trainer.steps=300000"])
    ),
    # 3 envs x 4 variants x 3 seeds = 36 runs.
    # Variants: baseline, ensemble-only, adaptive-only, adaptive+ensemble.
    "overnight_three_envs": make_control_suite(
        "overnight_three_envs", THREE_ENVS, SEEDS_FULL, steps=50000
    ),
    # Smaller version for testing the launcher: 3 envs x 4 variants x 1 seed.
    "overnight_three_envs_s0": make_control_suite(
        "overnight_three_envs_s0", THREE_ENVS, [0], steps=500000
    ),
}

# ── Manifest ──────────────────────────────────────────────────────────────────
FIELDS = ["exp_name", "group", "env_id", "method", "seed", "status",
          "start_time", "end_time", "cmd"]


def load_manifest():
    if not MANIFEST.exists():
        return {}
    with open(MANIFEST, newline="") as f:
        return {r["exp_name"]: r for r in csv.DictReader(f)}


def save_manifest(rows):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows.values():
            w.writerow(r)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_one(exp, dry_run, gpu_id=None):
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
    ok = (ret.returncode == 0)
    print(f"{'[DONE]' if ok else '[FAIL]'} {exp.exp_name} ({elapsed:.1f} min)")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=list(EXPERIMENT_GROUPS), default="smoke")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_done", action="store_true")
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    experiments = EXPERIMENT_GROUPS[args.group]

    if args.list:
        for e in experiments:
            print(f"{e.exp_name:<60} {e.group}")
        return

    manifest = load_manifest()
    if args.skip_done:
        experiments = [e for e in experiments
                       if manifest.get(e.exp_name, {}).get("status") != "done"]

    for e in experiments:
        if e.exp_name not in manifest:
            manifest[e.exp_name] = {
                "exp_name": e.exp_name, "group": e.group, "env_id": e.env_id,
                "method": e.method, "seed": e.seed, "status": "pending",
                "start_time": "", "end_time": "", "cmd": " ".join(e.cmd()),
            }
    save_manifest(manifest)

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
