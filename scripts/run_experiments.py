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
  ablation_ensemble -- Hopper × ensemble size {N=1 fixed_h20, N=2 fixed_h20, N=2 adaptive} × 3 seeds
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
    method: str           # fixed_h5 | fixed_h15 | fixed_h20 | adaptive_10_15_20
    seed: int
    extra_overrides: List[str]  # additional Hydra overrides
    base_config: str = "config"

    @property
    def exp_name(self) -> str:
        """Unique name used for run_dir and manifest."""
        env_tag = self.env_id.replace("-", "").lower()
        # Only include overrides that add information not already in the method name.
        # horizon_fixed is redundant (method=fixed_h15 already encodes it); skip it.
        # metric/ensemble/thresh are ablation-specific and always included.
        overrides_tag = "_".join(
            o.replace("=", "-").replace(".", "_").replace("/", "-")
            for o in self.extra_overrides
            if any(k in o for k in ["metric", "ensemble", "thresh"])
        )
        tag = f"{env_tag}_{self.method}"
        if overrides_tag:
            tag += f"_{overrides_tag}"
        tag += f"_s{self.seed}"   # shorter: s0 not seed0
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
    # ── Uncertainty signal: EMA WM obs_loss (use_obs_loss_unc=true) ─────────────
    # The original latent_mean_l2 ensemble-disagreement signal settles to a STABLE
    # per-seed value within ~500 WM updates (before actor_start_step) → each seed
    # locks to exactly one horizon for the whole run. No within-run switching.
    #
    # EMA obs_loss (wm/ema_obs_loss in TB) varies throughout training:
    #   High early (WM still learning) → gradually falls as WM converges
    #   → gives genuine within-run switching as WM quality improves.
    #   Initialised at 2.0 so the first horizon is always H=10 (cautious start).
    #
    # ── Hopper thresholds (obs_loss range: early≈0.20–0.28, late≈0.15–0.20) ────
    #   obs_loss > 0.28 → H=10  (WM struggling / policy exploring new states)
    #   obs_loss > 0.17 → H=15  (WM mostly converged, ~steps 50k–400k)
    #   obs_loss < 0.17 → H=20  (WM converged, ~steps 400k+)
    #   WHY 0.28 not 0.22: true Hopper obs_loss is 0.19–0.25. With thresh=0.22,
    #   the EMA (initialized to 2.0) converges below the threshold within ~215 WM
    #   updates (well before actor_start=8k), giving essentially NO H=10 phase
    #   during actual actor training. thresh=0.28 ensures H=10 is selected whenever
    #   the policy is actively exploring (obs_loss bumps to 0.22–0.26), giving a
    #   genuine 3-phase H=10→H=15→H=20 schedule across a 500k run.
    #
    # ── Walker2d thresholds (obs_loss range: early≈2.0–2.5, late≈1.1–1.7) ──────
    #   obs_loss > 1.50 → H=10  (WM inaccurate, ~steps 8k–50k with actor_start=8k)
    #   obs_loss > 1.00 → H=15  (WM improving, ~steps 50k–500k)
    #   obs_loss < 1.00 → H=20  (rarely reached at 500k; Walker2d WM still converging)
    "Hopper-v4": [
        "method.uncertainty_metric=latent_mean_l2",  # kept for fallback/logging
        "method.thresh_high=0.90", "method.thresh_mid=0.55",
        "imagination.horizons=[10,15,20]",
        "method.use_obs_loss_unc=true",
        "method.obs_loss_thresh_high=0.28",
        "method.obs_loss_thresh_mid=0.17",
    ],
    "Walker2d-v4": [
        "method.uncertainty_metric=latent_mean_l2",
        "method.thresh_high=1.50", "method.thresh_mid=0.80",
        "imagination.horizons=[10,15,20]",
        "method.use_obs_loss_unc=true",
        "method.obs_loss_thresh_high=1.50",
        "method.obs_loss_thresh_mid=1.00",
    ],
}

SHARED_TRAINING_OVERRIDES = [
    # ── Steps ───────────────────────────────────────────────────────────────────
    # 500k: extended from 300k after pilot. At 300k: adaptive > all fixed on both envs.
    # Hopper adaptive s0/s1 still trending up at 300k; Walker2d WM still converging.
    # Existing runs at 300k will continue for another 200k when relaunched.
    "training.total_steps=500000",
    "training.steps_per_chunk=500000",  # must match total_steps (single-subprocess design)

    # ── Env collection ──────────────────────────────────────────────────────────
    "env.num_envs=8",

    # ── Learning schedule ───────────────────────────────────────────────────────
    "training.start_learning=5000",
    # actor_start_step: start actor early so it trains while WM uncertainty is still
    # declining (critical window for testing the adaptive hypothesis).
    # WM uncertainty starts ~11 (random init) and drops to steady-state 0.5-1.8
    # over the first 10-15k steps. With actor_start=5k, the actor sees high-unc
    # early training where H=5 should be selected, then progressively transitions
    # to H=10/H=20 as WM improves — this IS the thesis hypothesis.
    # At step 5k the WM has had 0 gradient steps (start_learning=5k too), so we
    # use critic warmup (steps 5k→8k) to stabilise before actor begins.
    # Note: entropy_coef=3e-4 prevents the early instability that previously
    # required delaying the actor to step 20k.
    "training.actor_start_step=8000",
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

    # ── Actor-critic ─────────────────────────────────────────────────────────────
    # actor_lr=3e-5, critic_lr=3e-5: DreamerV3 reference values.
    #   Lower LR prevents rapid drift to tanh saturation boundary (actor) and
    #   prevents fast critic divergence during warmup that drives SPIKE_THEN_DROP.
    # entropy_coef=3e-4: DreamerV3 level (~1e-4 for their tasks).
    #   Old value 1e-2 was 30x too high — entropy dominated actor loss early,
    #   freezing policy_std≈1.0, then causing sudden collapse when return gradient
    #   grew large enough to override entropy (SPIKE_THEN_DROP in 13/24 runs).
    #   3e-4 uses analytic Gaussian entropy (always positive, never penalises exploration).
    # pretanh_reg_coef=1e-3: L2 penalty on raw pre-tanh MLP output prevents actor
    #   drifting to tanh saturation boundary.
    "agent.actor_lr=3e-5",
    "agent.critic_lr=3e-5",
    "agent.entropy_coef=3e-4",
    "agent.pretanh_reg_coef=1e-3",

    # ── Rollback ─────────────────────────────────────────────────────────────────
    # train_rollback_check is DISABLED: episode-return variance (~20 sigma with early policy)
    # triggers bad_streak every 2-3 episodes → hundreds of rollbacks in the first 50k steps
    # → cont predictor never converges → discounts stay ≈0.5 → actor can't learn.
    #
    # Eval-based rollback IS enabled: eval fires every 10k steps (not noisy), rollback_drop=15
    # requires a 15-point drop from best eval for rollback_patience=4 consecutive evals
    # (= 40k steps of sustained degradation) before triggering. The rollback_cooldown_steps=40000
    # guard in trainer.py prevents re-triggering within 40k steps, avoiding thrashing.
    # This recovers SPIKE_THEN_DROP runs (e.g., Walker2d fixed_h20 peak=206→-1) where the
    # best checkpoint is saved but the policy degrades as WM continues to update.
    "training.train_rollback_check=false",
    "training.rollback_drop=15",         # fire if eval drops 15 from best for 4 consecutive evals
    "training.rollback_patience=4",
    "training.rollback_cooldown_steps=40000",  # no re-rollback within 40k steps
    "training.save_best_train=false",    # no best_train.pt — nothing to roll back to
    # WM freeze after rollback: after restoring best.pt, suppress WM gradient updates
    # for 20k steps. Without this, the WM is immediately overwritten by current replay
    # data (~1000 gradient updates) → the restored actor/critic become invalid again
    # → rollback appears to do nothing. 20k steps ≈ 5 eval cycles; matches rollback_cooldown.
    "training.wm_freeze_after_rollback=20000",
]


# Walker2d has 17-dim obs and more complex dynamics than Hopper (11-dim).
# WM obs_loss at the default actor_start_step=8k is still ~2.0 (random-init WM).
# The actor finds a local gait in this inaccurate WM, peaks at eval=111-219, then
# collapses as the WM improves and the learned gait stops working (SPIKE_THEN_DROP).
#
# Fixed methods (fixed_h20): must delay to 50k so the WM is partially trained before
# the actor begins. Using H=20 on obs_loss≈2.0 is dangerous — long bootstrap chains
# amplify WM errors → SPIKE_THEN_DROP.
#
# Adaptive method: actor_start=8k (same as Hopper). At step 8k the WM has had ~3k
# gradient updates and obs_loss is still ~2.5 >> obs_loss_thresh_high=1.50 → H=10 is
# selected. This tests the thesis hypothesis: "adaptive H=10 early (when WM is bad)
# enables earlier, safer actor training than fixed H=20."
# NOTE: actor_start=15k was tried but with ~40k WM gradient steps the obs_loss
# already drops to ~1.4 < thresh_high=1.5 → H=15 immediately, no H=10 phase ever.
# actor_start=8k keeps obs_loss above the threshold → genuine H=10→H=15→H=20 schedule.
WALKER2D_FIXED_ACTOR_OVERRIDE    = ["training.actor_start_step=50000"]
WALKER2D_ADAPTIVE_ACTOR_OVERRIDE = ["training.actor_start_step=8000"]


def make_main_experiments() -> List[Experiment]:
    # Interleave methods and seeds so that with --parallel N, the first N jobs
    # include a mix of fixed AND adaptive runs (not all fixed first).
    # Order: seed first, then method — so seed=0 of all methods starts together.
    exps = []
    all_variants = list(FIXED_VARIANTS.items()) + [
        ("adaptive_10_15_20", None)  # overrides come from ADAPTIVE_OVERRIDES_PER_ENV
    ]
    for seed in SEEDS:
        for env in ENVS_MAIN:
            for method, method_overrides in all_variants:
                # Walker2d: fixed methods wait 50k for WM to stabilise; adaptive can
                # start at 15k because high uncertainty → H=10 (safe on bad WM).
                if env == "Walker2d-v4":
                    env_extra = (WALKER2D_ADAPTIVE_ACTOR_OVERRIDE
                                 if method == "adaptive_10_15_20"
                                 else WALKER2D_FIXED_ACTOR_OVERRIDE)
                else:
                    env_extra = []

                if method == "adaptive_10_15_20":
                    overrides = ADAPTIVE_OVERRIDES_PER_ENV[env] + SHARED_TRAINING_OVERRIDES + env_extra
                else:
                    overrides = method_overrides + SHARED_TRAINING_OVERRIDES + env_extra
                exps.append(Experiment(
                    group="main",
                    env_id=env,
                    method=method,
                    seed=seed,
                    extra_overrides=overrides,
                ))
    return exps


# ── SMOKE TEST GROUP ───────────────────────────────────────────────────────────
# Fast sanity check: Hopper × {fixed_h20, adaptive} × 2 seeds × 100k steps.
# Use this BEFORE running the full main group to verify:
#   1. WM loss (obs/rew/kl) is trending down (WM learning)
#   2. policy_std decreases from 1.0 (entropy fix working, no SPIKE_THEN_DROP)
#   3. adaptive shows horizon_trend ↑ early→late (H=10 early, H=20 late as WM converges)
# 4 runs, ~30 min on a single GPU (Hopper is fast).

SMOKE_OVERRIDES = [
    o for o in SHARED_TRAINING_OVERRIDES
    if not o.startswith("training.total_steps")
    and not o.startswith("training.steps_per_chunk")
    and not o.startswith("training.eval_every_steps")
    and not o.startswith("training.log_every_steps")
] + [
    "training.total_steps=100000",
    "training.steps_per_chunk=100000",
    "training.eval_every_steps=2000",   # eval every 2k for tight feedback loop
    "training.log_every_steps=500",     # log every 500 for dense diagnostics
]


def make_smoke_experiments() -> List[Experiment]:
    """4 runs: Hopper × {fixed_h20, adaptive} × seeds {0, 1}."""
    exps = []
    smoke_seeds = [0, 1]
    smoke_methods = [
        ("fixed_h20",      ["imagination.horizon_fixed=20"]),
        ("adaptive_10_15_20", None),
    ]
    for seed in smoke_seeds:
        for method, method_overrides in smoke_methods:
            if method == "adaptive_10_15_20":
                overrides = ADAPTIVE_OVERRIDES_PER_ENV["Hopper-v4"] + SMOKE_OVERRIDES
            else:
                overrides = method_overrides + SMOKE_OVERRIDES
            exps.append(Experiment(
                group="smoke",
                env_id="Hopper-v4",
                method=method,
                seed=seed,
                extra_overrides=overrides,
            ))
    return exps


# ── ABLATION: UNCERTAINTY METRIC ───────────────────────────────────────────────

UNCERTAINTY_METRICS = ["latent_mean_l2", "next_obs_mean_l2", "gaussian_kl"]

# Per-metric thresholds calibrated to each metric's natural scale on Hopper.
# Using identical thresholds across metrics would confound the ablation by
# effectively selecting different horizons for different metrics unintentionally.
#   latent_mean_l2:  z-space (N(0,I) prior), disagreement ≈ O(0.5–5)
#   next_obs_mean_l2: raw obs space (Hopper=11-dim, values O(1–5)), disagreement ≈ O(1–10)
#   gaussian_kl:      symmetric KL between posteriors, ≈ O(0.1–3)
METRIC_THRESHOLDS = {
    "latent_mean_l2":   ("3.0", "1.0"),
    "next_obs_mean_l2": ("6.0", "2.0"),
    "gaussian_kl":      ("1.5", "0.5"),
}

def make_ablation_metric_experiments() -> List[Experiment]:
    exps = []
    for metric in UNCERTAINTY_METRICS:
        th_high, th_mid = METRIC_THRESHOLDS[metric]
        for seed in SEEDS:
            exps.append(Experiment(
                group="ablation_metric",
                env_id="Hopper-v4",
                method="adaptive_10_15_20",
                seed=seed,
                extra_overrides=[
                    f"method.uncertainty_metric={metric}",
                    f"method.thresh_high={th_high}",
                    f"method.thresh_mid={th_mid}",
                    "imagination.horizons=[10,15,20]",
                ] + SHARED_TRAINING_OVERRIDES,
            ))
    return exps


# ── ABLATION: ENSEMBLE SIZE ────────────────────────────────────────────────────
# Tests whether the ensemble WM (N=2) improves over a single WM (N=1).
# N=1 with adaptive is meaningless (no disagreement signal, and obs_loss signal
# works the same regardless), so we use fixed horizons for the N comparison:
#   - N=1, Fixed-H20  (single WM; same horizon as our best fixed baseline)
#   - N=2, Fixed-H20  (ensemble WM; same horizon — isolates the ensemble effect)
#   - N=2, Adaptive   (our full method: ensemble + obs_loss adaptive horizon)
# This gives the thesis two clean claims:
#   (1) N=2 Fixed vs N=1 Fixed → does the ensemble WM alone improve stability/performance?
#   (2) N=2 Adaptive vs N=2 Fixed → does the adaptive horizon add on top of the ensemble?
# Uses Hopper-v4 (fast, 3 seeds, 500k steps) matching the main group setup.

def make_ablation_ensemble_experiments() -> List[Experiment]:
    # Adaptive overrides use the same obs_loss signal as the main group (Hopper thresholds)
    _adaptive_ov = [
        "method.uncertainty_metric=latent_mean_l2",
        "method.thresh_high=0.90", "method.thresh_mid=0.55",
        "imagination.horizons=[10,15,20]",
        "method.use_obs_loss_unc=true",
        "method.obs_loss_thresh_high=0.28",
        "method.obs_loss_thresh_mid=0.17",
        "world_model.ensemble_size=2",
    ]
    configs = [
        (1, "fixed_h20",      ["imagination.horizon_fixed=20", "world_model.ensemble_size=1"]),
        (2, "fixed_h20",      ["imagination.horizon_fixed=20", "world_model.ensemble_size=2"]),
        (2, "adaptive_10_15_20", _adaptive_ov),
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
# Vary (thresh_high, thresh_mid) for Adaptive on Hopper using latent_mean_l2.
# latent_mean_l2 operates in z-space (N(0,I)), typical disagreement ≈ O(0.5–5).
# Previous values (0.20–0.50) were calibrated for next_obs_mean_l2 and would
# collapse all adaptive runs to H_max when used with latent_mean_l2.
THRESHOLD_PAIRS = [
    (1.5, 0.5),    # tight — selects H_low more often
    (3.0, 1.0),    # default / calibrated
    (4.5, 1.5),    # loose
    (6.0, 2.0),    # very loose — H_high most of the time
]

def make_ablation_threshold_experiments() -> List[Experiment]:
    exps = []
    for (th, tm) in THRESHOLD_PAIRS:
        for seed in SEEDS:
            exps.append(Experiment(
                group="ablation_threshold",
                env_id="Hopper-v4",
                method="adaptive_10_15_20",
                seed=seed,
                extra_overrides=[
                    "method.uncertainty_metric=latent_mean_l2",
                    f"method.thresh_high={th}",
                    f"method.thresh_mid={tm}",
                    "imagination.horizons=[10,15,20]",
                ] + SHARED_TRAINING_OVERRIDES,
            ))
    return exps


# ── REGISTRY ───────────────────────────────────────────────────────────────────

EXPERIMENT_GROUPS = {
    "smoke":              make_smoke_experiments,       # quick 4-run sanity check
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
        # Strip NUL bytes produced by partial writes (process killed mid-save).
        reader = csv.DictReader(line.replace("\x00", "") for line in f)
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
