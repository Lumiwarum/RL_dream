# ThesisWM — DreamerV3-style World Model RL with Adaptive Imagination Horizon

Master thesis project. Core idea: replace DreamerV3's fixed planning horizon with an
**ensemble-based adaptive horizon** — when the world model ensemble disagrees (high
uncertainty), use a short rollout (H=5, safer); when it agrees (low uncertainty), use
a long rollout (H=20, better credit assignment).

---

## ⚡ RIGHT NOW — Run the smoke test first

**Do this before anything else.** The smoke test runs 4 experiments (Hopper ×
{fixed\_h20, adaptive} × 2 seeds × 100k steps) to verify all recent fixes are
working before committing to 24 full runs.

```bash
# 1. Kill any currently running experiments (they used the OLD config)
#    Find the PIDs:
ps aux | grep train.py
#    Then: kill <pid> <pid> ...

# 2. Dry-run to see what will be launched
python scripts/run_experiments.py --group smoke --dry_run

# 3. Run (uses 4 parallel jobs, 4 GPUs)
python scripts/run_experiments.py --group smoke --parallel 4 --gpus 4

# 4. While running — check diagnostics every ~10 min
python scripts/analyze_runs.py --runs_dir runs/ --filter smoke

# 5. After ~100k steps — generate figures
python scripts/make_thesis_figures.py --runs_dir runs/ --filter smoke
```

### What to look for in the smoke report

The report (`report_<timestamp>.txt`) will show a **WM Health** and **AC Health**
section for each run. Here is what healthy vs broken looks like:

| Metric | Healthy ✓ | Broken ✗ | What it means |
|--------|-----------|----------|---------------|
| `obs_loss` | early ~1.2, **late < 0.3** | stays flat/rises | WM learning to predict observations |
| `kl_loss` | **> 1.0** at late training | ≈ 0.0 throughout | KL>0 = posterior encodes info |
| `policy_std` | **decreasing** from 1.0 | stuck at 1.0 | Actor learning (entropy fix working) |
| `return_scale` | **< 80**, stable | at 100 cap | Actor gradients alive |
| `actor_grad` | **0.01–0.15** | < 0.005 | Policy updating |
| `actor_loss` | **negative** (< 0) | positive (> 0) | Positive = negative advantage = bad bootstrap |
| `horizon_trend` | **early H=10, late H=20** (adaptive) | H=20 100% always | Adaptive mechanism working |

**A healthy smoke run should show by step 50k:**
- `obs_loss` dropped from ~1.2 → below 0.4
- `policy_std` dropped from 1.0 → below 0.85
- `actor_loss` negative for both methods
- No `SPIKE_THEN_DROP` flag
- Adaptive runs: `horizon_trend Δ > +3` (horizon shifting from H=10 toward H=20 as WM converges)

---

## Full experiments (run after smoke test passes)

```bash
# 24 runs: Hopper + Walker2d × {fixed_h5, fixed_h15, fixed_h20, adaptive} × 3 seeds
# Interleaved order: seed 0 of all methods starts first (fixed + adaptive together)
python scripts/run_experiments.py --group main --parallel 8 --gpus 8

# Extend existing 300k runs to 500k (total_steps bumped to 500k in SHARED_TRAINING_OVERRIDES)
# --skip_done skips truly finished runs; completed-at-300k runs will resume automatically
python scripts/run_experiments.py --group main --parallel 8 --gpus 8

# Resume after a crash (skips runs that already have a complete checkpoint)
python scripts/run_experiments.py --group main --parallel 8 --gpus 8 --skip_done

# Dry-run first to verify commands
python scripts/run_experiments.py --group main --dry_run
```

> **GPU memory**: ~2–4 GB VRAM per job (ensemble=2, rollout_batch=512, hidden=256).
> 8 jobs × 8 GPUs = 1 job per GPU.

---

## Monitoring and diagnostics

```bash
# ── Live status table (run repeatedly while training) ─────────────────────────
python scripts/scan_logs.py --runs_dir runs/

# ── Full diagnostic report with WM health + AC health + root-cause diagnosis ──
python scripts/analyze_runs.py --runs_dir runs/

# ── Filter to one env or group ─────────────────────────────────────────────────
python scripts/analyze_runs.py --runs_dir runs/ --filter hopper
python scripts/analyze_runs.py --runs_dir runs/ --filter adaptive

# ── TensorBoard ────────────────────────────────────────────────────────────────
tensorboard --logdir runs/ --port 8032

# ── Thesis figures + LaTeX tables ─────────────────────────────────────────────
python scripts/make_thesis_figures.py --runs_dir runs/
# Output: thesis/figures/*.pdf  and  thesis/tables/*.tex
```

---

## Diagnosing problems

### Root cause lookup

| Flag in report | Root cause | Fix |
|---|---|---|
| `SPIKE_THEN_DROP` | Old `entropy_coef=1e-2` (100x too high) | Verify `entropy_coef=3e-4` in SHARED_TRAINING_OVERRIDES |
| `ACTOR_GRAD_TINY` | `return_scale` at cap (100) | Check `symlog_clamp=5.0`, `return_scale_max=100` |
| `POLICY_FROZEN` | Actor not getting signal | Check `actor_start_step=8000`, check WM health |
| `POSTERIOR_COLLAPSE` | `kl_loss ≈ 0` | Check `free_nats`, check KL implementation |
| `WM_NOT_LEARNING` | `obs_loss` not decreasing | Try `wm.lr: 3e-4→1e-4`; check replay is populated |
| `HORIZON_STUCK` + H=20:100% | Uncertainty < thresh\_mid at all times | Check `actor_start_step=8000` so actor trains during high-unc phase |

### TensorBoard key tags

```
loss/obs          WM observation head: start ~1.2, should reach <0.3 by 30k steps
loss/kl           KL term: should rise above free_nats (1.0) quickly
loss/reward       Reward head: should reach <0.05 by 30k steps
policy/std_mean   Policy std: should decrease from 1.0 (proves actor is learning)
value/return_scale  Return scale: should stay below 80; at 100 = actor_grad dead
grad/actor_norm   Actor gradient: should be 0.02-0.15; <0.005 = nothing learning
imagine/horizon_used  For adaptive: should show H=5 early, H=20 late
```

---

## Ablations (run after main experiments)

```bash
python scripts/run_experiments.py --group ablation_metric    --parallel 8 --gpus 8
python scripts/run_experiments.py --group ablation_ensemble  --parallel 8 --gpus 8
python scripts/run_experiments.py --group ablation_threshold --parallel 8 --gpus 8
```

---

## Key hyperparameters (as of 2026-04-13)

| Parameter | Value | Why |
|---|---|---|
| `entropy_coef` | `3e-4` | DreamerV3 level; old 1e-2 caused SPIKE\_THEN\_DROP |
| `actor_lr` | `3e-5` | DreamerV3 reference; prevents tanh saturation |
| `critic_lr` | `3e-5` | Slower warmup prevents critic divergence |
| `symlog_clamp` | `5.0` | symexp(5)=147, limits bootstrap to ~100 |
| `return_scale_max` | `100.0` | Hard cap prevents actor\_grad going to zero |
| `actor_start_step` | `8000` | Trains while WM still uncertain → adaptive switching |
| `kl_beta` | `0.5` | Obs reconstruction gets more weight early |
| Adaptive horizons | `[10, 15, 20]` | H=5 confirmed broken (positive actor\_loss); min raised to H=10 |
| Hopper thresholds | `high=0.90, mid=0.55` | unc>0.90→H=10; unc>0.55→H=15; else→H=20 |
| Walker2d thresholds | `high=1.50, mid=0.80` | unc>1.50→H=10; unc>0.80→H=15; else→H=20 |

See `CHANGELOG.md` for full history of what changed and why.

---

## Code structure

```
train.py                          entry point
configs/config.yaml               base config (all defaults)
CHANGELOG.md                      history of all HP changes with evidence

thesiswm/
  models/rssm.py                  RSSM + EnsembleWorldModel
  agents/actor_critic.py          TanhGaussianPolicy + ValueNet
  training/
    trainer.py                    orchestrator: collection, eval, checkpointing
    world_model_updater.py        WM loss + optimizer step
    actor_critic_updater.py       imagination rollout + AC loss
    imagination.py                lambda_returns(), decide_horizon()

scripts/
  run_experiments.py              launch experiment groups (parallel, multi-GPU)
  scan_logs.py                    live status table (TB log scanner, cached)
  analyze_runs.py                 full diagnostic report (WM health, AC health, diagnosis)
  make_thesis_figures.py          training curves, horizon dist, LaTeX tables
  visualize_runs.py               learning curve plots

thesis/
  thesis_skeleton.tex             LaTeX skeleton
  figures/                        auto-generated by make_thesis_figures.py
  tables/                         auto-generated LaTeX table fragments
```

---

## Install

```bash
pip install torch gymnasium[mujoco] hydra-core omegaconf tensorboard imageio[ffmpeg] tqdm
export MUJOCO_GL=egl   # headless rendering on server
```
