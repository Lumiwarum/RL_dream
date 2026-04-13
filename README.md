# ThesisWM — DreamerV3-style World Model RL with Adaptive Imagination Horizon

Master thesis project. Core idea: replace DreamerV3's fixed planning horizon with an
**ensemble-based adaptive horizon** — when the world model ensemble disagrees (high
uncertainty), use a short rollout (H=5, safer); when it agrees (low uncertainty), use
a long rollout (H=20, better credit assignment).

---

## Current phase: Baseline validation

**Goal**: confirm that `fixed_h20` (the DreamerV3 baseline) learns stably on Hopper and
Walker2d before running ablations. Only move to the adaptive comparison after the
baseline curves look clean.

**Success criteria for a healthy baseline run:**
- `eval/return_mean` rises consistently (Hopper: >1000 by 200k steps, Walker2d: >800 by 300k)
- `loss/actor` stays in `[0.001, 2.0]` — never NaN
- `value/return_scale` stays in `[1, 50]` — never jumps to hundreds
- `imagine/cont_prob_mean` stays in `[0.7, 0.99]` — never collapses to 0
- No `VALUE_DIVERGE` or `CONT_LOW` flags in scan_logs

---

## Quick reference

### 1. Run baseline experiments (8 GPUs)

```bash
# Dry-run first — prints all commands without executing
python scripts/run_experiments.py --group main --dry_run

# Run main comparison: Hopper + Walker2d × {fixed_h5, fixed_h15, fixed_h20, adaptive} × 3 seeds
# 24 experiments total; 8 GPUs, 8 parallel jobs
python scripts/run_experiments.py --group main --parallel 8 --gpus 8

# Resume after a crash (skips already-done runs)
python scripts/run_experiments.py --group main --parallel 8 --gpus 8 --skip_done

# Reset and re-run from scratch (wipes manifest status, keeps run dirs)
python scripts/run_experiments.py --group main --parallel 8 --gpus 8 --reset
```

> **GPU memory**: each job uses ~2–4 GB VRAM (ensemble=2, rollout_batch=512, hidden=256).
> 8 jobs across 8 GPUs = 1 job per GPU, safe on any 10+ GB card.

### 2. Monitor live

```bash
# Compact per-run status table (run this repeatedly while training)
python scripts/scan_logs.py

# TensorBoard
tensorboard --logdir runs --port 8032
```

### 3. Run diagnostics on a checkpoint

```bash
# Full health report: WM forward pass, AC stats, gradient norms, verdict
python scripts/diagnose.py --run_dir runs/<exp_name>

# Specific checkpoint (default: latest.pt)
python scripts/diagnose.py --run_dir runs/<exp_name> --checkpoint best.pt
```

### 4. Watch learning curves

```bash
# PNG plots grouped by method, saved to plots/
python scripts/visualize_runs.py --runs_dir runs/ --out plots/

# Heavier smoothing for noisy curves
python scripts/visualize_runs.py --runs_dir runs/ --smooth 0.8

# Different metric (e.g. training return)
python scripts/visualize_runs.py --runs_dir runs/ --metric train/return_mean_20
```

### 5. Generate agent videos

```bash
# All runs (picks best.pt automatically)
python scripts/eval_videos.py --runs_dir runs/ --episodes 3

# Single run
python scripts/eval_videos.py --run_dir runs/<exp_name> --episodes 5

# Specify device (cpu is fine for video generation)
python scripts/eval_videos.py --run_dir runs/<exp_name> --device cpu
```

---

## Diagnosing a bad run

### Step 1 — scan_logs

```bash
python scripts/scan_logs.py
```

Look for flags in the output:

| Flag | Meaning | Fix |
|------|---------|-----|
| `CONT_LOW` | `cont_prob_mean < 0.35` — WM thinks episodes always end | Check `cont_disc_floor=0.9` is set; check `cont_loss_weight=0.5` |
| `VALUE_DIVERGE` | `return_scale > 50` — critic bootstrap exploded | Check `symlog_critic=true`, `symlog_clamp=9.0` in config |
| `STD_HIGH` | `actor_std > 0.9` — policy not specialising | Normal early in training; bad if it persists past 50k steps |
| `ACTOR_GRAD_TINY` | `actor_grad_norm < 0.001` | Check `entropy_coef=1e-2`, `pretanh_reg_coef=1e-3` are set |
| `NAN` | Any NaN in losses | Immediate stop; check `symlog_clamp` and `ValueNet` zero-init |

### Step 2 — diagnose

```bash
python scripts/diagnose.py --run_dir runs/<exp_name>
```

The script prints a **VERDICT** section at the end with `OK` / `PROBLEM` for each
health check. Key things to read:

- **KL encoding active** — should be `OK` (>30% of timesteps have KL > free_nats).
  If `PROBLEM`, the posterior is collapsed: z carries no information.
- **Gaussian entropy > 1 nat** — measures exploration. If `PROBLEM`, std has
  collapsed and the policy is deterministic too early.
- **Saturated actions < 10%** — if `PROBLEM`, the actor is stuck at ±1 and
  `pretanh_reg_coef` may need to be raised.
- **Reward RMSE < 1.0** — if `PROBLEM`, the world model hasn't learned reward
  dynamics yet (may just be early training).

### Step 3 — TensorBoard deep-dive

Open TensorBoard and check:

```
value/bootstrap_mean     ← should be in [-500, 500], not ±1e6
value/return_scale       ← should be in [1, 50]
imagine/cont_prob_mean   ← should be in [0.7, 0.99]
policy/std_mean          ← should start ~0.7, settle ~0.5
loss/world_model         ← should decrease and flatten
loss/actor               ← should decrease from ~1.0, never NaN
```

---

## Ablations (run after baseline is confirmed)

```bash
# Uncertainty metric ablation (latent_mean_l2 vs next_obs_mean_l2 vs gaussian_kl)
python scripts/run_experiments.py --group ablation_metric --parallel 8 --gpus 8

# Ensemble size ablation (N=1,2,3)
python scripts/run_experiments.py --group ablation_ensemble --parallel 8 --gpus 8

# Threshold sensitivity (τ_high, τ_mid pairs)
python scripts/run_experiments.py --group ablation_threshold --parallel 8 --gpus 8

# Collect all results into CSVs + thesis figures
python scripts/collect_results.py
```

---

## Code structure

```
train.py                          entry point
configs/
  config.yaml                     base config (all defaults)
  hopper_baseline.yaml            Hopper fixed_h15 reference
  hopper_adaptive.yaml            Hopper adaptive
  walker2d_baseline.yaml
  walker2d_adaptive.yaml
  quickstart.yaml                 InvertedPendulum smoke test

thesiswm/
  models/
    rssm.py                       RSSM + EnsembleWorldModel
    networks.py                   MLP, DiagGaussian
  agents/
    actor_critic.py               TanhGaussianPolicy + ValueNet
  training/
    trainer.py                    orchestrator: collection loop, eval, checkpointing
    world_model_updater.py        update_world_model() — WM loss + optimizer step
    actor_critic_updater.py       update_actor_critic() — imagination + AC loss
    imagination.py                lambda_returns(), decide_horizon()
  data/
    replay_buffer.py              sequence replay buffer
  utils/
    symlog.py                     symlog/symexp (DreamerV3 §A.2)
    checkpoint.py                 save/load with rollback support
    sigreg.py                     SIGReg regulariser
    seed.py / rng.py              seeding + RNG state capture
    video.py                      MP4 writer (imageio-ffmpeg)
    logger.py                     TBLogger wrapper

scripts/
  run_experiments.py              launch experiment groups (parallel, multi-GPU)
  scan_logs.py                    live run health monitor
  diagnose.py                     checkpoint health report (WM + AC stats + verdict)
  visualize_runs.py               learning curve plots from TensorBoard
  eval_videos.py                  batch video generation from checkpoints
  collect_results.py              full thesis results: CSVs + publication figures
  smoke_test.py                   fast sanity check (~2 min, no GPU needed)
  check_mujoco_render.py          verify headless MuJoCo rendering works
  test_symlog_critic.py           unit tests for symlog critic

thesis/
  thesis_skeleton.tex             LaTeX skeleton with \RESULT{} markers
  plain_language_guide.md         plain-language notes on the method
```

---

## Key design decisions (thesis contribution)

| Component | Choice | Why |
|-----------|--------|-----|
| Adaptive horizon | Ensemble disagreement → H ∈ {5, 10, 20} | Core contribution: uncertainty-aware planning depth |
| Uncertainty metric | `latent_mean_l2` (pairwise between ensemble members) | Operates in z-space (≈N(0,I)); scale matches thresholds |
| Critic | Symlog space + EMA target (decay=0.98) | Prevents bootstrap divergence (DreamerV3 §A.2) |
| Actor | TanhGaussian, Gaussian entropy, pretanh_reg | Prevents saturation, keeps gradients alive |
| WM regulariser | SIGReg (weight=0.01) | Pushes posterior z toward N(0,I), aids KL |
| Cont head | Positive bias init (+3.0), floor=0.9 | Prevents early cont collapse killing horizon |

---

## Install

```bash
pip install torch gymnasium[mujoco] hydra-core omegaconf tensorboard imageio[ffmpeg] tqdm
```

Set `MUJOCO_GL=egl` for headless rendering on a server (required for videos):
```bash
export MUJOCO_GL=egl
```
