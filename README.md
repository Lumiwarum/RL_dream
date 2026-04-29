# DreamerV3 — Adaptive Imagination Horizon (Thesis Implementation)

A DreamerV3-style world-model agent extended with two thesis contributions:

1. **Adaptive imagination horizon** — the planning horizon H is selected at runtime based on an EMA of the world model's prior-prediction quality. When the WM is unreliable (high prior loss), a shorter H is used to limit compounding errors.
2. **World-model ensemble** — N independent world models provide diverse reward estimates during imagination. Optionally averages rewards across members for a pessimistic/robust signal.

Built on top of [r2dreamer](https://github.com/NM512/r2dreamer) (NM512). The representation loss (InfoNCE / Barlow-Twins) has been removed; the model uses the standard DreamerV3 decoder reconstruction objective instead.

---

## Architecture

```
Observations ──► Encoder (MLP, symlog) ──► RSSM
                                               │
                              ┌────────────────┴──────────────────────┐
                              │  Posterior z_t (observe step)         │
                              │  Prior z_t     (imagine step)         │
                              └────────────────┬──────────────────────┘
                                               │
                              ┌────────────────▼──────────────────────┐
                              │  Decoder  (obs reconstruction)        │
                              │  Reward head (symexp two-hot bins)    │
                              │  Continue head (Bernoulli)            │
                              └───────────────────────────────────────┘
                                               │
                         Actor ◄───── features (h_t ⊕ z_t) ──────► Critic
```

Key details:
- **RSSM**: block-GRU deterministic core, categorical stochastic latent (32×32 → 1024-dim)
- **Encoder/Decoder**: 3-layer MLP with SiLU + RMSNorm; symlog input preprocessing
- **Actor**: bounded Normal distribution with learnable std
- **Critic**: symexp two-hot binned distribution (255 bins), slow EMA target
- **Optimizer**: LaProp with AGC gradient clipping and linear warmup

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, CUDA 12.8 (for `torch==2.8.0`), and MuJoCo 3.x.

To verify the environment:

```bash
cd dreamer_impl
python scripts/env_scan.py
```

---

## Quick Start

### Single training run

```bash
cd dreamer_impl

# Fixed horizon H=15, single WM, Hopper
python train.py exp_name=hopper_h15 env.id=Hopper-v4

# Fixed horizon H=15, Pendulum (faster smoke test)
python train.py exp_name=pendulum_h15 env.id=InvertedPendulum-v5 trainer.steps=200000

# Adaptive horizon, single WM
python train.py exp_name=hopper_adaptive env.id=Hopper-v4 \
    adaptive.enabled=true

# Ensemble (N=2) with averaged rewards
python train.py exp_name=hopper_ens env.id=Hopper-v4 \
    model.ensemble_size=2 model.ensemble_avg_rewards=true
```

All results land under `dreamer_impl/runs/<exp_name>/`.

### Overriding config values

Any config key can be overridden on the command line with `key=value` or `section.key=value`:

```bash
python train.py exp_name=test env.id=Walker2d-v4 \
    trainer.steps=200000 \
    env.num_envs=4 \
    adaptive.enabled=true \
    adaptive.horizons=[10,15,20] \
    adaptive.thresh_high=1.8 adaptive.thresh_mid=1.3
```

---

## Running Experiment Groups

`scripts/run_experiments.py` manages predefined experiment groups and a CSV manifest.

```bash
# List experiments in a group
python scripts/run_experiments.py --group smoke --list

# Dry-run (print commands without executing)
python scripts/run_experiments.py --group smoke --dry_run

# Run all experiments in the smoke group sequentially
python scripts/run_experiments.py --group smoke

# Run 4 experiments in parallel across 2 GPUs
python scripts/run_experiments.py --group main --parallel 4 --gpus 2

# Skip experiments already marked done in the manifest
python scripts/run_experiments.py --group main --skip_done
```

Available groups:

| Group | Environments | Methods | Seeds |
|-------|-------------|---------|-------|
| `smoke` | InvertedPendulum | fixed_h15, adaptive_n1 | 0 |
| `main` | Hopper | fixed_h10/h15/h20, adaptive_n1 | 0,1,2 |
| `control` | Hopper | fixed_h20, fixed_h20_ens, adaptive, adaptive_ens | 0,1 |
| `control_full` | Hopper, InvertedPendulum | same as control | 0,1,2 |

The manifest is written to `experiments/manifest.csv` and tracks `status`, `start_time`, `end_time` per experiment.

---

## TensorBoard Monitoring

```bash
cd dreamer_impl
tensorboard --logdir runs
```

Key metrics to watch:

| Tag | Meaning |
|-----|---------|
| `eval/return_mean` | Mean eval return across eval envs |
| `eval/best_return` | Best eval return seen so far |
| `train/loss/policy` | Actor loss |
| `train/loss/value` | Value loss |
| `train/loss/dyn` | KL divergence (dynamics) |
| `train/loss/rep` | KL divergence (representation) |
| `train/loss/obs` | Observation reconstruction loss |
| `train/wm/prior_pred_loss` | Prior prediction loss (horizon signal) |
| `train/wm/ema_obs_loss` | EMA of prior_pred_loss |
| `train/imagine/horizon` | Selected imagination horizon H |
| `train/ret` | Mean imagined lambda-return |
| `train/adv` | Mean advantage |
| `train/action_entropy` | Policy entropy |

---

## Analyzing Run Logs

```bash
cd dreamer_impl

# Quick table scan (all runs)
python scripts/scan_logs.py

# Filter to specific runs
python scripts/scan_logs.py --runs_dir runs --filter hopper

# Verbose output with per-run detail
python scripts/scan_logs.py --verbose

# Skip TensorBoard cache
python scripts/scan_logs.py --no_cache
```

The scanner reports per-run summary statistics including peak return, stability (% of eval steps ≥ 70% of peak), and flags like `SPIKE_THEN_DROP` and `STD_FLOOR_STUCK`.

---

## Evaluating Checkpoints

After training, evaluate saved checkpoints over more episodes for reliable statistics:

```bash
cd dreamer_impl

# Evaluate best.pt in every run directory (30 episodes each)
python scripts/eval_checkpoints.py --runs_dir runs --episodes 30

# Evaluate only runs matching "hopper"
python scripts/eval_checkpoints.py --runs_dir runs --filter hopper --checkpoint best --episodes 50

# Prefer latest checkpoint
python scripts/eval_checkpoints.py --runs_dir runs --checkpoint latest --episodes 20

# Evaluate a single checkpoint directly
python scripts/eval_checkpoints.py --checkpoint runs/hopper_h15_s0/checkpoints/best.pt --episodes 50

# List found checkpoints without running
python scripts/eval_checkpoints.py --runs_dir runs --list
```

Results are written to `runs/eval_results_<checkpoint>_<N>eps_<timestamp>.txt` and a matching `.csv`.

Checkpoint preference order: `best` → `latest` → `final` (falls back along the chain if a file is missing).

---

## Recording Videos

```bash
cd dreamer_impl

# Record 3 episodes per run (uses best.pt by default)
python scripts/record_videos.py --runs_dir runs --episodes 3

# Record from a specific checkpoint
python scripts/record_videos.py --runs_dir runs --filter pendulum --checkpoint final --episodes 2

# Write all videos to a shared directory
python scripts/record_videos.py --runs_dir runs --out_dir videos/ --episodes 3

# Change FPS
python scripts/record_videos.py --runs_dir runs --fps 60
```

Videos are saved as `<run_dir>/videos/<checkpoint_stem>/episode_000_return_123.4_len_200.mp4`.

Requires `imageio` and `ffmpeg`:

```bash
pip install imageio imageio-ffmpeg
```

---

## Configuration Reference

The full config is in `configs/config.yaml`. Key sections:

### Thesis features

```yaml
model:
  ensemble_size: 1           # 1 = single WM baseline; 2+ = ensemble
  ensemble_avg_rewards: false # average reward across WM members during imagination
  imag_horizon: 15           # fixed horizon when adaptive.enabled=false

adaptive:
  enabled: false
  horizons: [10, 15, 20]    # [H_low, H_mid, H_high]
  thresh_high: 0.27          # ema_obs_loss > this → H_low
  thresh_mid: 0.20           # ema_obs_loss > this → H_mid
  ema_alpha: 0.003
  ema_init: 2.0              # large → start at H_low until WM quality improves
```

Recommended thresholds per environment:

| Environment | `thresh_high` | `thresh_mid` |
|-------------|--------------|-------------|
| Hopper-v4 | 0.27 | 0.20 |
| Walker2d-v4 | 1.80 | 1.30 |
| InvertedPendulum-v5 | 0.30 | 0.08 |

### Training

```yaml
trainer:
  steps: 500000
  train_ratio: 512           # env steps per gradient update (controls update frequency)
  eval_every: 10000
  eval_episodes: 5
  checkpoint_every: 10000    # save latest.pt every N env steps
  save_best: true            # keep best.pt when eval return improves
  save_periodic: false       # also save step_<N>.pt snapshots
```

### Checkpoints

Checkpoints are saved under `runs/<exp_name>/checkpoints/`:

| File | When saved |
|------|-----------|
| `best.pt` | When eval return improves |
| `latest.pt` | After each eval and at checkpoint_every intervals |
| `final.pt` | At the end of training |
| `step_<N>.pt` | Periodically, if `save_periodic=true` |

Each checkpoint contains:
- `agent_state_dict` — full model weights
- `optims_state_dict` — optimizer state (for resuming)
- `step` — training step at save time
- `updates` — gradient update count
- `eval_return` — eval return at save time
- `best_eval_return` — best seen so far
- `cfg` — full resolved Hydra config

### DreamerV3 hyperparameters

```yaml
horizon: 333        # discount horizon (γ = 1 - 1/333 ≈ 0.997)
lamb: 0.95          # TD-λ mixing (0=1-step TD, 1=MC)
act_entropy: 3e-4   # entropy regularization on actor
kl_free: 1.0        # KL free-nats
lr: 4e-5            # LaProp learning rate (shared for all components)
agc: 0.3            # adaptive gradient clipping threshold
warmup: 1000        # linear LR warmup steps
```

---

## Directory Structure

```
dreamer_impl/
├── train.py                  # Entry point
├── configs/
│   └── config.yaml           # All hyperparameters
├── dreamer/
│   ├── agent.py              # Dreamer agent (WorldModel, ensemble, actor-critic)
│   ├── trainer.py            # Online training loop + checkpointing
│   ├── envs.py               # MuJoCo environment wrapper
│   ├── rssm.py               # RSSM (block-GRU + categorical latent)
│   ├── networks.py           # Encoder, Decoder, MLPHead, ReturnEMA
│   ├── buffer.py             # Replay buffer (TorchRL SliceSampler)
│   ├── distributions.py      # Custom distributions (symexp_twohot, bounded_normal)
│   ├── tools.py              # Logger, EMA, seeding utilities
│   └── parallel.py           # Vectorised environment wrapper
├── scripts/
│   ├── run_experiments.py    # Experiment launcher with manifest tracking
│   ├── scan_logs.py          # Fast TensorBoard log scanner
│   ├── eval_checkpoints.py   # Post-hoc checkpoint evaluation
│   ├── record_videos.py      # Video recording from checkpoints
│   └── env_scan.py           # Environment discovery utility
├── r2dreamer_src/            # Original r2dreamer source for reference
└── experiments/
    └── manifest.csv          # Auto-generated experiment status tracker
```

---

## Differences from Original r2dreamer

| Feature | r2dreamer | This repo |
|---------|-----------|-----------|
| Representation loss | InfoNCE / Barlow-Twins / DreamerPro | Standard decoder reconstruction |
| World models | Single | Ensemble of N |
| Imagination horizon | Fixed | Fixed or adaptive |
| Environments | Atari, DMC, Crafter, MetaWorld | MuJoCo Gymnasium |
| Checkpointing | Manual `torch.save` | Structured best/latest/final |
| Evaluation | In-loop only | In-loop + standalone eval script |
| Video recording | Not supported | `record_videos.py` |
