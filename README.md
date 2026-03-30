# ThesisWM — DreamerV3-style World Model RL with Adaptive Imagination Horizon (Ensemble)

This repo is a compact DreamerV3-inspired agent for **robotics RL** (Gymnasium / MuJoCo style state-vector tasks).
The core idea is to learn a latent dynamics model (RSSM) and train an actor-critic purely from **imagined rollouts**.

## Key idea: Adaptive imagination horizon using an ensemble
Dreamer typically imagines a fixed horizon (e.g., H=15). Here we keep an **ensemble of world models** and estimate
**epistemic uncertainty** via **disagreement** between ensemble members. At decision time (during actor/critic updates),
we choose a horizon:
- high disagreement → short rollout (safer, less model bias)
- low disagreement → long rollout (more compute-efficient and better long-term credit assignment)

So the agent can “look ahead longer” only when the model is confident.

### Components
- `rssm.py`: RSSM latent dynamics + prediction heads (obs/reward/continue)
- `trainer.py`: data collection, world model training, imagined rollouts, actor-critic updates, checkpointing
- `imagination.py`: lambda-returns + adaptive horizon chooser
- `actor_critic.py`: tanh-squashed Gaussian policy + value network
- `replay_buffer.py`: sequence replay buffer (for RSSM training)
- `eval.py`: evaluate a checkpoint; can record MP4 videos
- `video.py`: MP4 writer (imageio)

## Install
Create a venv and install dependencies (typical):
- torch
- gymnasium (+ mujoco)
- hydra-core, omegaconf
- tensorboard
- imageio[ffmpeg]

(Exact dependency list depends on your server setup.)

## Useful commands
### Quickstart: smoke test
```bash
python scripts/smoke_test.py
```
### Train (example)
```
python train.py --config quickstart
```
### Resume training
```
python train.py --config quickstart --resume
```
### Override steps per chunk
```
python train.py --config quickstart --steps_per_chunk 30000 --resume
```
### Launch tensorboard page
```
tensorboard --logdir runs --port 8032
```
### Evaluate a checkpoint (print returns)
```
python eval.py --config quickstart --checkpoint runs/<exp_name>/checkpoints/latest.pt --episodes 10
```
### Record evaluation video (MP4)
```
python eval.py --config quickstart \
  --checkpoint runs/<exp_name>/checkpoints/latest.pt \
  --episodes 3 --record_video --video_dir runs/<exp_name>/videos
```

### Evaluate video 
```
python eval_latest_video.py --config quickstart --run_dir runs/<exp_name> --episodes 5 --device cuda --video_dir runs/<exp_name>/videos_latest
```

## Experiments

### Experiment groups

| Group | Description | Experiments |
|---|---|---|
| `main` | Hopper + Walker2d × {fixed_h5, fixed_h15, fixed_h20, adaptive} × 3 seeds | 24 |
| `ablation_metric` | Hopper × 3 uncertainty metrics × 3 seeds | 9 |
| `ablation_ensemble` | Hopper × ensemble sizes {1,2,3} × 3 seeds | 12 |
| `ablation_threshold` | Hopper × 4 threshold pairs × 3 seeds | 12 |

### Running experiments

```bash
# 1. Preview commands without running (dry run)
python scripts/run_experiments.py --group main --dry_run

# 2. List all experiments in a group
python scripts/run_experiments.py --group main --list

# 3. Run sequentially (single GPU)
python scripts/run_experiments.py --group main

# 4. Run in parallel — 3 jobs at once, cycling across 2 GPUs
python scripts/run_experiments.py --group main --parallel 3 --gpus 2

# 5. Run in parallel on a single GPU (2 jobs share the GPU)
python scripts/run_experiments.py --group main --parallel 2 --gpus 1

# 6. Resume: skip already-completed experiments
python scripts/run_experiments.py --group main --parallel 6 --gpus 3 --skip_done

# 7. Run all groups back-to-back
python scripts/run_experiments.py --group all --parallel 6 --gpus 3 --skip_done

# 8. Re-run a group from scratch (e.g. after deleting run directories)
#    --reset marks all experiments in the group back to 'pending' before running.
#    Other groups in the manifest are NOT affected.
python scripts/run_experiments.py --group main --parallel 6 --gpus 3 --reset
```

> **Tip — GPU memory:** each job uses ~2–4 GB VRAM (ensemble_size=2, rollout_batch=512).
> On a 24 GB card you can safely run `--parallel 4 --gpus 1`.
> With 2 GPUs of 12 GB each, use `--parallel 4 --gpus 2`.

> **Tip — FPS and `num_envs`:** adding more parallel environments does **not** improve FPS if
> `updates_per_step` is scaled proportionally. With N envs and `collect_per_step=4`, each outer
> loop collects `4×N` transitions and fires `updates_per_step` GPU gradient updates.
> FPS ∝ `(4×N) / (updates_per_step × T_gpu)`.
> To actually gain speed from more envs, keep `updates_per_step` fixed (e.g. 4) regardless of N.
> With `num_envs=8` and `updates_per_step=4` you get ~2× the FPS vs `num_envs=4` at a slightly
> lower update intensity (1 update per 8 transitions instead of 1 per 4).

### Collecting results and plotting

```bash
python scripts/collect_results.py
```

Reads `experiments/manifest.csv`, aggregates TensorBoard logs, and writes summary CSVs + plots to `experiments/results/`.

### Full workflow

```bash
# Step 1 — main comparison (fixed horizons vs. adaptive)
python scripts/run_experiments.py --group main --parallel 3 --gpus 2

# Step 2 — ablations (can run while main is still going with --skip_done)
python scripts/run_experiments.py --group ablation_metric --parallel 3 --gpus 2
python scripts/run_experiments.py --group ablation_ensemble --parallel 3 --gpus 2
python scripts/run_experiments.py --group ablation_threshold --parallel 3 --gpus 2

# Step 3 — collect and plot
python scripts/collect_results.py

# Step 4 — monitor training live
tensorboard --logdir runs --port 8032

# Step 5 — fill in thesis
# open thesis/thesis_skeleton.tex and replace \RESULT{} markers
```

### Checkpoint policy

By default only two checkpoints are kept per run to save disk space:
- `best.pt` — saved whenever a new best evaluation return is achieved
- `latest.pt` — saved at the end of each training chunk (used for resuming)

To re-enable periodic step snapshots (e.g. every 10k steps), set:
```bash
python train.py ... training.save_periodic_checkpoints=true
```
