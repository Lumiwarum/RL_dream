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

1. python scripts/run_experiments.py --group main --dry_run   # preview
2. python scripts/run_experiments.py --group main             # run ~24 experiments
3. python scripts/run_experiments.py --group ablation_metric  # then ablations
4. python scripts/collect_results.py                          # extract + plot
5. open thesis/thesis_skeleton.tex and replace \RESULT{} markers
