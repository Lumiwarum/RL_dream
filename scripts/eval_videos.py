"""
Generate evaluation videos for trained agents.

Usage:
    # All runs in a directory (picks best.pt or latest.pt):
    python scripts/eval_videos.py --runs_dir runs/

    # Single run:
    python scripts/eval_videos.py --run_dir runs/hopperv5_fixed_h20_s0

    # Specific checkpoint:
    python scripts/eval_videos.py --ckpt runs/hopperv5_fixed_h20_s0/checkpoints/best.pt

Options:
    --episodes      Number of eval episodes to record (default: 3)
    --fps           Video frame rate (default: 30)
    --max_steps     Max steps per episode (default: 1000)
    --device        cuda or cpu (default: cpu — faster startup for single eval)
    --out_dir       Where to save videos (default: next to checkpoint)
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import numpy as np
import torch

# Add repo root to path so we can import thesiswm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thesiswm.envs.make_env import make_env
from thesiswm.models.rssm import RSSMState
from thesiswm.training.trainer import Trainer
from thesiswm.utils.video import write_mp4


# ── checkpoint discovery ──────────────────────────────────────────────────────

def _find_ckpt(run_dir: str) -> Optional[str]:
    """Prefer best.pt, fall back to latest.pt, then any .pt file."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    for name in ("best.pt", "best_train.pt", "latest.pt"):
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            return p
    pts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
    return os.path.join(ckpt_dir, pts[-1]) if pts else None


def _find_config(run_dir: str) -> Optional[str]:
    p = os.path.join(run_dir, "config_snapshot.yaml")
    return p if os.path.exists(p) else None


# ── single-run recorder ───────────────────────────────────────────────────────

def record_run(
    run_dir: str,
    ckpt_path: Optional[str],
    episodes: int,
    fps: int,
    max_steps: int,
    device_str: str,
    out_dir: Optional[str],
):
    # Load config
    cfg_path = _find_config(run_dir)
    if cfg_path is None:
        print(f"  [skip] no config_snapshot.yaml in {run_dir}")
        return

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(cfg_path)
    cfg.device = device_str
    cfg.training.resume = False       # don't reload from checkpoint automatically
    cfg.env.num_envs = 1              # single env for recording

    # Build trainer (no env yet — we'll render ourselves)
    trainer = Trainer(cfg, build_env=False)
    device  = trainer.device

    # Load checkpoint
    if ckpt_path is None:
        ckpt_path = _find_ckpt(run_dir)
    if ckpt_path is None:
        print(f"  [skip] no checkpoint found in {run_dir}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    trainer.checkpointer.load_into(trainer, ckpt)
    step = int(trainer.state.env_step)
    print(f"  Loaded {ckpt_path} (step={step})")

    # Output dir
    if out_dir is None:
        out_dir = os.path.join(run_dir, "videos")
    os.makedirs(out_dir, exist_ok=True)

    wm0    = trainer.world_model_ensemble.models[0]
    agent  = trainer.agent
    env_id = str(cfg.env.id)

    for ep in range(episodes):
        env = make_env(env_id, seed=42 + ep, render_mode="rgb_array")
        obs, _ = env.reset(seed=42 + ep)
        done = trunc = False
        ep_ret = 0.0
        frames: List[np.ndarray] = []

        state       = wm0.rssm.init_state(1, device)
        prev_action = torch.zeros(1, wm0.rssm.act_dim, device=device)

        for _ in range(max_steps):
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                h    = wm0.rssm.deter_step(state.h, state.z, prev_action)
                z    = wm0.rssm.posterior(h, obs_t).mean
                feat = torch.cat([h, z], dim=-1)
                action = agent.actor.mean_action(feat)

            state       = RSSMState(h=h.float(), z=z.float())
            prev_action = action.detach().float()
            action_np   = action.squeeze(0).float().cpu().numpy()

            obs, reward, done, trunc, _ = env.step(action_np)
            ep_ret += float(reward)

            if done or trunc:
                break

        env.close()

        out_path = os.path.join(out_dir, f"ep{ep:02d}_ret{ep_ret:.0f}.mp4")
        write_mp4(frames, out_path, fps=fps)
        print(f"    ep{ep}: return={ep_ret:.1f}, {len(frames)} frames → {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate agent evaluation videos.")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--runs_dir",  help="Scan all run subdirectories")
    group.add_argument("--run_dir",   help="Single run directory")
    parser.add_argument("--ckpt",      default=None, help="Explicit checkpoint path")
    parser.add_argument("--episodes",  default=3,    type=int)
    parser.add_argument("--fps",       default=30,   type=int)
    parser.add_argument("--max_steps", default=1000, type=int)
    parser.add_argument("--device",    default="cpu")
    parser.add_argument("--out_dir",   default=None, help="Override video output directory")
    args = parser.parse_args()

    if args.runs_dir:
        runs_dir = args.runs_dir
        run_names = sorted(
            d for d in os.listdir(runs_dir)
            if os.path.isdir(os.path.join(runs_dir, d))
        )
        print(f"Found {len(run_names)} runs in {runs_dir}")
        for name in run_names:
            run_path = os.path.join(runs_dir, name)
            print(f"\n── {name}")
            record_run(
                run_dir=run_path,
                ckpt_path=args.ckpt,
                episodes=args.episodes,
                fps=args.fps,
                max_steps=args.max_steps,
                device_str=args.device,
                out_dir=args.out_dir,
            )
    elif args.run_dir:
        print(f"── {args.run_dir}")
        record_run(
            run_dir=args.run_dir,
            ckpt_path=args.ckpt,
            episodes=args.episodes,
            fps=args.fps,
            max_steps=args.max_steps,
            device_str=args.device,
            out_dir=args.out_dir,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
