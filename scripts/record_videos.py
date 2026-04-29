#!/usr/bin/env python3
"""Record full deterministic episodes from trained Dreamer checkpoints.

Examples:
  python scripts/record_videos.py --runs_dir runs --episodes 3
  python scripts/record_videos.py --runs_dir runs --filter hopper --checkpoint final
  python scripts/record_videos.py --checkpoint runs/my_run/checkpoints/final.pt --episodes 2

Videos are written by default to:
  <run_dir>/videos/<checkpoint_stem>/episode_000_return_123.4.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

def _early_mujoco_gl_backend(argv: list[str]) -> str:
    """Read --mujoco_gl before importing Gym/MuJoCo.

    MuJoCo must see MUJOCO_GL before its rendering modules are imported. On
    headless Linux, egl is usually the right default; osmesa is the fallback for
    CPU-only software rendering.
    """
    for i, arg in enumerate(argv):
        if arg == "--mujoco_gl" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--mujoco_gl="):
            return arg.split("=", 1)[1]
    return "egl" if sys.platform.startswith("linux") else "glfw"


os.environ.setdefault("MUJOCO_GL", _early_mujoco_gl_backend(sys.argv[1:]))

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_checkpoints import (
    find_checkpoint,
    load_config,
    make_agent,
    obs_to_tensor,
    resolve_runs_dir,
    scan_entries,
)


def load_imageio():
    try:
        import imageio.v2 as imageio
        return imageio
    except Exception as exc:
        raise RuntimeError(
            "imageio is required for MP4 writing. Install with: pip install imageio imageio-ffmpeg"
        ) from exc


@torch.no_grad()
def record_episode(agent, env, out_path: Path, seed: int, device: torch.device, fps: int) -> tuple[float, int, int]:
    imageio = load_imageio()
    frames = []

    obs = env.reset(seed=seed)
    frame = env.render()
    if frame is not None:
        frames.append(np.asarray(frame))

    state = agent.get_initial_state(1)
    action, state = agent.act(obs_to_tensor(obs, 0.0, device), state, eval=True)

    total = 0.0
    length = 0
    done = False
    while not done:
        obs, reward, done, _ = env.step(action.squeeze(0).detach().cpu().numpy())
        total += float(reward)
        length += 1
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))
        action, state = agent.act(obs_to_tensor(obs, reward, device), state, eval=True)

    if not frames:
        raise RuntimeError("environment returned no RGB frames; render_mode='rgb_array' may be unsupported")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps, macro_block_size=16)
    return total, length, len(frames)


def record_for_run(run_dir: Path, ckpt_path: Path, args, device: torch.device) -> list[str]:
    config = load_config(run_dir)
    agent, env = make_agent(config, device, render_mode="rgb_array")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["agent_state_dict"])
        agent.eval()

        base_dir = Path(args.out_dir) if args.out_dir else run_dir / "videos"
        if not base_dir.is_absolute():
            base_dir = ROOT / base_dir
        video_dir = base_dir / ckpt_path.stem

        lines = [
            f"run: {run_dir.name}",
            f"env: {config.env.id}",
            f"checkpoint: {ckpt_path}",
            f"output_dir: {video_dir}",
        ]
        for ep in range(args.episodes):
            tmp_name = f"episode_{ep:03d}.mp4"
            tmp_path = video_dir / tmp_name
            ret, length, frame_count = record_episode(
                agent, env, tmp_path, args.seed + ep, device, args.fps
            )
            final_path = video_dir / f"episode_{ep:03d}_return_{ret:.1f}_len_{length}.mp4"
            if final_path.exists():
                final_path.unlink()
            tmp_path.rename(final_path)
            lines.append(
                f"episode {ep:03d}: return={ret:.2f}, length={length}, frames={frame_count}, file={final_path.name}"
            )
            print(f"  wrote {final_path}")

        summary_path = video_dir / "video_summary.txt"
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return lines
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--filter", default="")
    parser.add_argument("--checkpoint", default="final", help="best, latest, final, step_*.pt, or direct .pt path")
    parser.add_argument("--episodes", type=int, default=3, help="Videos per checkpoint")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--mujoco_gl",
        default=os.environ.get("MUJOCO_GL", ""),
        choices=["egl", "osmesa", "glfw", ""],
        help="MuJoCo render backend. Must be set before imports; default is egl on Linux.",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out_dir", default="", help="Optional shared output dir; default is <run_dir>/videos")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}")

    runs_dir = resolve_runs_dir(args.runs_dir)
    entries = scan_entries(runs_dir, args.filter, args.checkpoint)
    if not entries:
        print(f"No checkpoints found in {runs_dir}")
        return

    for run_dir, ckpt in entries:
        print(f"{run_dir.name:<55} {ckpt}")
    if args.list:
        return

    device = torch.device(args.device)
    failures = []
    for i, (run_dir, ckpt_path) in enumerate(entries, start=1):
        print(f"[{i}/{len(entries)}] recording {run_dir.name} ({ckpt_path.name})")
        try:
            record_for_run(run_dir, ckpt_path, args, device)
        except Exception:
            err = traceback.format_exc(limit=8)
            failures.append((run_dir.name, err))
            print(f"  FAILED {run_dir.name}: {err.splitlines()[-1]}")

    if failures:
        print("\nFailures:")
        for name, err in failures:
            print(f"\n{name}\n{err}")


if __name__ == "__main__":
    main()
