import argparse
import os

import numpy as np

from thesiswm.envs.make_env import make_env
from thesiswm.utils.video import write_mp4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Hopper-v4")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--out", type=str, default="runs/_render_test/test.mp4")
    args = parser.parse_args()

    print("=== MuJoCo Headless Render Diagnostics ===")
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '(not set)')}")
    print(f"PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM', '(not set)')}")
    print(f"Env: {args.env}")

    env = make_env(args.env, seed=0, render_mode="rgb_array")
    obs, _ = env.reset(seed=0)

    frames = []
    for t in range(args.steps):
        action = env.action_space.sample().astype(np.float32)
        obs, reward, done, trunc, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if done or trunc:
            obs, _ = env.reset(seed=100 + t)

    env.close()
    write_mp4(frames, args.out, fps=30)
    print(f"Wrote MP4: {args.out}")
    print(f"Captured frames: {len(frames)}")
    print("OK")


if __name__ == "__main__":
    main()
