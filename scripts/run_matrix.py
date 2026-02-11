import argparse
import subprocess
from typing import List


METHOD_TO_CONFIG = {
    "fixed_h5": "hopper_baseline",      # overridden per env below
    "fixed_h15": "hopper_baseline",
    "fixed_h20": "hopper_baseline",
    "adaptive_5_10_20": "hopper_adaptive",
}


def build_overrides(env_id: str, method: str, seed: int, total_steps: int, steps_per_chunk: int) -> List[str]:
    exp = f"{env_id.replace('-', '').lower()}_{method}_seed{seed}"
    overrides = [
        f"exp_name={exp}",
        f"env.id={env_id}",
        f"method.name={method}",
        f"seed={seed}",
        f"training.total_steps={total_steps}",
        f"training.steps_per_chunk={steps_per_chunk}",
        "training.resume=true",
    ]
    # Map fixed_h20 to horizon_fixed=20, etc.
    if method == "fixed_h5":
        overrides += ["imagination.horizon_fixed=5"]
    elif method == "fixed_h15":
        overrides += ["imagination.horizon_fixed=15"]
    elif method == "fixed_h20":
        overrides += ["imagination.horizon_fixed=20"]
    return overrides


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", nargs="+", default=["Hopper-v4", "Walker2d-v4"])
    p.add_argument("--methods", nargs="+", default=["fixed_h5", "fixed_h15", "fixed_h20", "adaptive_5_10_20"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--total_steps", type=int, default=1_000_000)
    p.add_argument("--steps_per_chunk", type=int, default=200_000)
    p.add_argument("--base_config", type=str, default="config")
    args = p.parse_args()

    for env_id in args.envs:
        for method in args.methods:
            for seed in args.seeds:
                overrides = build_overrides(env_id, method, seed, args.total_steps, args.steps_per_chunk)
                cmd = ["python", "train.py", "--config", args.base_config] + overrides
                print("\n=== RUN ===")
                print(" ".join(cmd))
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
