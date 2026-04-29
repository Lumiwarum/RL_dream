#!/usr/bin/env python3
"""Evaluate trained Dreamer checkpoints across completed runs.

Examples:
  python scripts/eval_checkpoints.py --runs_dir runs --episodes 30
  python scripts/eval_checkpoints.py --runs_dir runs --filter hopper --checkpoint final
  python scripts/eval_checkpoints.py --checkpoint runs/my_run/checkpoints/final.pt --episodes 50
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dreamer.agent import Dreamer
from dreamer.envs import MuJoCoWrapper


CHECKPOINT_PRIORITY = {
    "best": ["best.pt", "latest.pt", "final.pt"],
    "latest": ["latest.pt", "final.pt", "best.pt"],
    "final": ["final.pt", "latest.pt", "best.pt"],
}


@dataclass
class EvalResult:
    run_name: str
    env_id: str
    checkpoint: str
    step: int
    episodes: int
    mean: float = math.nan
    std: float = math.nan
    median: float = math.nan
    p5: float = math.nan
    p95: float = math.nan
    min_return: float = math.nan
    max_return: float = math.nan
    mean_length: float = math.nan
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def resolve_runs_dir(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def find_checkpoint(run_dir: Path, prefer: str) -> Optional[Path]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    if prefer.endswith(".pt"):
        p = Path(prefer)
        return p if p.is_absolute() else run_dir / prefer
    for name in CHECKPOINT_PRIORITY.get(prefer, CHECKPOINT_PRIORITY["final"]):
        p = ckpt_dir / name
        if p.exists():
            return p
    step_ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    return step_ckpts[-1] if step_ckpts else None


def scan_entries(runs_dir: Path, name_filter: str, checkpoint: str) -> list[tuple[Path, Path]]:
    if checkpoint.endswith(".pt"):
        ckpt = Path(checkpoint)
        if not ckpt.is_absolute():
            ckpt = ROOT / ckpt
        return [(ckpt.parents[1], ckpt)] if ckpt.exists() else []

    entries = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if name_filter and name_filter.lower() not in run_dir.name.lower():
            continue
        ckpt = find_checkpoint(run_dir, checkpoint)
        if ckpt is not None and ckpt.exists():
            entries.append((run_dir, ckpt))
    return entries


def load_config(run_dir: Path):
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"missing Hydra config: {config_path}")
    return OmegaConf.load(config_path)


def make_agent(config, device: torch.device, render_mode: str | None = None) -> tuple[Dreamer, MuJoCoWrapper]:
    env = MuJoCoWrapper(
        str(config.env.id),
        seed=int(config.seed) + 12345,
        action_repeat=int(config.env.action_repeat),
        render_mode=render_mode,
    )
    config.device = str(device)
    config.rssm.device = str(device)
    config.buffer.device = str(device)
    config.encoder.mlp.device = str(device)
    config.reward.device = str(device)
    config.cont.device = str(device)
    config.actor.device = str(device)
    config.critic.device = str(device)
    agent = Dreamer(config, env.observation_space, env.action_space).to(device)
    return agent, env


def obs_to_tensor(obs: dict, reward: float, device: torch.device):
    data = {k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0) for k, v in obs.items()}
    data["reward"] = torch.as_tensor([[reward]], dtype=torch.float32, device=device)
    return data


@torch.no_grad()
def run_episode(agent: Dreamer, env: MuJoCoWrapper, seed: int, device: torch.device) -> tuple[float, int]:
    obs = env.reset(seed=seed)
    state = agent.get_initial_state(1)
    action, state = agent.act(obs_to_tensor(obs, 0.0, device), state, eval=True)

    total = 0.0
    length = 0
    done = False
    while not done:
        obs, reward, done, _ = env.step(action.squeeze(0).detach().cpu().numpy())
        total += float(reward)
        length += 1
        action, state = agent.act(obs_to_tensor(obs, reward, device), state, eval=True)
    return total, length


def evaluate_one(run_dir: Path, ckpt_path: Path, episodes: int, device: torch.device, seed: int) -> EvalResult:
    config = load_config(run_dir)
    result = EvalResult(
        run_name=run_dir.name,
        env_id=str(config.env.id),
        checkpoint=str(ckpt_path.relative_to(run_dir) if ckpt_path.is_relative_to(run_dir) else ckpt_path),
        step=0,
        episodes=episodes,
    )
    env = None
    try:
        agent, env = make_agent(config, device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["agent_state_dict"])
        result.step = int(ckpt.get("step", 0))
        agent.eval()

        returns, lengths = [], []
        for ep in range(episodes):
            ret, length = run_episode(agent, env, seed + ep, device)
            returns.append(ret)
            lengths.append(length)

        arr = np.asarray(returns, dtype=np.float64)
        result.mean = float(arr.mean())
        result.std = float(arr.std())
        result.median = float(np.median(arr))
        result.p5 = float(np.percentile(arr, 5))
        result.p95 = float(np.percentile(arr, 95))
        result.min_return = float(arr.min())
        result.max_return = float(arr.max())
        result.mean_length = float(np.mean(lengths))
    except Exception:
        result.error = traceback.format_exc(limit=8)
    finally:
        if env is not None:
            env.close()
    return result


def format_results(results: list[EvalResult], runs_dir: Path, args) -> str:
    lines = []
    lines.append("Dreamer checkpoint evaluation")
    lines.append(f"created: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"runs_dir: {runs_dir}")
    lines.append(f"checkpoint preference: {args.checkpoint}")
    lines.append(f"episodes per checkpoint: {args.episodes}")
    lines.append("")

    header = (
        f"{'RUN':<48} {'ENV':<28} {'STEP':>8} {'MEAN':>10} {'STD':>9} "
        f"{'MED':>10} {'P5':>10} {'P95':>10} {'MIN':>10} {'MAX':>10} {'LEN':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in sorted(results, key=lambda x: (x.env_id, -x.mean if x.ok else float("inf"))):
        if r.ok:
            lines.append(
                f"{r.run_name:<48} {r.env_id:<28} {r.step:>8} {r.mean:>10.2f} {r.std:>9.2f} "
                f"{r.median:>10.2f} {r.p5:>10.2f} {r.p95:>10.2f} "
                f"{r.min_return:>10.2f} {r.max_return:>10.2f} {r.mean_length:>7.1f}"
            )
        else:
            lines.append(f"{r.run_name:<48} {r.env_id:<28} {'FAIL':>8} {r.error.splitlines()[-1]}")

    lines.append("")
    ok = [r for r in results if r.ok]
    if ok:
        lines.append("Per-environment best means:")
        for env_id in sorted({r.env_id for r in ok}):
            best = max((r for r in ok if r.env_id == env_id), key=lambda r: r.mean)
            lines.append(f"  {env_id:<28} {best.mean:>10.2f}  {best.run_name} ({best.checkpoint})")
    bad = [r for r in results if not r.ok]
    if bad:
        lines.append("")
        lines.append("Failures:")
        for r in bad:
            lines.append(f"  {r.run_name}:")
            lines.append(r.error)
    return "\n".join(lines) + "\n"


def write_csv(results: list[EvalResult], path: Path) -> None:
    fields = list(EvalResult.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--filter", default="")
    parser.add_argument("--checkpoint", default="final", help="best, latest, final, step_*.pt, or direct .pt path")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="", help="Optional output .txt path")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

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
    results = []
    for i, (run_dir, ckpt) in enumerate(entries, start=1):
        print(f"[{i}/{len(entries)}] evaluating {run_dir.name} ({ckpt.name})")
        results.append(evaluate_one(run_dir, ckpt, args.episodes, device, args.seed))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else runs_dir / f"eval_results_{args.checkpoint}_{args.episodes}eps_{stamp}.txt"
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(format_results(results, runs_dir, args), encoding="utf-8")
    write_csv(results, out_path.with_suffix(".csv"))
    print(f"\nWrote {out_path}")
    print(out_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
