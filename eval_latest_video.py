import os
import glob
import argparse
import numpy as np
import torch

from thesiswm.envs.make_env import make_env
from thesiswm.training.trainer import Trainer
from thesiswm.utils.video import write_mp4
from thesiswm.utils.seed import set_global_seeds

from omegaconf import OmegaConf
import hydra

def find_latest_ckpt(run_dir: str) -> str:
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    latest = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(latest):
        return latest
    # fallback: pick newest *.pt
    pts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")), key=os.path.getmtime)
    if not pts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return pts[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="quickstart", help="Hydra config name in configs/")
    ap.add_argument("--run_dir", required=True, help="runs/<exp_name>")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--video_dir", default=None)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    # Load hydra config file directly (simple + robust for a utility script)
    cfg_path = os.path.join("configs", f"{args.config}.yaml")
    cfg = OmegaConf.load(cfg_path)
    cfg.device = args.device

    set_global_seeds(int(cfg.seed), deterministic=True)

    ckpt_path = find_latest_ckpt(args.run_dir)
    ckpt = torch.load(ckpt_path, map_location=torch.device(args.device), weights_only=False)

    # Ensure dims + env id match the checkpoint
    cfg.world_model.obs_dim = int(ckpt["cfg"]["world_model"]["obs_dim"])
    cfg.world_model.act_dim = int(ckpt["cfg"]["world_model"]["act_dim"])
    cfg.env.id = str(ckpt["cfg"]["env"]["id"])

    trainer = Trainer(cfg, build_env=False)
    trainer.checkpointer.load_into(trainer, ckpt)
    trainer.agent.eval()
    trainer.world_model_ensemble.eval()

    record = args.video_dir is not None
    env = make_env(cfg.env.id, seed=int(cfg.seed), render_mode="rgb_array" if record else None)

    os.makedirs(args.video_dir, exist_ok=True) if record else None

    rets, lens = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=int(cfg.seed) + 10000 + ep)
        done = False
        trunc = False
        frames = []
        ret = 0.0
        length = 0

        while not (done or trunc):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=torch.device(args.device)).unsqueeze(0)
            with torch.no_grad():
                act = trainer.agent.act_deterministic_from_obs(obs_t, trainer.world_model_ensemble, device=torch.device(args.device))
            act_np = act.squeeze(0).cpu().numpy().astype(np.float32)

            if record:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            obs, reward, done, trunc, _ = env.step(act_np)
            ret += float(reward)
            length += 1

        rets.append(ret)
        lens.append(length)

        if record:
            out_path = os.path.join(args.video_dir, f"latest_ep{ep:03d}.mp4")
            write_mp4(frames, out_path, fps=args.fps)

        print(f"[EP {ep}] return={ret:.2f} len={length}")

    env.close()
    print(f"[EVAL] ckpt={ckpt_path}")
    print(f"[EVAL] return_mean={float(np.mean(rets)):.2f} ± {float(np.std(rets)):.2f}  len_mean={float(np.mean(lens)):.1f}")

if __name__ == "__main__":
    main()
