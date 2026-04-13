import os
import sys
from dataclasses import asdict
from typing import List, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from thesiswm.envs.make_env import make_env
from thesiswm.models.rssm import RSSMState
from thesiswm.training.trainer import load_checkpoint_into_trainer_state
from thesiswm.utils.seed import set_global_seeds
from thesiswm.utils.video import write_mp4


def _preprocess_argv(argv: List[str]) -> List[str]:
    """
    User-friendly args:
      --config <name>         selects configs/<name>.yaml
      --checkpoint <path>     eval.checkpoint_path=<path>
      --episodes <int>        eval.episodes=<int>
      --record_video          eval.record_video=true
      --video_dir <dir>       eval.video_dir=<dir>
    """
    out: List[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--checkpoint":
            out.append(f"eval.checkpoint_path={argv[i+1]}")
            i += 2
        elif a == "--episodes":
            out.append(f"eval.episodes={int(argv[i+1])}")
            i += 2
        elif a == "--record_video":
            out.append("eval.record_video=true")
            i += 1
        elif a == "--video_dir":
            out.append(f"eval.video_dir={argv[i+1]}")
            i += 2
        elif a == "--config":
            out.append(a)
            out.append(argv[i+1])
            i += 2
        else:
            out.append(a)
            i += 1
    return out


def _extract_config_name(argv: List[str], default_name: str = "config") -> str:
    if "--config" in argv:
        idx = argv.index("--config")
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return default_name


def main():
    argv = _preprocess_argv(sys.argv[1:])
    config_name = _extract_config_name(argv, default_name="config")
    if "--config" in argv:
        idx = argv.index("--config")
        del argv[idx:idx+2]
    sys.argv = [sys.argv[0]] + argv

    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    _hydra_entrypoint(config_name)


def _hydra_entrypoint(config_name: str):
    @hydra.main(version_base=None, config_path="configs", config_name=config_name)
    def _run(cfg: DictConfig) -> None:
        set_global_seeds(int(cfg.seed), deterministic=bool(cfg.deterministic))

        ckpt_path = str(cfg.eval.checkpoint_path)
        if ckpt_path == "" or ckpt_path is None:
            raise ValueError("Provide --checkpoint <path> or set eval.checkpoint_path in config.")

        device = torch.device(cfg.device)
        env = make_env(cfg.env.id, seed=int(cfg.seed), render_mode="rgb_array" if cfg.eval.record_video else None)

        # Build a minimal trainer state and load weights; we reuse Trainer's module construction.
        from thesiswm.training.trainer import Trainer
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Pull dims from checkpoint config and inject into current cfg BEFORE building modules
        cfg.world_model.obs_dim = int(ckpt["cfg"]["world_model"]["obs_dim"])
        cfg.world_model.act_dim = int(ckpt["cfg"]["world_model"]["act_dim"])

        # Also make sure env id matches what was trained (optional but helps avoid confusion)
        cfg.env.id = str(ckpt["cfg"]["env"]["id"])

        trainer = Trainer(cfg, build_env=False)
        trainer.checkpointer.load_into(trainer, ckpt)

        actor = trainer.agent.actor.to(device)
        actor.eval()

        episode_returns = []
        episode_lengths = []

        os.makedirs(str(cfg.eval.video_dir), exist_ok=True)

        wm0 = trainer.world_model_ensemble.models[0]
        for ep in range(int(cfg.eval.episodes)):
            obs, _ = env.reset(seed=int(cfg.seed) + 10_000 + ep)
            done = False
            trunc = False
            ret = 0.0
            length = 0
            frames = []
            rssm_state = wm0.rssm.init_state(1, device)
            prev_action = torch.zeros(1, wm0.rssm.act_dim, device=device)

            while not (done or trunc):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    h = wm0.rssm.deter_step(rssm_state.h, rssm_state.z, prev_action)
                    z = wm0.rssm.posterior(h, obs_t).mean
                    action = trainer.agent.actor.mean_action(torch.cat([h, z], dim=-1))
                rssm_state = RSSMState(h=h, z=z)
                prev_action = action.detach()
                action_np = action.squeeze(0).cpu().numpy().astype(np.float32)

                if cfg.eval.record_video:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)

                obs, reward, done, trunc, _ = env.step(action_np)
                ret += float(reward)
                length += 1

            episode_returns.append(ret)
            episode_lengths.append(length)

            if cfg.eval.record_video:
                out_path = os.path.join(str(cfg.eval.video_dir), f"eval_ep{ep:03d}.mp4")
                write_mp4(frames, out_path, fps=int(cfg.eval.video_fps))

        env.close()

        mean_ret = float(np.mean(episode_returns)) if episode_returns else 0.0
        mean_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0
        print(f"[EVAL] episodes={len(episode_returns)}  return_mean={mean_ret:.2f}  len_mean={mean_len:.1f}")
        print(f"[EVAL] checkpoint={ckpt_path}")
        if cfg.eval.record_video:
            print(f"[EVAL] videos saved to: {cfg.eval.video_dir}")

    _run()


if __name__ == "__main__":
    main()
