from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
from omegaconf import OmegaConf

from thesiswm.utils.rng import capture_rng_state, restore_rng_state


class Checkpointer:
    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def latest_path(self) -> Optional[str]:
        p = os.path.join(self.ckpt_dir, "latest.pt")
        return p if os.path.exists(p) else None

    def make_checkpoint(self, trainer: Any) -> Dict[str, Any]:
        """
        Create a full checkpoint dictionary.
        """
        ckpt = {
            "state": {
                "env_step": int(trainer.state.env_step),
                "updates": int(trainer.state.updates),
            },
            "cfg": OmegaConf.to_container(trainer.cfg, resolve=True),
            "rng": capture_rng_state(),
            "replay": trainer.replay.state_dict(),
            "world_model": trainer.world_model_ensemble.state_dict(),
            "agent": trainer.agent.state_dict(),
            "target_critic": trainer.target_critic.state_dict(),
            "opt": {
                "wm_opt": trainer.wm_opt.state_dict(),
                "actor_opt": trainer.actor_opt.state_dict(),
                "critic_opt": trainer.critic_opt.state_dict(),
            },
            "metrics": {
                "best_eval_return":  float(getattr(trainer, "best_eval_return",  -1e9)),
                "best_train_return": float(getattr(trainer, "best_train_return", -1e9)),
                "return_scale":      float(getattr(trainer, "_return_scale",     1.0)),
            }
        }
        return ckpt

    def make_weights_checkpoint(self, trainer: Any) -> Dict[str, Any]:
        """Lightweight checkpoint: weights + metrics only, no replay buffer or optimizer state.
        Used for best_train rollback checkpoints to keep save cost low (~5MB vs ~100MB)."""
        return {
            "state": {
                "env_step": int(trainer.state.env_step),
                "updates": int(trainer.state.updates),
            },
            "world_model": trainer.world_model_ensemble.state_dict(),
            "agent": trainer.agent.state_dict(),
            "target_critic": trainer.target_critic.state_dict(),
            "metrics": {
                "best_eval_return":  float(getattr(trainer, "best_eval_return",  -1e9)),
                "best_train_return": float(getattr(trainer, "best_train_return", -1e9)),
                "return_scale":      float(getattr(trainer, "_return_scale",     1.0)),
            }
        }

    def save(self, ckpt: Dict[str, Any], tag: str, make_latest: bool = True) -> str:
        path = os.path.join(self.ckpt_dir, f"{tag}.pt")
        torch.save(ckpt, path)
        if make_latest:
            latest = os.path.join(self.ckpt_dir, "latest.pt")
            torch.save(ckpt, latest)
        return path

    def load_into(
        self,
        trainer: Any,
        ckpt: Dict[str, Any],
        restore_step: bool = True,
        restore_optimizer: bool = True,
        restore_replay: bool = True,
        noise_scale: float = 0.0,
    ) -> None:
        """
        Load checkpoint into trainer.

        Args:
            restore_step: If False, keeps current env_step/updates (for rollback)
            restore_optimizer: If False, keeps current optimizer state
            restore_replay: If False, keeps current replay buffer
            noise_scale: If > 0, adds Gaussian noise to weights (avoid identical trajectory)
        """
        # FIXED: Option to NOT restore step counter during rollback
        # This prevents rollback loops where the model repeatedly diverges at the same step
        if restore_step:
            trainer.state.env_step = int(ckpt["state"]["env_step"])
            trainer.state.updates = int(ckpt["state"]["updates"])

        trainer.world_model_ensemble.load_state_dict(ckpt["world_model"])
        trainer.agent.load_state_dict(ckpt["agent"])
        if "target_critic" in ckpt:
            trainer.target_critic.load_state_dict(ckpt["target_critic"])
        else:
            # Older checkpoint: initialize target as copy of live critic.
            trainer.target_critic.load_state_dict(trainer.agent.critic.state_dict())

        # Add noise to weights if requested (helps avoid repeating same divergence)
        if noise_scale > 0.0:
            with torch.no_grad():
                for param in trainer.agent.actor.parameters():
                    param.add_(torch.randn_like(param) * noise_scale)

        if restore_optimizer:
            trainer.wm_opt.load_state_dict(ckpt["opt"]["wm_opt"])
            trainer.actor_opt.load_state_dict(ckpt["opt"]["actor_opt"])
            trainer.critic_opt.load_state_dict(ckpt["opt"]["critic_opt"])

        if restore_replay:
            trainer.replay.load_state_dict(ckpt["replay"])

        trainer.best_eval_return  = float(ckpt["metrics"]["best_eval_return"])
        trainer.best_train_return = float(ckpt["metrics"]["best_train_return"])
        if "return_scale" in ckpt.get("metrics", {}):
            trainer._return_scale = float(ckpt["metrics"]["return_scale"])

        if restore_step:  # Only restore RNG on full restore (not rollback)
            restore_rng_state(ckpt["rng"])
        
    def best_path(self) -> Optional[str]:
        p = os.path.join(self.ckpt_dir, "best.pt")
        return p if os.path.exists(p) else None
