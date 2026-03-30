from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: F401
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from collections import deque
import math

from thesiswm.agents.actor_critic import ActorCritic
from thesiswm.data.replay_buffer import ReplayBuffer
from thesiswm.envs.make_env import make_env, make_vec_env
from thesiswm.models.rssm import EnsembleWorldModel, RSSMState
from thesiswm.training.imagination import decide_horizon, lambda_returns
from thesiswm.utils.checkpoint import Checkpointer
from thesiswm.utils.logger import TBLogger
from thesiswm.utils.rng import capture_rng_state, restore_rng_state
from thesiswm.utils.sigreg import SIGReg
from thesiswm.models.networks import DiagGaussian


@dataclass
class TrainerState:
    env_step: int = 0
    updates: int = 0


def load_checkpoint_into_trainer_state(trainer: "Trainer", ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    trainer.checkpointer.load_into(trainer, ckpt)
    return ckpt


class Trainer:
    def __init__(self, cfg: DictConfig, build_env: bool = True):
        self.cfg = cfg
        self.device = torch.device(str(cfg.device))

        # paths
        self.run_dir = os.path.join(str(cfg.paths.runs_dir), str(cfg.exp_name))
        self.ckpt_dir = os.path.join(self.run_dir, str(cfg.paths.ckpt_dirname))
        self.tb_dir = os.path.join(self.run_dir, str(cfg.paths.tb_dirname))
        self.videos_dir = os.path.join(self.run_dir, str(cfg.paths.videos_dirname))
        self.best_ckpt_path = os.path.join(self.ckpt_dir, "best.pt")
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)

        # persist config snapshot
        with open(os.path.join(self.run_dir, "config_snapshot.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

        self.writer = SummaryWriter(log_dir=self.tb_dir)
        self.tb = TBLogger(self.writer)

        # env
        self.num_envs = int(getattr(cfg.env, "num_envs", 1))
        self.env = None
        if build_env:
            if self.num_envs > 1:
                self.env = make_vec_env(str(cfg.env.id), num_envs=self.num_envs, seed=int(cfg.seed))
            else:
                self.env = make_env(str(cfg.env.id), seed=int(cfg.seed), render_mode=None)

        # infer dims
        if build_env:
            obs, _ = self.env.reset(seed=int(cfg.seed))
            obs = np.asarray(obs, dtype=np.float32)
            if self.num_envs > 1:
                obs_dim = int(obs.shape[1])
                act_dim = int(self.env.single_action_space.shape[0])
            else:
                obs_dim = int(obs.shape[0])
                act_dim = int(self.env.action_space.shape[0])
        else:
            obs_dim = int(cfg.world_model.obs_dim) if cfg.world_model.obs_dim is not None else 0
            act_dim = int(cfg.world_model.act_dim) if cfg.world_model.act_dim is not None else 0

        # write back to cfg (in-memory)
        self.cfg.world_model.obs_dim = obs_dim
        self.cfg.world_model.act_dim = act_dim

        # replay
        self.replay = ReplayBuffer(int(cfg.replay.capacity), obs_dim=obs_dim, act_dim=act_dim)

        # models
        self.world_model_ensemble = EnsembleWorldModel(
            n=int(cfg.world_model.ensemble_size),
            obs_dim=obs_dim,
            act_dim=act_dim,
            latent_dim=int(cfg.world_model.latent_dim),
            deter_dim=int(cfg.world_model.deter_dim),
            hidden_dim=int(cfg.world_model.hidden_dim),
        ).to(self.device)

        feat_dim = int(cfg.world_model.deter_dim + cfg.world_model.latent_dim)
        self.agent = ActorCritic(
            feat_dim=feat_dim,
            act_dim=act_dim,
            actor_hidden=int(cfg.agent.actor_hidden),
            critic_hidden=int(cfg.agent.critic_hidden),
        ).to(self.device)

        # optimizers
        self.wm_opt = torch.optim.Adam(self.world_model_ensemble.parameters(), lr=float(cfg.world_model.lr))
        self.actor_opt = torch.optim.Adam(self.agent.actor.parameters(), lr=float(cfg.agent.actor_lr))
        self.critic_opt = torch.optim.Adam(self.agent.critic.parameters(), lr=float(cfg.agent.critic_lr))

        self.state = TrainerState()
        self.bad_eval_streak = 0
        # EMA of return std used to normalize the actor loss scale across training.
        # Prevents gradient magnitude from exploding 100x as the agent improves.
        self._return_scale = 1.0

        # AMP: bfloat16 autocast — same dynamic range as float32, no GradScaler needed.
        # Enabled only on CUDA; silently off on CPU.
        self.use_amp = (self.device.type == "cuda") and bool(getattr(cfg, "use_amp", False))

        # SIGReg: characteristic-function Gaussian regularizer (Le-WM, 2025).
        # Pushes posterior z marginal toward N(0,I), complementing the KL term.
        self.sigreg = SIGReg(knots=17, num_proj=1024).to(self.device)

        # checkpointing
        self.checkpointer = Checkpointer(self.ckpt_dir)

        # resume if requested
        if bool(cfg.training.resume):
            # Always resume from latest.pt (most recent progress).
            # best.pt may be at a much earlier step and would cause chunk_idx to
            # recompute incorrectly, making end_step wrong on resume.
            latest = self.checkpointer.latest_path()
            if latest is not None:
                ckpt = torch.load(latest, map_location=self.device, weights_only=False)
                self.checkpointer.load_into(self, ckpt)
                print(f"[RESUME] loaded {latest} (env_step={self.state.env_step})", flush=True)
            else:
                print("[RESUME] no checkpoint found; starting fresh.")

    def close(self):
        if self.env is not None:
            self.env.close()
        self.writer.close()
        
    @torch.no_grad()
    def evaluate_deterministic(self, episodes: int = 5) -> float:
        cfg = self.cfg
        device = self.device
        wm0 = self.world_model_ensemble.models[0]
        _amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp)
        max_ep_steps = int(getattr(cfg.training, "eval_max_episode_steps", 1000))

        env = make_env(cfg.env.id, seed=int(cfg.seed) + 12345, render_mode=None)
        returns = []

        for ep in range(int(episodes)):
            obs, _ = env.reset(seed=int(cfg.seed) + 20000 + ep)
            done = False
            trunc = False
            ret = 0.0
            ep_steps = 0

            # Maintain RSSM state across steps: avoids re-initializing from zeros every step.
            # Previous action starts as zeros (no history at episode start).
            state = wm0.rssm.init_state(1, device)
            prev_action = torch.zeros(1, wm0.rssm.act_dim, device=device)

            while not (done or trunc) and ep_steps < max_ep_steps:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with _amp_ctx:
                    # One GRU step using previous z + previous action, then posterior from obs
                    h = wm0.rssm.deter_step(state.h, state.z, prev_action)
                    post_dist = wm0.rssm.posterior(h, obs_t)
                    z = post_dist.mean  # deterministic (mean) for eval
                    feat = torch.cat([h, z], dim=-1)
                    action = self.agent.actor.mean_action(feat)
                # Keep state in float32 so next GRU step inputs are consistent
                state = RSSMState(h=h.float(), z=z.float())

                action_np = action.squeeze(0).float().cpu().numpy()
                prev_action = action.detach().float()

                obs, reward, done, trunc, _ = env.step(action_np)
                ret += float(reward)
                ep_steps += 1

            returns.append(ret)

        env.close()
        return float(np.mean(returns)) if returns else 0.0


    def run(self):
        cfg = self.cfg
        assert self.env is not None, "Trainer.run requires build_env=True"
        
        N = self.num_envs  # number of parallel envs
        ep_returns = np.zeros(N, dtype=np.float64)
        ep_lens    = np.zeros(N, dtype=np.int64)
        ep_count = 0
        recent_returns = deque(maxlen=20)
        recent_lens = deque(maxlen=20)
        log_ep_every = int(getattr(cfg.training, "log_episode_every", 10))

        total_steps = int(cfg.training.total_steps)
        chunk_steps = int(cfg.training.steps_per_chunk)
        start_step = int(self.state.env_step)
        chunk_idx = start_step // chunk_steps
        end_step = min(total_steps, (chunk_idx + 1) * chunk_steps)
        # periodic deterministic eval + best checkpoint + rollback
        eval_every = int(getattr(cfg.training, "eval_every_steps", 2000))
        eval_episodes = int(getattr(cfg.training, "eval_episodes", 5))
        best_min_delta = float(getattr(cfg.training, "best_min_delta", 0.0))
        rollback_drop = float(getattr(cfg.training, "rollback_drop", 10.0))  # drop in return
        rollback_patience = int(getattr(cfg.training, "rollback_patience", 4))
        min_steps_before_rollback = int(getattr(cfg.training, "min_steps_before_rollback", 500))
        rollback_lr_scale = float(getattr(cfg.training, "rollback_lr_scale", 1.0))

        # Hybrid approach: training-based early warning system
        train_rollback_check = bool(getattr(cfg.training, "train_rollback_check", True))
        train_rollback_drop = float(getattr(cfg.training, "train_rollback_drop", 20.0))
        train_rollback_patience = int(getattr(cfg.training, "train_rollback_patience", 3))
        train_bad_streak = 0

        _save_periodic     = bool(getattr(cfg.training, "save_periodic_checkpoints", False))
        _save_latest_every = int(getattr(cfg.training, "save_latest_every_steps", 10000))

        obs, _ = self.env.reset(seed=int(cfg.seed) + self.state.env_step)
        obs = np.asarray(obs, dtype=np.float32)
        if N == 1:
            obs = obs[np.newaxis]  # [1, obs_dim] — unify shape for single env

        self.best_eval_return = self.evaluate_deterministic(eval_episodes)
        self.best_train_return = -1e9  # Track best training performance
        save_best_train = bool(getattr(cfg.training, "save_best_train", True))  # Enable intermediate best checkpoints
        train_best_min_delta = float(getattr(cfg.training, "train_best_min_delta", 1.0))  # Minimum improvement to save

        t0 = time.time()
        eval_time_total = 0.0  # cumulative seconds spent in evaluate_deterministic; excluded from FPS
        while self.state.env_step < end_step:
            # Collect transition(s) — vectorized over N parallel envs.
            # obs shape: [N, obs_dim]; VectorEnv auto-resets on episode end.
            act_dim = (self.env.single_action_space if N > 1 else self.env.action_space).shape[0]
            for _ in range(int(cfg.training.collect_per_step)):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)  # [N, obs_dim]
                if self.state.env_step < int(cfg.training.start_learning):
                    action = np.random.uniform(-1.0, 1.0, size=(N, act_dim)).astype(np.float32)
                else:
                    with torch.no_grad():
                        action_t = self.agent.act_stochastic_from_obs(obs_t, self.world_model_ensemble, device=self.device)
                    action = action_t.float().cpu().numpy().astype(np.float32)  # [N, act_dim]

                if N > 1:
                    next_obs, rewards, dones, truncs, _ = self.env.step(action)
                else:
                    next_obs, rewards, dones, truncs, _ = self.env.step(action[0])
                    next_obs = np.asarray(next_obs, dtype=np.float32)[np.newaxis]
                    rewards  = np.asarray([rewards], dtype=np.float64)
                    dones    = np.asarray([dones])
                    truncs   = np.asarray([truncs])

                next_obs = np.asarray(next_obs, dtype=np.float32)

                for i in range(N):
                    terminated_i = bool(dones[i])
                    truncated_i  = bool(truncs[i])
                    self.replay.add(obs[i], action[i], float(rewards[i]),
                                    terminated_i, truncated_i, next_obs[i])
                    ep_returns[i] += float(rewards[i])
                    ep_lens[i]    += 1

                    if terminated_i or truncated_i:
                        ep_count += 1
                        recent_returns.append(float(ep_returns[i]))
                        recent_lens.append(int(ep_lens[i]))

                        if (ep_count % log_ep_every) == 0:
                            self.tb.scalar("train/episode_return", ep_returns[i], self.state.env_step)
                            self.tb.scalar("train/episode_length", ep_lens[i], self.state.env_step)
                            self.tb.scalar("train/return_mean_20", float(np.mean(recent_returns)), self.state.env_step)
                            self.tb.scalar("train/len_mean_20", float(np.mean(recent_lens)), self.state.env_step)
                            self.tb.scalar("train/episodes", ep_count, self.state.env_step)

                        # Save best training checkpoint based on recent performance
                        if save_best_train and len(recent_returns) >= 5 and self.state.env_step >= int(cfg.training.start_learning):
                            mean_train_ret = float(np.mean(recent_returns))
                            if mean_train_ret > (self.best_train_return + train_best_min_delta):
                                self.best_train_return = mean_train_ret
                                ckpt = self.checkpointer.make_weights_checkpoint(self)
                                self.checkpointer.save(ckpt, tag="best_train", make_latest=False)
                                self.tb.scalar("train/best_return", self.best_train_return, self.state.env_step)
                                print(f"[BEST_TRAIN] New best training return: {self.best_train_return:.2f} at step {self.state.env_step}")
                                train_bad_streak = 0

                            # Training-based early rollback detection
                            if train_rollback_check and self.best_train_return > -1e8 and self.state.env_step >= min_steps_before_rollback:
                                if mean_train_ret < (self.best_train_return - train_rollback_drop):
                                    train_bad_streak += 1
                                    self.tb.scalar("train/bad_streak", float(train_bad_streak), self.state.env_step)

                                    if train_bad_streak >= train_rollback_patience:
                                        best_train_path = os.path.join(self.checkpointer.ckpt_dir, "best_train.pt")
                                        if os.path.exists(best_train_path):
                                            rollback_noise = float(getattr(cfg.training, "rollback_noise_scale", 0.01))
                                            ckpt = torch.load(best_train_path, map_location=self.device, weights_only=False)
                                            current_step = self.state.env_step
                                            ckpt_step = ckpt["state"]["env_step"]
                                            self.checkpointer.load_into(
                                                self, ckpt,
                                                restore_step=False,
                                                restore_optimizer=False,
                                                restore_replay=False,
                                                noise_scale=rollback_noise
                                            )
                                            if rollback_lr_scale != 1.0:
                                                for pg in self.actor_opt.param_groups:
                                                    pg["lr"] *= rollback_lr_scale
                                            train_bad_streak = 0
                                            print(f"[TRAIN_ROLLBACK] step={current_step}, loaded best_train from step={ckpt_step}, "
                                                  f"train_return dropped from {self.best_train_return:.2f} to {mean_train_ret:.2f}")
                                            self.tb.scalar("train/rollback", 1.0, self.state.env_step)
                                            self.tb.scalar("train/rollback_from_step", float(ckpt_step), self.state.env_step)
                                else:
                                    if train_bad_streak > 0:
                                        train_bad_streak = max(0, train_bad_streak - 1)

                        ep_returns[i] = 0.0
                        ep_lens[i]    = 0
                        # VectorEnv auto-resets; no manual env.reset() needed.

                obs = next_obs
                self.state.env_step += N

                if self.state.env_step >= end_step:
                    break

            # Updates
            update_every = int(getattr(cfg.training, "update_every", 1))
            
            if (self.state.env_step >= int(cfg.training.start_learning)
                and len(self.replay) >= int(cfg.replay.seq_len) + 2
                and (self.state.env_step % update_every == 0)
                ):
                for _ in range(int(cfg.training.updates_per_step)):
                    wm_loss, kl_loss, obs_loss, rew_loss, sigreg_loss = self.update_world_model()

                    if self.state.env_step >= cfg.training.actor_start_step:
                        actor_loss, critic_loss, horizon_used, unc_mean = self.update_actor_critic()
                    else:
                        actor_loss, critic_loss, horizon_used, unc_mean = 0.0, 0.0, 0, 0.0


                    self.state.updates += 1

                    # logging
                    if self.state.env_step % int(cfg.training.log_every_steps) == 0:
                        # Exclude eval time from FPS so the metric reflects true training throughput,
                        # not the growing cost of eval episodes as the agent improves.
                        train_time = max(1e-6, (time.time() - t0) - eval_time_total)
                        fps = (self.state.env_step - start_step) / train_time
                        self.tb.scalar("perf/fps", fps, self.state.env_step)
                        self.tb.scalar("loss/world_model", wm_loss, self.state.env_step)
                        self.tb.scalar("loss/kl", kl_loss, self.state.env_step)
                        self.tb.scalar("loss/obs", obs_loss, self.state.env_step)
                        self.tb.scalar("loss/reward", rew_loss, self.state.env_step)
                        self.tb.scalar("loss/sigreg", sigreg_loss, self.state.env_step)
                        self.tb.scalar("loss/actor", actor_loss, self.state.env_step)
                        self.tb.scalar("loss/critic", critic_loss, self.state.env_step)
                        self.tb.scalar("imagination/horizon_used", horizon_used, self.state.env_step)
                        self.tb.scalar("imagination/uncertainty_mean", unc_mean, self.state.env_step)
                        # Terse progress line — picked up by orchestrator log-tailer
                        print(
                            f"[PROGRESS] step={self.state.env_step:>7d}/{end_step}"
                            f"  fps={fps:5.0f}"
                            f"  wm={wm_loss:.3f}  actor={actor_loss:.4f}  critic={critic_loss:.3f}"
                            f"  best_eval={self.best_eval_return:.1f}",
                            flush=True,
                        )

            # Overwrite latest.pt periodically — no new files, just crash-recovery insurance.
            # With 200k-step chunks and no periodic step files, a crash would lose the whole chunk.
            if self.state.env_step % _save_latest_every == 0:
                self.save_checkpoint(tag="latest", make_latest=False)

            # Named step snapshots (disabled by default, set save_periodic_checkpoints=true to enable)
            if (_save_periodic
                    and self.state.env_step % int(cfg.training.checkpoint_every_steps) == 0):
                self.save_checkpoint(tag=f"step_{self.state.env_step}")

            if eval_every > 0 and (self.state.env_step % eval_every == 0) and (self.state.env_step >= int(cfg.training.start_learning)):
                _eval_t0 = time.time()
                mean_ret = self.evaluate_deterministic(eval_episodes)
                eval_time_total += time.time() - _eval_t0
                self.tb.scalar("eval/return_mean", mean_ret, self.state.env_step)
                self.tb.scalar("eval/best_return", self.best_eval_return, self.state.env_step)
                self.tb.scalar("perf/eval_time_s", eval_time_total, self.state.env_step)

                # Save best
                if mean_ret > (self.best_eval_return + best_min_delta):
                    self.best_eval_return = mean_ret
                    self.bad_eval_streak = 0
                    ckpt = self.checkpointer.make_checkpoint(self)
                    self.checkpointer.save(ckpt, tag="best", make_latest=False)
                    self.tb.scalar("eval/best_return", self.best_eval_return, self.state.env_step)
                    

                # Rollback logic
                if rollback_drop > 0 and (self.state.env_step >= min_steps_before_rollback) and (self.best_eval_return > -1e8):
                    if mean_ret < (self.best_eval_return - rollback_drop):
                        self.bad_eval_streak += 1
                    else:
                        self.bad_eval_streak = 0

                    self.tb.scalar("eval/bad_streak", float(self.bad_eval_streak), self.state.env_step)

                    if self.bad_eval_streak >= rollback_patience:
                        best_path = os.path.join(self.checkpointer.ckpt_dir, "best.pt")
                        if os.path.exists(best_path):
                            # FIXED ROLLBACK LOOP: Don't restore step counter, add noise to weights
                            # This prevents infinite loop where model diverges at same step repeatedly
                            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
                            
                            # Get rollback config
                            rollback_noise = float(getattr(cfg.training, "rollback_noise_scale", 0.01))
                            
                            # Save current step before rollback
                            current_step = self.state.env_step
                            ckpt_step = ckpt["state"]["env_step"]
                            
                            # Load weights but keep current step/optimizer/replay
                            self.checkpointer.load_into(
                                self, ckpt,
                                restore_step=False,        # Keep current step (prevent loop!)
                                restore_optimizer=False,   # Keep current optimizer momentum
                                restore_replay=False,      # Keep current replay buffer
                                noise_scale=rollback_noise # Add noise to avoid identical trajectory
                            )

                            # Optional: reduce actor lr to prevent re-collapse
                            if rollback_lr_scale != 1.0:
                                for pg in self.actor_opt.param_groups:
                                    pg["lr"] *= rollback_lr_scale

                            self.bad_eval_streak = 0
                            # Log rollback with details
                            print(f"[ROLLBACK] step={current_step}, loaded weights from step={ckpt_step}, "
                                  f"noise_scale={rollback_noise:.4f}, lr_scale={rollback_lr_scale}", flush=True)
                            self.tb.scalar("eval/rollback", 1.0, self.state.env_step)
                            self.tb.scalar("eval/rollback_from_step", float(ckpt_step), self.state.env_step)
                        else:
                            self.tb.scalar("eval/rollback", 0.0, self.state.env_step)

        # Final save for chunk
        self.save_checkpoint(tag=f"step_{self.state.env_step}", make_latest=True)
        print(f"[DONE] chunk complete: env_step={self.state.env_step}/{total_steps}  run_dir={self.run_dir}", flush=True)
        self.close()

    def save_checkpoint(self, tag: str, make_latest: bool = True):
        ckpt = self.checkpointer.make_checkpoint(self)
        self.checkpointer.save(ckpt, tag=tag, make_latest=make_latest)

    def update_world_model(self) -> Tuple[float, float, float, float]:
        cfg = self.cfg
        batch = self.replay.sample_sequences(int(cfg.replay.batch_size), int(cfg.replay.seq_len))

        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)              # [B,T,obs_dim]
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)      # [B,T,act_dim]
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)      # [B,T]
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)   # [B,T,obs_dim]
        terminated = torch.as_tensor(batch["terminated"], dtype=torch.float32, device=self.device)  # [B,T]
        is_last    = torch.as_tensor(batch["is_last"],    dtype=torch.float32, device=self.device)  # [B,T] terminated OR truncated

        kl_beta = float(cfg.world_model.kl_beta)
        kl_balance = float(getattr(cfg.world_model, "kl_balance", 0.8))
        free_nats = float(getattr(cfg.world_model, "free_nats", 1.0))
        sigreg_weight = float(getattr(cfg.world_model, "sigreg_weight", 0.0))

        wm_losses, kl_losses, obs_losses, rew_losses, done_losses, sigreg_losses = [], [], [], [], [], []

        self.wm_opt.zero_grad(set_to_none=True)
        
        def _kl_per_dim(d1: DiagGaussian, d2: DiagGaussian) -> torch.Tensor:
            """KL(d1 || d2) per latent dimension — returns [B, latent_dim]."""
            s1 = torch.exp(d1.log_std)
            s2 = torch.exp(d2.log_std)
            return (
                d2.log_std - d1.log_std
                + (s1.pow(2) + (d1.mean - d2.mean).pow(2)) / (2.0 * s2.pow(2) + 1e-8)
                - 0.5
            )

        _amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp)
        for wm in self.world_model_ensemble.models:
            with _amp_ctx:
                states, post_dists, prior_dists = wm.rssm.observe_sequence(obs, actions, device=self.device, sample=True)

                # Reuse h from observe_sequence (states[t+1].h == imagine_step(states[t], a_t).h).
                # Only the final step needs one extra GRU forward.
                T = len(states)
                h_nexts_list = [states[t].h for t in range(1, T)]
                h_T = wm.rssm.deter_step(states[-1].h, states[-1].z, actions[:, -1])
                h_nexts = torch.stack(h_nexts_list + [h_T], dim=1)  # [B, T, deter_dim]
                # Vectorised prior over all T next h's — avoids T-1 redundant GRU calls
                h_flat = h_nexts.reshape(-1, h_nexts.shape[-1])          # [B*T, deter_dim]
                prior_params = wm.rssm.prior_net(h_flat)
                mean_p, std_param = torch.chunk(prior_params, 2, dim=-1)
                std_p = F.softplus(std_param) + wm.rssm.min_std
                z_nexts = mean_p + torch.randn_like(mean_p) * std_p      # reparameterised sample
                flat_next = torch.cat([h_flat, z_nexts], dim=-1)         # [B*T, feat_dim]
                cont_logits = wm.predict_continue_logit(flat_next).reshape(terminated.shape)

                # ---- IMPORTANT FIX: predict next_obs, not obs ----
                obs_pred = wm.predict_obs(flat_next).reshape(next_obs.shape)      # [B,T,obs_dim]
                rew_pred = wm.predict_reward(flat_next).reshape(rewards.shape)    # [B,T]

                obs_loss = F.mse_loss(obs_pred, next_obs)
                rew_loss = F.mse_loss(rew_pred, rewards)
                cont_target = 1.0 - is_last   # episode ended for ANY reason (fall OR timeout)
                pos = cont_target.mean().clamp_min(1e-6)
                neg = (1.0 - cont_target).mean().clamp_min(1e-6)
                pos_weight = torch.clamp((neg / pos).detach(), max=10.0)
                done_loss = F.binary_cross_entropy_with_logits(
                    cont_logits, cont_target, pos_weight=pos_weight
                )

                # KL across time, mean over batch/time.
                # Free bits applied to the TOTAL KL (sum over latent dims), not per-dimension.
                # Per-dimension clamping with free_nats=1.0 and latent_dim=64 gives a 64-nat
                # floor — all dims stay clamped, gradients are zero, posterior collapses.
                # DreamerV3 convention: max(KL_sum, free_nats) keeps exactly 1 nat of slack.
                kls = []
                for t in range(len(post_dists)):
                    q = post_dists[t]
                    p = prior_dists[t]
                    qg = DiagGaussian(q.mean, q.log_std)
                    pg = DiagGaussian(p.mean, p.log_std)
                    q_sg = DiagGaussian(qg.mean.detach(), qg.log_std.detach())
                    p_sg = DiagGaussian(pg.mean.detach(), pg.log_std.detach())
                    kl_lhs_per_dim = _kl_per_dim(q_sg, pg)
                    kl_rhs_per_dim = _kl_per_dim(qg,   p_sg)
                    kl_per_dim = kl_balance * kl_lhs_per_dim + (1.0 - kl_balance) * kl_rhs_per_dim
                    kl_t = torch.clamp(kl_per_dim.sum(dim=-1), min=free_nats)  # total cap
                    kls.append(kl_t)
                kl = torch.stack(kls, dim=1).mean()

                loss = obs_loss + rew_loss + done_loss + kl_beta * kl

                # SIGReg: push posterior z marginal toward N(0,I).
                # Collect z samples from all time steps: list of [B, latent_dim] → [B*T, latent_dim]
                if sigreg_weight > 0.0:
                    z_all = torch.stack([s.z for s in states], dim=1).reshape(-1, states[0].z.shape[-1])
                    sigreg_loss = self.sigreg(z_all)
                    loss = loss + sigreg_weight * sigreg_loss
                else:
                    sigreg_loss = torch.zeros(1, device=self.device)

            loss.backward()

            wm_losses.append(loss.detach().item())
            kl_losses.append(kl.detach().item())
            obs_losses.append(obs_loss.detach().item())
            rew_losses.append(rew_loss.detach().item())
            done_losses.append(done_loss.detach().item())
            sigreg_losses.append(sigreg_loss.detach().item())

        torch.nn.utils.clip_grad_norm_(self.world_model_ensemble.parameters(), float(cfg.world_model.grad_clip))
        self.wm_opt.step()

        return (
            float(np.mean(wm_losses)),
            float(np.mean(kl_losses)),
            float(np.mean(obs_losses)),
            float(np.mean(rew_losses)),
            float(np.mean(sigreg_losses)),
        )


    def _choose_horizon(self, obs0: torch.Tensor, act0: torch.Tensor) -> Tuple[int, float]:
        cfg = self.cfg
        name = str(cfg.method.name)
        if name.startswith("fixed_h"):
            # parse fixed horizon
            if name == "fixed_h5":
                return 5, 0.0
            if name == "fixed_h15":
                return 15, 0.0
            if name == "fixed_h20":
                return 20, 0.0
            # fallback to config
            return int(cfg.imagination.horizon_fixed), 0.0

        # adaptive
        H, u = decide_horizon(
            ensemble=self.world_model_ensemble,
            obs=obs0,
            action=act0,
            metric=str(cfg.method.uncertainty_metric),
            thresh_high=float(cfg.method.thresh_high),
            thresh_mid=float(cfg.method.thresh_mid),
            horizons=tuple(int(x) for x in cfg.imagination.horizons),
            device=self.device,
        )
        return int(H), float(u.mean().item())

    def update_actor_critic(self) -> Tuple[float, float, int, float]:
        """
        Dreamer-like actor-critic using imagined rollouts.

        Key Dreamer property:
        - World model parameters are frozen, BUT gradients flow through imagination w.r.t. actions
        - Actor is optimized via pathwise gradients of imagened returns (not REINFORCE).
        """
        cfg = self.cfg
        device = self.device
        B = int(cfg.imagination.rollout_batch)

        # --- sample a context sequence (same kind you use for world model training) ---
        ctx_len = int(getattr(self.cfg.imagination, "context_len", self.cfg.replay.seq_len))
        seq = self.replay.sample_sequences(B, ctx_len)

        obs_seq = torch.as_tensor(seq["obs"], dtype=torch.float32, device=self.device)         # [B,T,obs]
        act_seq = torch.as_tensor(seq["actions"], dtype=torch.float32, device=self.device)     # [B,T,act]

        # Use the last real transition (obs_t, act_t) for uncertainty-based horizon selection
        obs0 = obs_seq[:, -1]   # [B, obs]
        act0 = act_seq[:, -1]   # [B, act]
        H, unc_mean = self._choose_horizon(obs0, act0)

        # Pick a world model
        wm_id = np.random.randint(0, len(self.world_model_ensemble.models))
        wm = self.world_model_ensemble.models[wm_id]
        wm.eval()
        _prev_req = [p.requires_grad for p in wm.parameters()]
        for p in wm.parameters():
            p.requires_grad_(False)
            
            
        try:

            # Infer starting latent state from the *context posterior* (not a 1-step state)
            gamma = float(cfg.agent.discount)
            lam = float(cfg.agent.lambda_)

            _ac_amp = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp)
            with _ac_amp:
                states, _, _ = wm.rssm.observe_sequence(obs_seq, act_seq, device=self.device, sample=False)
                s = states[-1]

                feats: list[torch.Tensor] = []
                rewards: list[torch.Tensor] = []
                logps: list[torch.Tensor] = []
                done_probs: list[torch.Tensor] = []
                actor_means: list[torch.Tensor] = []
                actor_log_stds: list[torch.Tensor] = []

                # --- imagination rollout ---
                for _ in range(H):
                    # IMPORTANT: no torch.no_grad() here; WM params are frozen, but we need grads wrt actions.
                    feat = wm.features(s)  # [B, feat_dim]
                    feats.append(feat)

                    # actor samples action; grads flow into actor and through dynamics via action
                    a, logp = self.agent.actor.sample(feat)
                    logps.append(logp)

                    with torch.no_grad():
                        mean_t, log_std_t = self.agent.actor.forward(feat)
                        actor_means.append(mean_t)
                        actor_log_stds.append(log_std_t)

                    # Imagine next state; DO NOT detach action (we need pathwise gradient wrt action)
                    s, _ = wm.rssm.imagine_step(s, a, sample=True)
                    feat_next = wm.features(s)
                    r = wm.predict_reward(feat_next)  # [B]
                    rewards.append(r)

                    # Dreamer: continue probability (P(not terminal))
                    cont_logit = wm.predict_continue_logit(feat_next)
                    cont_prob = torch.sigmoid(cont_logit).clamp(0.0, 1.0)
                    done_probs.append(cont_prob)

                feat_last = wm.features(s)

                # stack rollout tensors
                feats_t = torch.stack(feats, dim=1)        # [B,H,feat]
                rewards_t = torch.stack(rewards, dim=1)    # [B,H]
                logps_t = torch.stack(logps, dim=1)        # [B,H]
                conts_t = torch.stack(done_probs, dim=1)   # [B,H]

                # critic uses detached feats (no grad into WM)
                feats_det = feats_t.detach()
                flat_feats = feats_det.reshape(-1, feats_det.shape[-1])  # [B*H, feat]

                # ---- critic predictions ----
                # 1) Predictions WITH grad (for critic loss)
                values_flat_pred = self.agent.critic(flat_feats)          # [B*H] requires grad
                values_pred = values_flat_pred.reshape(B, H)              # [B,H]
                # 2) Bootstrap value WITHOUT grad (stabilizes target and avoids critic->target leakage)
                with torch.no_grad():
                    v_last = self.agent.critic(feat_last)                 # [B]

                # Build bootstrap values tensor (detached)
                values_ext_det = torch.cat([values_pred.detach(), v_last.detach().unsqueeze(1)], dim=1)  # [B,H+1]

                # per-step discounts should not backprop through continue head (stabilizes)
                discounts = gamma * conts_t.detach()  # [B,H]

                ## Actor target WITH grad through rewards_t only (values are detached)
                target_actor = lambda_returns(
                    rewards=rewards_t,
                    values=values_ext_det,
                    discounts=discounts,
                    lambda_=lam,
                )  # [B,H]

                # Critic target is detached
                target = target_actor.detach()

            # --- diagnostics ---
            step = int(self.state.env_step)
            self.tb.scalar("imagine/horizon_used", int(H), step)
            self.tb.scalar("imagine/uncertainty_mean", float(unc_mean), step)

            self._tb_stats("imagine/reward_pred", rewards_t, step)
            self._tb_stats("value/target", target, step)
            self._tb_stats("value/pred", values_pred, step)
            self._tb_stats("policy/logp", logps_t, step)
            self._tb_stats("imagine/cont_prob", conts_t, step)

            self._tb_flag_nan("reward_pred", rewards_t, step)
            self._tb_flag_nan("target", target, step)
            self._tb_flag_nan("value_pred", values_pred, step)
            self._tb_flag_nan("logp", logps_t, step)


            # FIXED ISSUE #4: Use cached actor distribution params instead of redundant forward pass
            # Previously called self.agent.actor.forward(flat_feats) again, wasting 15-20% compute
            # Now use the cached values from imagination loop
            mean = torch.cat(actor_means, dim=0)  # [B*H, act_dim]
            log_std = torch.cat(actor_log_stds, dim=0)  # [B*H, act_dim]
            std = torch.exp(log_std)
            self.tb.scalar("policy/std_mean", std.mean().item(), step)
            self._tb_stats("policy/log_std", log_std, step)
            self._tb_stats("policy/mean", mean, step)


            # --- critic update (Huber) ---
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss = F.smooth_l1_loss(values_pred, target)
            critic_loss.backward()
            self.tb.scalar("grad/critic_norm", self._grad_norm(self.agent.critic), step)
            torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), 10.0)
            self.critic_opt.step()

            # --- actor update ---
            self.actor_opt.zero_grad(set_to_none=True)

            # Normalize returns so gradient scale stays stable as the agent improves.
            # Without this, gradient norm grows ~100x from early (return≈5) to late (return≈500)
            # causing the policy collapse / rollback cycles seen at 169 score.
            with torch.no_grad():
                batch_std = float(target_actor.float().std().item())
                self._return_scale = 0.99 * self._return_scale + 0.01 * max(batch_std, 1.0)
            self.tb.scalar("value/return_scale", self._return_scale, step)

            # Dreamer pathwise objective: maximize normalized imagined returns.
            # (WM params frozen; gradients flow through action -> dynamics -> rewards/values)
            actor_loss = -(target_actor / self._return_scale).mean()
            if float(cfg.agent.entropy_coef) != 0.0:
                entropy_est = (-logps_t).mean()
                actor_loss = actor_loss - float(cfg.agent.entropy_coef) * entropy_est

            actor_loss.backward()
            self.tb.scalar("grad/actor_norm", self._grad_norm(self.agent.actor), step)
            torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), 10.0)
            self.actor_opt.step()
            
        finally:
            wm.train()
            for p, req in zip(wm.parameters(), _prev_req):
                p.requires_grad_(req)
    
        return float(actor_loss.item()), float(critic_loss.item()), int(H), float(unc_mean)


    def _tb_stats(self, tag: str, x: torch.Tensor, step: int) -> None:
        """Log basic stats of a tensor to TensorBoard."""
        if x is None:
            return
        with torch.no_grad():
            if not torch.is_tensor(x):
                return
            x = x.detach()
            if x.numel() == 0:
                return
            x_f = x.float()
            self.tb.scalar(f"{tag}_mean", x_f.mean().item(), step)
            self.tb.scalar(f"{tag}_std", x_f.std(unbiased=False).item(), step)
            self.tb.scalar(f"{tag}_maxabs", x_f.abs().max().item(), step)

    def _tb_flag_nan(self, tag: str, x: torch.Tensor, step: int) -> None:
        with torch.no_grad():
            if x is None or (not torch.is_tensor(x)) or x.numel() == 0:
                return
            flag = float(torch.isnan(x).any().item() or torch.isinf(x).any().item())
            self.tb.scalar(f"debug/has_nan_{tag}", flag, step)

    def _grad_norm(self, module: torch.nn.Module) -> float:
        """Compute global grad norm (L2) over parameters that have gradients."""
        total = 0.0
        for p in module.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            total += float((g * g).sum().item())
        return float(total ** 0.5)