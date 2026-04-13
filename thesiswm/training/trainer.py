"""
Trainer — orchestrates data collection, world model updates, and actor-critic updates.

Responsibilities:
  - Environment collection loop with persistent RSSM state
  - Coordinating WM and AC update calls
  - Evaluation, checkpointing, rollback, logging
"""
from __future__ import annotations

import copy
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from thesiswm.agents.actor_critic import ActorCritic
from thesiswm.data.replay_buffer import ReplayBuffer
from thesiswm.envs.make_env import make_env, make_vec_env
from thesiswm.models.rssm import EnsembleWorldModel, RSSMState
from thesiswm.training.actor_critic_updater import update_actor_critic, warmup_critic
from thesiswm.training.world_model_updater import update_world_model
from thesiswm.utils.checkpoint import Checkpointer
from thesiswm.utils.logger import TBLogger
from thesiswm.utils.sigreg import SIGReg


@dataclass
class TrainerState:
    env_step: int = 0
    updates:  int = 0


def load_checkpoint_into_trainer_state(trainer: "Trainer", ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    trainer.checkpointer.load_into(trainer, ckpt)
    return ckpt


class Trainer:
    def __init__(self, cfg: DictConfig, build_env: bool = True):
        self.cfg    = cfg
        self.device = torch.device(str(cfg.device))

        # ── directories ───────────────────────────────────────────────────────
        self.run_dir    = os.path.join(str(cfg.paths.runs_dir), str(cfg.exp_name))
        self.ckpt_dir   = os.path.join(self.run_dir, str(cfg.paths.ckpt_dirname))
        self.tb_dir     = os.path.join(self.run_dir, str(cfg.paths.tb_dirname))
        self.videos_dir = os.path.join(self.run_dir, str(cfg.paths.videos_dirname))
        for d in (self.run_dir, self.ckpt_dir, self.tb_dir, self.videos_dir):
            os.makedirs(d, exist_ok=True)

        with open(os.path.join(self.run_dir, "config_snapshot.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

        self.writer = SummaryWriter(log_dir=self.tb_dir)
        self.tb     = TBLogger(self.writer)

        # ── environment ───────────────────────────────────────────────────────
        self.num_envs = int(getattr(cfg.env, "num_envs", 1))
        self.env      = None
        if build_env:
            if self.num_envs > 1:
                self.env = make_vec_env(str(cfg.env.id), num_envs=self.num_envs, seed=int(cfg.seed))
            else:
                self.env = make_env(str(cfg.env.id), seed=int(cfg.seed), render_mode=None)

        # ── infer obs/act dims ────────────────────────────────────────────────
        if build_env:
            obs, _ = self.env.reset(seed=int(cfg.seed))
            obs    = np.asarray(obs, dtype=np.float32)
            if self.num_envs > 1:
                obs_dim = int(obs.shape[1])
                act_dim = int(self.env.single_action_space.shape[0])
            else:
                obs_dim = int(obs.shape[0])
                act_dim = int(self.env.action_space.shape[0])
        else:
            obs_dim = int(cfg.world_model.obs_dim) if cfg.world_model.obs_dim is not None else 0
            act_dim = int(cfg.world_model.act_dim) if cfg.world_model.act_dim is not None else 0

        self.cfg.world_model.obs_dim = obs_dim
        self.cfg.world_model.act_dim = act_dim

        # ── models ────────────────────────────────────────────────────────────
        self.replay = ReplayBuffer(int(cfg.replay.capacity), obs_dim=obs_dim, act_dim=act_dim)

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

        # EMA target critic — frozen copy that provides stable bootstrap values.
        self.target_critic = copy.deepcopy(self.agent.critic)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)
        self._critic_ema = float(getattr(cfg.agent, "critic_ema", 0.98))

        # ── optimizers ────────────────────────────────────────────────────────
        self.wm_opt     = torch.optim.Adam(self.world_model_ensemble.parameters(), lr=float(cfg.world_model.lr))
        self.actor_opt  = torch.optim.Adam(self.agent.actor.parameters(),          lr=float(cfg.agent.actor_lr))
        self.critic_opt = torch.optim.Adam(self.agent.critic.parameters(),         lr=float(cfg.agent.critic_lr))

        # ── misc ──────────────────────────────────────────────────────────────
        self.state        = TrainerState()
        self._return_scale = 1.0
        self.bad_eval_streak = 0

        self.use_amp = (self.device.type == "cuda") and bool(getattr(cfg, "use_amp", False))

        # SIGReg: pushes posterior z marginal toward N(0,I), complementing the KL term.
        self.sigreg = SIGReg(knots=17, num_proj=1024).to(self.device)

        self.checkpointer = Checkpointer(self.ckpt_dir)

        if bool(cfg.training.resume):
            latest = self.checkpointer.latest_path()
            if latest is not None:
                ckpt = torch.load(latest, map_location=self.device, weights_only=False)
                self.checkpointer.load_into(self, ckpt)
                print(f"[RESUME] loaded {latest} (env_step={self.state.env_step})", flush=True)
            else:
                print("[RESUME] no checkpoint found; starting fresh.")

    # ── evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_deterministic(self, episodes: int = 5) -> float:
        cfg    = self.cfg
        device = self.device
        wm0    = self.world_model_ensemble.models[0]
        amp    = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp)
        max_ep = int(getattr(cfg.training, "eval_max_episode_steps", 1000))

        env = make_env(cfg.env.id, seed=int(cfg.seed) + 12345, render_mode=None)
        returns = []
        for ep in range(int(episodes)):
            obs, _   = env.reset(seed=int(cfg.seed) + 20000 + ep)
            done = trunc = False
            ret  = 0.0
            ep_steps = 0
            state      = wm0.rssm.init_state(1, device)
            prev_action = torch.zeros(1, wm0.rssm.act_dim, device=device)

            while not (done or trunc) and ep_steps < max_ep:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with amp:
                    h            = wm0.rssm.deter_step(state.h, state.z, prev_action)
                    z            = wm0.rssm.posterior(h, obs_t).mean
                    feat         = torch.cat([h, z], dim=-1)
                    action       = self.agent.actor.mean_action(feat)
                state       = RSSMState(h=h.float(), z=z.float())
                prev_action = action.detach().float()
                action_np   = action.squeeze(0).float().cpu().numpy()
                obs, reward, done, trunc, _ = env.step(action_np)
                ret += float(reward)
                ep_steps += 1
            returns.append(ret)

        env.close()
        return float(np.mean(returns)) if returns else 0.0

    # ── main training loop ────────────────────────────────────────────────────

    def run(self):
        cfg = self.cfg
        assert self.env is not None, "Trainer.run requires build_env=True"

        N           = self.num_envs
        ep_returns  = np.zeros(N, dtype=np.float64)
        ep_lens     = np.zeros(N, dtype=np.int64)
        ep_count    = 0
        recent_returns = deque(maxlen=20)
        recent_lens    = deque(maxlen=20)
        log_ep_every = int(getattr(cfg.training, "log_episode_every", 10))

        total_steps  = int(cfg.training.total_steps)
        chunk_steps  = int(cfg.training.steps_per_chunk)
        start_step   = int(self.state.env_step)
        chunk_idx    = start_step // chunk_steps
        end_step     = min(total_steps, (chunk_idx + 1) * chunk_steps)

        eval_every              = int(getattr(cfg.training, "eval_every_steps",         2000))
        eval_episodes           = int(getattr(cfg.training, "eval_episodes",            5))
        best_min_delta          = float(getattr(cfg.training, "best_min_delta",         0.0))
        rollback_drop           = float(getattr(cfg.training, "rollback_drop",          10.0))
        rollback_patience       = int(getattr(cfg.training, "rollback_patience",        4))
        min_steps_before_rollback = int(getattr(cfg.training, "min_steps_before_rollback", 500))
        rollback_lr_scale       = float(getattr(cfg.training, "rollback_lr_scale",      1.0))
        rollback_noise          = float(getattr(cfg.training, "rollback_noise_scale",   0.01))

        train_rollback_check   = bool(getattr(cfg.training, "train_rollback_check",   True))
        train_rollback_drop    = float(getattr(cfg.training, "train_rollback_drop",   20.0))
        train_rollback_patience = int(getattr(cfg.training, "train_rollback_patience", 3))
        train_bad_streak = 0

        save_periodic     = bool(getattr(cfg.training, "save_periodic_checkpoints", False))
        save_latest_every = int(getattr(cfg.training, "save_latest_every_steps",   10000))
        save_best_train   = bool(getattr(cfg.training, "save_best_train",          True))
        train_best_delta  = float(getattr(cfg.training, "train_best_min_delta",    1.0))

        obs, _ = self.env.reset(seed=int(cfg.seed) + self.state.env_step)
        obs    = np.asarray(obs, dtype=np.float32)
        if N == 1:
            obs = obs[np.newaxis]

        self.best_eval_return  = self.evaluate_deterministic(eval_episodes)
        self.best_train_return = -1e9

        act_dim = (self.env.single_action_space if N > 1 else self.env.action_space).shape[0]

        # Persistent per-env RSSM state: prevents train/eval distribution mismatch.
        # The actor sees recurrent features during both collection and training.
        wm0     = self.world_model_ensemble.models[0]
        _rssm_h = torch.zeros(N, wm0.rssm.deter_dim,  device=self.device)
        _rssm_z = torch.zeros(N, wm0.rssm.latent_dim, device=self.device)
        _prev_a = torch.zeros(N, act_dim,              device=self.device)

        t0 = time.time()
        eval_time_total = 0.0

        while self.state.env_step < end_step:
            # ── collection ────────────────────────────────────────────────────
            for _ in range(int(cfg.training.collect_per_step)):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    _h = wm0.rssm.deter_step(_rssm_h, _rssm_z, _prev_a)
                    _z = wm0.rssm.posterior(_h, obs_t).mean

                if self.state.env_step < int(cfg.training.start_learning):
                    action   = np.random.uniform(-1.0, 1.0, size=(N, act_dim)).astype(np.float32)
                    action_t = torch.as_tensor(action, device=self.device)
                else:
                    with torch.no_grad():
                        action_t, _ = self.agent.actor.sample(torch.cat([_h, _z], dim=-1))
                    action = action_t.float().cpu().numpy().astype(np.float32)

                _rssm_h = _h.detach()
                _rssm_z = _z.detach()
                _prev_a = action_t.detach().float()

                if N > 1:
                    next_obs, rewards, dones, truncs, infos = self.env.step(action)
                    next_obs = np.asarray(next_obs, dtype=np.float32)
                    # Gymnasium AsyncVectorEnv autoreset: true final obs in infos["final_observation"].
                    _final = infos.get("final_observation") if isinstance(infos, dict) else None
                    if _final is not None:
                        next_obs = next_obs.copy()
                        for i in range(N):
                            if (bool(dones[i]) or bool(truncs[i])) and _final[i] is not None:
                                next_obs[i] = np.asarray(_final[i], dtype=np.float32)
                else:
                    next_obs, rewards, dones, truncs, _ = self.env.step(action[0])
                    next_obs = np.asarray(next_obs, dtype=np.float32)[np.newaxis]
                    rewards  = np.asarray([rewards],  dtype=np.float64)
                    dones    = np.asarray([dones])
                    truncs   = np.asarray([truncs])

                for i in range(N):
                    term_i  = bool(dones[i])
                    trunc_i = bool(truncs[i])
                    self.replay.add(obs[i], action[i], float(rewards[i]),
                                    term_i, trunc_i, next_obs[i])
                    ep_returns[i] += float(rewards[i])
                    ep_lens[i]    += 1

                    if term_i or trunc_i:
                        ep_count += 1
                        recent_returns.append(float(ep_returns[i]))
                        recent_lens.append(int(ep_lens[i]))

                        if ep_count % log_ep_every == 0:
                            self.tb.scalar("train/episode_return",  ep_returns[i], self.state.env_step)
                            self.tb.scalar("train/episode_length",  ep_lens[i],    self.state.env_step)
                            self.tb.scalar("train/return_mean_20",  float(np.mean(recent_returns)), self.state.env_step)
                            self.tb.scalar("train/len_mean_20",     float(np.mean(recent_lens)),    self.state.env_step)
                            self.tb.scalar("train/episodes",        ep_count,      self.state.env_step)

                        if save_best_train and len(recent_returns) >= 5 and self.state.env_step >= int(cfg.training.start_learning):
                            mean_train = float(np.mean(recent_returns))
                            if mean_train > (self.best_train_return + train_best_delta):
                                self.best_train_return = mean_train
                                self.checkpointer.save(
                                    self.checkpointer.make_weights_checkpoint(self),
                                    tag="best_train", make_latest=False,
                                )
                                self.tb.scalar("train/best_return", self.best_train_return, self.state.env_step)
                                print(f"[BEST_TRAIN] {self.best_train_return:.2f} at step {self.state.env_step}")
                                train_bad_streak = 0

                            if train_rollback_check and self.best_train_return > -1e8 and self.state.env_step >= min_steps_before_rollback:
                                if mean_train < (self.best_train_return - train_rollback_drop):
                                    train_bad_streak += 1
                                    self.tb.scalar("train/bad_streak", float(train_bad_streak), self.state.env_step)
                                    if train_bad_streak >= train_rollback_patience:
                                        self._do_rollback("best_train", rollback_noise, rollback_lr_scale)
                                        train_bad_streak = 0
                                else:
                                    train_bad_streak = max(0, train_bad_streak - 1)

                        ep_returns[i] = 0.0
                        ep_lens[i]    = 0
                        _rssm_h[i]    = 0.0
                        _rssm_z[i]    = 0.0
                        _prev_a[i]    = 0.0

                obs = next_obs
                self.state.env_step += N

                if self.state.env_step >= end_step:
                    break

            # ── updates ───────────────────────────────────────────────────────
            update_every = int(getattr(cfg.training, "update_every", 1))
            if (
                self.state.env_step >= int(cfg.training.start_learning)
                and len(self.replay) >= int(cfg.replay.seq_len) + 2
                and self.state.env_step % update_every == 0
            ):
                for _ in range(int(cfg.training.updates_per_step)):
                    wm_loss, kl_loss, obs_loss, rew_loss, sigreg_loss = update_world_model(
                        ensemble=self.world_model_ensemble,
                        replay=self.replay,
                        wm_opt=self.wm_opt,
                        sigreg=self.sigreg,
                        cfg=cfg,
                        device=self.device,
                        use_amp=self.use_amp,
                    )

                    if self.state.env_step >= int(cfg.training.actor_start_step):
                        actor_loss, critic_loss, horizon_used, unc_mean, self._return_scale = update_actor_critic(
                            ensemble=self.world_model_ensemble,
                            agent=self.agent,
                            target_critic=self.target_critic,
                            replay=self.replay,
                            actor_opt=self.actor_opt,
                            critic_opt=self.critic_opt,
                            return_scale=self._return_scale,
                            cfg=cfg,
                            device=self.device,
                            use_amp=self.use_amp,
                            tb_fn=self.tb.scalar,
                            step=int(self.state.env_step),
                            critic_ema=self._critic_ema,
                        )
                    else:
                        # WM warmup window: train critic without actor so it has a
                        # calibrated baseline ready when actor_start_step is reached.
                        warmup_critic(
                            ensemble=self.world_model_ensemble,
                            agent=self.agent,
                            target_critic=self.target_critic,
                            replay=self.replay,
                            critic_opt=self.critic_opt,
                            cfg=cfg,
                            device=self.device,
                            use_amp=self.use_amp,
                            tb_fn=self.tb.scalar,
                            step=int(self.state.env_step),
                            critic_ema=self._critic_ema,
                        )
                        actor_loss, critic_loss, horizon_used, unc_mean = 0.0, 0.0, 0, 0.0

                    self.state.updates += 1

                    if self.state.env_step % int(cfg.training.log_every_steps) == 0:
                        step = int(self.state.env_step)
                        train_time = max(1e-6, (time.time() - t0) - eval_time_total)
                        fps = (self.state.env_step - start_step) / train_time
                        self.tb.scalar("perf/fps",                  fps,          step)
                        self.tb.scalar("loss/world_model",          wm_loss,      step)
                        self.tb.scalar("loss/kl",                   kl_loss,      step)
                        self.tb.scalar("loss/obs",                  obs_loss,     step)
                        self.tb.scalar("loss/reward",               rew_loss,     step)
                        self.tb.scalar("loss/sigreg",               sigreg_loss,  step)
                        self.tb.scalar("loss/actor",                actor_loss,   step)
                        self.tb.scalar("loss/critic",               critic_loss,  step)
                        self.tb.scalar("imagination/horizon_used",  horizon_used, step)
                        self.tb.scalar("imagination/uncertainty_mean", unc_mean,  step)
                        print(
                            f"[PROGRESS] step={step:>7d}/{end_step}"
                            f"  fps={fps:5.0f}"
                            f"  wm={wm_loss:.3f}  actor={actor_loss:.4f}  critic={critic_loss:.3f}"
                            f"  best_eval={self.best_eval_return:.1f}",
                            flush=True,
                        )

            # ── periodic saves ────────────────────────────────────────────────
            if self.state.env_step % save_latest_every == 0:
                self.save_checkpoint(tag="latest", make_latest=False)

            if save_periodic and self.state.env_step % int(cfg.training.checkpoint_every_steps) == 0:
                self.save_checkpoint(tag=f"step_{self.state.env_step}")

            # ── eval ──────────────────────────────────────────────────────────
            if eval_every > 0 and self.state.env_step % eval_every == 0 and self.state.env_step >= int(cfg.training.start_learning):
                _eval_t0 = time.time()
                mean_ret = self.evaluate_deterministic(eval_episodes)
                eval_time_total += time.time() - _eval_t0

                step = int(self.state.env_step)
                self.tb.scalar("eval/return_mean", mean_ret,               step)
                self.tb.scalar("eval/best_return", self.best_eval_return,  step)
                self.tb.scalar("perf/eval_time_s", eval_time_total,        step)

                if mean_ret > (self.best_eval_return + best_min_delta):
                    self.best_eval_return = mean_ret
                    self.bad_eval_streak  = 0
                    self.checkpointer.save(self.checkpointer.make_checkpoint(self), tag="best", make_latest=False)
                    self.tb.scalar("eval/best_return", self.best_eval_return, step)

                if rollback_drop > 0 and self.state.env_step >= min_steps_before_rollback and self.best_eval_return > -1e8:
                    if mean_ret < (self.best_eval_return - rollback_drop):
                        self.bad_eval_streak += 1
                    else:
                        self.bad_eval_streak = 0

                    self.tb.scalar("eval/bad_streak", float(self.bad_eval_streak), step)

                    if self.bad_eval_streak >= rollback_patience:
                        self._do_rollback("best", rollback_noise, rollback_lr_scale)
                        self.bad_eval_streak = 0

        self.save_checkpoint(tag=f"step_{self.state.env_step}", make_latest=True)
        print(f"[DONE] env_step={self.state.env_step}/{total_steps}  run_dir={self.run_dir}", flush=True)
        self.close()

    # ── helpers ───────────────────────────────────────────────────────────────

    def save_checkpoint(self, tag: str, make_latest: bool = True):
        ckpt = self.checkpointer.make_checkpoint(self)
        self.checkpointer.save(ckpt, tag=tag, make_latest=make_latest)

    def _do_rollback(self, tag: str, noise_scale: float, lr_scale: float):
        path = os.path.join(self.checkpointer.ckpt_dir, f"{tag}.pt")
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        current_step = self.state.env_step
        ckpt_step    = ckpt["state"]["env_step"]
        self.checkpointer.load_into(
            self, ckpt,
            restore_step=False,
            restore_optimizer=False,
            restore_replay=False,
            noise_scale=noise_scale,
        )
        if lr_scale != 1.0:
            for pg in self.actor_opt.param_groups:
                pg["lr"] *= lr_scale
        print(f"[ROLLBACK/{tag}] step={current_step} ← ckpt_step={ckpt_step}, "
              f"noise={noise_scale:.4f}, lr_scale={lr_scale}", flush=True)
        self.tb.scalar("rollback/event",    1.0,              current_step)
        self.tb.scalar("rollback/from_step", float(ckpt_step), current_step)

    def close(self):
        if self.env is not None:
            self.env.close()
        self.writer.close()
