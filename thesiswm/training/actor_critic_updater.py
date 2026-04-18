"""
Actor-critic update step — extracted from Trainer for modularity.

Implements the DreamerV3-style imagination-based actor-critic:
- Sample context from replay, infer posterior latent state
- Roll out H steps in imagination (WM frozen, actor gradients flow through)
- Compute lambda-returns as actor targets (pathwise gradient)
- Train critic with Huber loss on symlog targets
- Train actor with normalized advantage + Gaussian entropy bonus
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from thesiswm.agents.actor_critic import ActorCritic
from thesiswm.data.replay_buffer import ReplayBuffer
from thesiswm.models.rssm import EnsembleWorldModel
from thesiswm.training.imagination import decide_horizon, lambda_returns
from thesiswm.utils.symlog import symexp, symlog


def choose_horizon(
    ensemble: EnsembleWorldModel,
    cfg: DictConfig,
    obs0: torch.Tensor,
    act0: torch.Tensor,
    device: torch.device,
    ema_obs_loss: float = 1.0,
) -> Tuple[int, float]:
    """
    Select imagination horizon based on method config.

    Fixed baselines use a single pre-set H.

    Adaptive methods have two uncertainty signals:
      use_obs_loss_unc=false (default): ensemble latent disagreement — stable per-seed,
        selects a horizon once and keeps it for the whole run.
      use_obs_loss_unc=true: EMA WM obs_loss — varies throughout training (high early,
        low late), giving genuine within-run horizon switching as the WM converges.
    """
    name = str(cfg.method.name)
    if name.startswith("fixed_h"):
        h_map = {"fixed_h5": 5, "fixed_h15": 15, "fixed_h20": 20}
        return h_map.get(name, int(cfg.imagination.horizon_fixed)), 0.0

    horizons = tuple(int(x) for x in cfg.imagination.horizons)

    if bool(getattr(cfg.method, "use_obs_loss_unc", False)):
        # Dynamic signal: EMA of WM obs reconstruction loss.
        # High obs_loss → WM still learning → short horizon (less error amplification).
        # Low obs_loss → WM converged → long horizon (better credit assignment).
        h_low, h_mid, h_high = horizons
        thresh_h = float(getattr(cfg.method, "obs_loss_thresh_high", 0.30))
        thresh_m = float(getattr(cfg.method, "obs_loss_thresh_mid",  0.18))
        if ema_obs_loss > thresh_h:
            return h_low,  float(ema_obs_loss)
        if ema_obs_loss > thresh_m:
            return h_mid,  float(ema_obs_loss)
        return h_high, float(ema_obs_loss)

    # Default: ensemble disagreement (original signal)
    H, u = decide_horizon(
        ensemble=ensemble,
        obs=obs0,
        action=act0,
        metric=str(cfg.method.uncertainty_metric),
        thresh_high=float(cfg.method.thresh_high),
        thresh_mid=float(cfg.method.thresh_mid),
        horizons=horizons,
        device=device,
    )
    return int(H), float(u.mean().item())


def update_actor_critic(
    ensemble: EnsembleWorldModel,
    agent: ActorCritic,
    target_critic: torch.nn.Module,
    replay: ReplayBuffer,
    actor_opt: torch.optim.Optimizer,
    critic_opt: torch.optim.Optimizer,
    return_scale: float,
    cfg: DictConfig,
    device: torch.device,
    use_amp: bool,
    tb_fn,          # callable(tag, value, step) — TBLogger.scalar
    step: int,
    critic_ema: float = 0.98,
    ema_obs_loss: float = 1.0,
) -> Tuple[float, float, int, float, float]:
    """
    One imagination-based actor-critic update.

    Returns: (actor_loss, critic_loss, horizon_used, unc_mean, new_return_scale)
    """
    B = int(cfg.imagination.rollout_batch)
    ctx_len = int(getattr(cfg.imagination, "context_len", cfg.replay.seq_len))
    seq = replay.sample_sequences(B, ctx_len)

    obs_seq = torch.as_tensor(seq["obs"],     dtype=torch.float32, device=device)  # [B,T,obs]
    act_seq = torch.as_tensor(seq["actions"], dtype=torch.float32, device=device)  # [B,T,act]

    obs0 = obs_seq[:, -1]
    act0 = act_seq[:, -1]
    H, unc_mean = choose_horizon(ensemble, cfg, obs0, act0, device, ema_obs_loss=ema_obs_loss)

    # Pick one ensemble member at random for imagination rollouts.
    # WM is frozen during actor-critic updates; actor gradients flow through
    # imagined transitions via the reparameterisation trick (pathwise gradient).
    wm_id = np.random.randint(0, len(ensemble.models))
    wm = ensemble.models[wm_id]
    wm.eval()
    saved_req = [p.requires_grad for p in wm.parameters()]
    for p in wm.parameters():
        p.requires_grad_(False)

    use_symlog = bool(getattr(cfg.agent, "symlog_critic", True))
    slc        = float(getattr(cfg.agent, "symlog_clamp", 5.0))
    gamma      = float(cfg.agent.discount)
    lam        = float(cfg.agent.lambda_)
    cont_floor = float(getattr(cfg.world_model, "cont_disc_floor", 0.9))

    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

    try:
        with amp_ctx:
            # Warm up recurrent state via the real context sequence (posterior).
            # This gives the actor a meaningful starting latent instead of zeros.
            states, _, _ = wm.rssm.observe_sequence(obs_seq, act_seq, device=device, sample=False)
            s = states[-1]

            feats, rewards_im, logps = [], [], []
            done_probs = []
            actor_means_log, actor_log_stds_log = [], []

            feat = wm.features(s)
            for _ in range(H):
                feats.append(feat)
                a, logp = agent.actor.sample(feat)
                logps.append(logp)

                # Collect distribution stats without building a gradient graph (logging only).
                with torch.no_grad():
                    m_t, ls_t = agent.actor.forward(feat)
                    actor_means_log.append(m_t)
                    actor_log_stds_log.append(ls_t)

                s, _ = wm.rssm.imagine_step(s, a, sample=True)
                feat = wm.features(s)
                rewards_im.append(wm.predict_reward(feat).clamp(-5.0, 5.0))
                done_probs.append(torch.sigmoid(wm.predict_continue_logit(feat)).clamp(0.0, 1.0))

            feat_last = feat  # bootstrap feature

            feats_t    = torch.stack(feats,     dim=1)  # [B,H,feat]
            rewards_t  = torch.stack(rewards_im, dim=1)  # [B,H]
            logps_t    = torch.stack(logps,      dim=1)  # [B,H]
            conts_t    = torch.stack(done_probs, dim=1)  # [B,H]

            feats_det  = feats_t.detach()
            flat_feats = feats_det.reshape(-1, feats_det.shape[-1])  # [B*H, feat]

            # Live critic predicts in symlog space; targets are also in symlog space.
            values_flat = agent.critic(flat_feats)
            values_pred = values_flat.reshape(B, H)  # [B,H], symlog space if use_symlog

            # EMA target critic: stable bootstrap values prevent V→target→V feedback loop.
            with torch.no_grad():
                targ_raw      = target_critic(flat_feats).reshape(B, H)
                targ_raw_last = target_critic(feat_last)
                targ_vals     = symexp(targ_raw.clamp(-slc, slc))      if use_symlog else targ_raw
                targ_v_last   = symexp(targ_raw_last.clamp(-slc, slc)) if use_symlog else targ_raw_last

                _log_scalar(tb_fn, "value/bootstrap", targ_v_last, step)

                values_ext = torch.cat([targ_vals, targ_v_last.unsqueeze(1)], dim=1)  # [B,H+1]

            # Per-step discounts: clamp cont from below so a pessimistic WM never kills the horizon.
            discounts = gamma * conts_t.clamp(min=cont_floor).detach()  # [B,H]

            # Lambda-returns in original (non-symlog) space — rewards are in original scale.
            target_actor = lambda_returns(
                rewards=rewards_t,
                values=values_ext,
                discounts=discounts,
                lambda_=lam,
            )  # [B,H]

            # Critic trains in symlog space.
            target_critic_tgt = (symlog(target_actor) if use_symlog else target_actor).detach()

        # ── diagnostics ──────────────────────────────────────────────────────
        tb_fn("imagine/horizon_used",       float(H),       step)
        tb_fn("imagine/uncertainty_mean",   float(unc_mean), step)
        tb_fn("imagine/cont_prob_mean",     float(conts_t.mean().item()), step)
        tb_fn("imagine/cont_prob_min",      float(conts_t.min().item()),  step)

        _log_stats(tb_fn, "imagine/reward_pred",  rewards_t,    step)
        _log_stats(tb_fn, "value/pred_symlog",    values_pred,  step)
        _log_stats(tb_fn, "value/pred",
                   symexp(values_pred.detach()) if use_symlog else values_pred, step)
        _log_stats(tb_fn, "value/target_symlog",  target_critic_tgt, step)
        _log_stats(tb_fn, "value/target",         target_actor.detach(), step)
        _log_stats(tb_fn, "policy/logp",          logps_t,       step)
        _log_stats(tb_fn, "imagine/cont_prob",    conts_t,       step)

        _log_nan(tb_fn, "reward_pred", rewards_t,        step)
        _log_nan(tb_fn, "target",      target_critic_tgt, step)
        _log_nan(tb_fn, "value_pred",  values_pred,      step)
        _log_nan(tb_fn, "logp",        logps_t,          step)

        _mean_log   = torch.cat(actor_means_log,    dim=0)
        _log_std_log = torch.cat(actor_log_stds_log, dim=0)
        tb_fn("policy/std_mean", _log_std_log.exp().mean().item(), step)
        _log_stats(tb_fn, "policy/log_std", _log_std_log, step)
        _log_stats(tb_fn, "policy/mean",    _mean_log,    step)

        # ── critic update (Huber loss) ─────────────────────────────────────
        critic_opt.zero_grad(set_to_none=True)
        critic_loss = F.smooth_l1_loss(values_pred, target_critic_tgt)
        critic_loss.backward()
        c_gnorm = torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
        tb_fn("grad/critic_norm", float(c_gnorm), step)
        critic_opt.step()

        # EMA target critic update
        with torch.no_grad():
            for p_live, p_tgt in zip(agent.critic.parameters(), target_critic.parameters()):
                p_tgt.data.lerp_(p_live.data, 1.0 - critic_ema)

        # ── actor update ───────────────────────────────────────────────────
        actor_opt.zero_grad(set_to_none=True)

        baseline  = (symexp(values_pred.clamp(-slc, slc)) if use_symlog else values_pred).detach()
        advantage = target_actor - baseline
        _log_stats(tb_fn, "value/advantage", advantage, step)

        # DreamerV3 §A.4 return scale: p95 - p5 of lambda-return targets (not advantage std).
        # advantage.std() is sticky at large values when the critic overshoots — ACTOR_GRAD_TINY.
        # target_actor is bounded by WM output (rewards + clamped bootstrap) so this stays sane
        # even when the critic drifts, preventing the SPIKE_THEN_DROP death spiral.
        scale_max = float(getattr(cfg.agent, "return_scale_max", 100.0))
        with torch.no_grad():
            ret_5  = torch.quantile(target_actor.float(), 0.05)
            ret_95 = torch.quantile(target_actor.float(), 0.95)
            new_sample = float((ret_95 - ret_5).clamp(min=1.0).item())
            new_sample = min(new_sample, scale_max)
        return_scale = min(0.95 * return_scale + 0.05 * new_sample, scale_max)
        tb_fn("value/return_scale", return_scale, step)

        # Fresh forward pass WITH gradients (cached feats from rollout have no grad graph).
        mean, log_std = agent.actor.forward(flat_feats)

        actor_loss = -(advantage / return_scale).mean()

        if float(cfg.agent.entropy_coef) != 0.0:
            # Analytic Gaussian entropy: avoids -E[log π] going negative when policy saturates.
            gaussian_entropy = (0.5 * (1.0 + math.log(2.0 * math.pi)) + log_std).sum(dim=-1).mean()
            actor_loss = actor_loss - float(cfg.agent.entropy_coef) * gaussian_entropy
            tb_fn("policy/entropy_gaussian", float(gaussian_entropy.item()), step)
            tb_fn("policy/entropy_logp",     float((-logps_t).mean().item()), step)

        pretanh_coef = float(getattr(cfg.agent, "pretanh_reg_coef", 0.0))
        if pretanh_coef > 0.0:
            raw_mean = torch.atanh(mean.clamp(-1 + 1e-6, 1 - 1e-6))
            pretanh_reg = raw_mean.pow(2).mean()
            actor_loss  = actor_loss + pretanh_coef * pretanh_reg
            tb_fn("policy/pretanh_mean_reg", float(pretanh_reg.item()), step)

        actor_loss.backward()
        a_gnorm = torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
        tb_fn("grad/actor_norm", float(a_gnorm), step)
        actor_opt.step()

    finally:
        wm.train()
        for p, req in zip(wm.parameters(), saved_req):
            p.requires_grad_(req)

    return (
        float(actor_loss.item()),
        float(critic_loss.item()),
        int(H),
        float(unc_mean),
        float(return_scale),
    )


def warmup_critic(
    ensemble: EnsembleWorldModel,
    agent: ActorCritic,
    target_critic: torch.nn.Module,
    replay: ReplayBuffer,
    critic_opt: torch.optim.Optimizer,
    cfg: DictConfig,
    device: torch.device,
    use_amp: bool,
    tb_fn,
    step: int,
    critic_ema: float = 0.98,
    ema_obs_loss: float = 1.0,
) -> float:
    """
    Critic-only update — used during the WM warmup window (start_learning → actor_start_step).

    Trains the critic on lambda returns from imagined rollouts without touching the actor.
    This gives the critic a head start so that when the actor begins training, the baseline
    is already reasonably calibrated and advantages are meaningful (not dominated by bootstrap
    from an untrained critic).

    Returns: critic_loss
    """
    B = int(cfg.imagination.rollout_batch)
    ctx_len = int(getattr(cfg.imagination, "context_len", cfg.replay.seq_len))
    seq = replay.sample_sequences(B, ctx_len)

    obs_seq = torch.as_tensor(seq["obs"],     dtype=torch.float32, device=device)
    act_seq = torch.as_tensor(seq["actions"], dtype=torch.float32, device=device)

    # Use the same horizon-selection logic as the full actor-critic update so that
    # the critic is calibrated for the horizon length it will see during actor training.
    name = str(cfg.method.name)
    if name.startswith("fixed_h"):
        H_map = {"fixed_h5": 5, "fixed_h15": 15, "fixed_h20": 20}
        H     = H_map.get(name, int(cfg.imagination.horizon_fixed))
    else:
        obs0 = obs_seq[:, -1]
        act0 = act_seq[:, -1]
        H, _ = choose_horizon(ensemble, cfg, obs0, act0, device, ema_obs_loss=ema_obs_loss)

    wm_id = np.random.randint(0, len(ensemble.models))
    wm    = ensemble.models[wm_id]
    wm.eval()
    saved_req = [p.requires_grad for p in wm.parameters()]
    for p in wm.parameters():
        p.requires_grad_(False)

    use_symlog  = bool(getattr(cfg.agent, "symlog_critic", True))
    slc         = float(getattr(cfg.agent, "symlog_clamp", 5.0))
    gamma       = float(cfg.agent.discount)
    lam         = float(cfg.agent.lambda_)
    cont_floor  = float(getattr(cfg.world_model, "cont_disc_floor", 0.9))

    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

    try:
        with amp_ctx:
            with torch.no_grad():
                states, _, _ = wm.rssm.observe_sequence(obs_seq, act_seq, device=device, sample=False)
                s = states[-1]

                feats, rewards_im, done_probs = [], [], []
                feat = wm.features(s)
                for _ in range(H):
                    feats.append(feat)
                    a, _ = agent.actor.sample(feat)
                    s, _ = wm.rssm.imagine_step(s, a, sample=True)
                    feat = wm.features(s)
                    rewards_im.append(wm.predict_reward(feat).clamp(-5.0, 5.0))
                    done_probs.append(torch.sigmoid(wm.predict_continue_logit(feat)).clamp(0.0, 1.0))

                feat_last  = feat
                feats_t    = torch.stack(feats,     dim=1)
                rewards_t  = torch.stack(rewards_im, dim=1)
                conts_t    = torch.stack(done_probs, dim=1)

                flat_feats  = feats_t.detach().reshape(-1, feats_t.shape[-1])
                targ_raw      = target_critic(flat_feats).reshape(B, H)
                targ_raw_last = target_critic(feat_last)
                targ_vals     = symexp(targ_raw.clamp(-slc, slc))      if use_symlog else targ_raw
                targ_v_last   = symexp(targ_raw_last.clamp(-slc, slc)) if use_symlog else targ_raw_last
                values_ext    = torch.cat([targ_vals, targ_v_last.unsqueeze(1)], dim=1)

                discounts = gamma * conts_t.clamp(min=cont_floor)
                target_actor = lambda_returns(rewards=rewards_t, values=values_ext,
                                              discounts=discounts, lambda_=lam)
                target_tgt = (symlog(target_actor) if use_symlog else target_actor).detach()

        # Critic forward WITH grad
        values_pred = agent.critic(flat_feats).reshape(B, H)

        critic_opt.zero_grad(set_to_none=True)
        critic_loss = F.smooth_l1_loss(values_pred, target_tgt)
        critic_loss.backward()
        c_gnorm = torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
        tb_fn("grad/critic_norm_warmup", float(c_gnorm), step)
        critic_opt.step()

        with torch.no_grad():
            for p_live, p_tgt in zip(agent.critic.parameters(), target_critic.parameters()):
                p_tgt.data.lerp_(p_live.data, 1.0 - critic_ema)

        tb_fn("loss/critic_warmup", float(critic_loss.item()), step)
        return float(critic_loss.item())

    finally:
        wm.train()
        for p, req in zip(wm.parameters(), saved_req):
            p.requires_grad_(req)


# ── Logging helpers ────────────────────────────────────────────────────────────

def _log_stats(tb_fn, tag: str, x: torch.Tensor, step: int) -> None:
    if x is None or not torch.is_tensor(x) or x.numel() == 0:
        return
    with torch.no_grad():
        x_f = x.detach().float()
        tb_fn(f"{tag}_mean",   x_f.mean().item(),                  step)
        tb_fn(f"{tag}_std",    x_f.std(unbiased=False).item(),     step)
        tb_fn(f"{tag}_maxabs", x_f.abs().max().item(),             step)


def _log_scalar(tb_fn, tag: str, x: torch.Tensor, step: int) -> None:
    if x is None or not torch.is_tensor(x) or x.numel() == 0:
        return
    with torch.no_grad():
        tb_fn(f"{tag}_mean",   x.detach().float().mean().item(),   step)
        tb_fn(f"{tag}_maxabs", x.detach().float().abs().max().item(), step)


def _log_nan(tb_fn, tag: str, x: torch.Tensor, step: int) -> None:
    if x is None or not torch.is_tensor(x) or x.numel() == 0:
        return
    with torch.no_grad():
        flag = float(torch.isnan(x).any().item() or torch.isinf(x).any().item())
        tb_fn(f"debug/has_nan_{tag}", flag, step)
