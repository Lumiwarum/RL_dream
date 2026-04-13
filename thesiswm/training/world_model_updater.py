"""
World model update step — extracted from Trainer for modularity.

Trains each ensemble member independently on the same batch, then does
a single optimizer step on all parameters together.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from thesiswm.data.replay_buffer import ReplayBuffer
from thesiswm.models.networks import DiagGaussian
from thesiswm.models.rssm import EnsembleWorldModel
from thesiswm.utils.sigreg import SIGReg


def _kl_balanced(
    post: DiagGaussian,
    prior: DiagGaussian,
    kl_balance: float,
    free_nats: float,
) -> torch.Tensor:
    """
    KL(posterior || prior) with KL-balance (DreamerV3 §A.1) and free-nats floor.

    kl_balance=0.8 means 80% of gradient goes toward the prior (prior learning),
    20% toward the posterior (representation learning). This prevents the
    posterior from collapsing to the prior too fast.

    free_nats is applied on the total KL (sum over dims), not per-dimension.
    Per-dim clamping with latent_dim=64 gives 64× the intended slack.
    """
    def _kl_per_dim(d1: DiagGaussian, d2: DiagGaussian) -> torch.Tensor:
        s1 = torch.exp(d1.log_std)
        s2 = torch.exp(d2.log_std)
        return (
            d2.log_std - d1.log_std
            + (s1.pow(2) + (d1.mean - d2.mean).pow(2)) / (2.0 * s2.pow(2) + 1e-8)
            - 0.5
        )

    q_sg = DiagGaussian(post.mean.detach(), post.log_std.detach())
    p_sg = DiagGaussian(prior.mean.detach(), prior.log_std.detach())

    kl_lhs = _kl_per_dim(q_sg, prior)   # posterior sg, prior has grad → prior learning
    kl_rhs = _kl_per_dim(post, p_sg)    # posterior has grad, prior sg → representation learning

    kl_per_dim = kl_balance * kl_lhs + (1.0 - kl_balance) * kl_rhs
    return torch.clamp(kl_per_dim.sum(dim=-1), min=free_nats)   # [B], total-KL floor


def update_world_model(
    ensemble: EnsembleWorldModel,
    replay: ReplayBuffer,
    wm_opt: torch.optim.Optimizer,
    sigreg: SIGReg,
    cfg: DictConfig,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float, float, float, float]:
    """
    Sample a batch, compute WM losses for every ensemble member, and step the optimizer.

    Returns: (total_loss, kl_loss, obs_loss, rew_loss, sigreg_loss) — mean over members.
    """
    batch = replay.sample_sequences(int(cfg.replay.batch_size), int(cfg.replay.seq_len))

    obs        = torch.as_tensor(batch["obs"],        dtype=torch.float32, device=device)  # [B,T,obs]
    actions    = torch.as_tensor(batch["actions"],    dtype=torch.float32, device=device)  # [B,T,act]
    rewards    = torch.as_tensor(batch["rewards"],    dtype=torch.float32, device=device)  # [B,T]
    next_obs   = torch.as_tensor(batch["next_obs"],   dtype=torch.float32, device=device)  # [B,T,obs]
    terminated = torch.as_tensor(batch["terminated"], dtype=torch.float32, device=device)  # [B,T]

    kl_beta       = float(cfg.world_model.kl_beta)
    kl_balance    = float(getattr(cfg.world_model, "kl_balance",    0.8))
    free_nats     = float(getattr(cfg.world_model, "free_nats",     1.0))
    sigreg_weight = float(getattr(cfg.world_model, "sigreg_weight", 0.0))
    cont_loss_w   = float(getattr(cfg.world_model, "cont_loss_weight", 0.5))

    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp)

    wm_losses, kl_losses, obs_losses, rew_losses, sigreg_losses = [], [], [], [], []

    wm_opt.zero_grad(set_to_none=True)

    for wm in ensemble.models:
        with amp_ctx:
            states, post_dists, prior_dists = wm.rssm.observe_sequence(
                obs, actions, device=device, sample=True
            )

            # Build next-h for all T steps efficiently:
            # h_nexts[t] = GRU(h_t, z_t, a_t), which equals states[t+1].h for t < T-1.
            # Only the final step needs an extra GRU call.
            T = len(states)
            h_nexts_list = [states[t].h for t in range(1, T)]
            h_T = wm.rssm.deter_step(states[-1].h, states[-1].z, actions[:, -1])
            h_nexts = torch.stack(h_nexts_list + [h_T], dim=1)  # [B,T,deter_dim]

            # Vectorised predictions over all T steps
            h_flat      = h_nexts.reshape(-1, h_nexts.shape[-1])
            prior_params = wm.rssm.prior_net(h_flat)
            mean_p, std_param = torch.chunk(prior_params, 2, dim=-1)
            std_p  = F.softplus(std_param) + wm.rssm.min_std
            z_next = mean_p + torch.randn_like(mean_p) * std_p          # reparameterised
            flat_next = torch.cat([h_flat, z_next], dim=-1)             # [B*T, feat_dim]

            obs_pred   = wm.predict_obs(flat_next).reshape(next_obs.shape)
            rew_pred   = wm.predict_reward(flat_next).reshape(rewards.shape)
            cont_logits = wm.predict_continue_logit(flat_next).reshape(terminated.shape)

            obs_loss = F.mse_loss(obs_pred, next_obs)
            rew_loss = F.mse_loss(rew_pred, rewards)

            # cont target: 1 for non-terminal, 0 for genuine falls (not truncation).
            cont_target = 1.0 - terminated
            done_loss = cont_loss_w * F.binary_cross_entropy_with_logits(cont_logits, cont_target)

            # KL with balanced gradients and free-nats floor
            kl = torch.stack([
                _kl_balanced(post_dists[t], prior_dists[t], kl_balance, free_nats)
                for t in range(len(post_dists))
            ], dim=1).mean()  # [B,T] → scalar

            loss = obs_loss + rew_loss + done_loss + kl_beta * kl

            # SIGReg: characteristic-function Gaussian regularizer pushing posterior z toward N(0,I).
            if sigreg_weight > 0.0:
                z_all = torch.stack([s.z for s in states], dim=1).reshape(-1, states[0].z.shape[-1])
                sigreg_loss = sigreg(z_all)
                loss = loss + sigreg_weight * sigreg_loss
            else:
                sigreg_loss = torch.zeros(1, device=device)

        loss.backward()

        wm_losses.append(loss.detach().item())
        kl_losses.append(kl.detach().item())
        obs_losses.append(obs_loss.detach().item())
        rew_losses.append(rew_loss.detach().item())
        sigreg_losses.append(sigreg_loss.detach().item())

    torch.nn.utils.clip_grad_norm_(ensemble.parameters(), float(cfg.world_model.grad_clip))
    wm_opt.step()

    return (
        float(np.mean(wm_losses)),
        float(np.mean(kl_losses)),
        float(np.mean(obs_losses)),
        float(np.mean(rew_losses)),
        float(np.mean(sigreg_losses)),
    )
