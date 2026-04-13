"""
diagnose.py  —  health check for a ThesisWM checkpoint.

Loads a checkpoint, runs one WM forward pass and one AC update on a sampled batch,
and prints a structured report of all key statistics.

Usage:
    python scripts/diagnose.py --run_dir runs/<exp_name>
    python scripts/diagnose.py --run_dir runs/<exp_name> --checkpoint checkpoints/latest.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
from thesiswm.training.trainer import Trainer
from thesiswm.training.world_model_updater import update_world_model
from thesiswm.training.actor_critic_updater import update_actor_critic


# ── helpers ────────────────────────────────────────────────────────────────────

def _stats(name: str, x: torch.Tensor) -> None:
    x = x.detach().float()
    print(f"  {name:<35} mean={x.mean():+.4f}  std={x.std():6.4f}  "
          f"min={x.min():+.4f}  max={x.max():+.4f}  "
          f"nan={int(torch.isnan(x).any())}")


def _hdr(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="Path to a run directory (contains checkpoints/)")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint file name relative to run_dir/checkpoints/ (default: latest.pt)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_name = args.checkpoint or "latest.pt"
    ckpt_path = ckpt_dir / ckpt_name

    if not ckpt_path.exists():
        print(f"[ERROR] checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # ── load config snapshot ───────────────────────────────────────────────────
    cfg_path = run_dir / "config_snapshot.yaml"
    if not cfg_path.exists():
        print(f"[ERROR] config_snapshot.yaml not found in {run_dir}")
        sys.exit(1)
    cfg = OmegaConf.load(cfg_path)
    cfg.device = args.device
    cfg.training.resume = False   # don't try to load checkpoints again

    # config_snapshot.yaml is written before obs_dim/act_dim are resolved from the
    # env, so they are still null.  The checkpoint embeds the resolved cfg — read
    # the real dims from there so the model is built with the correct architecture.
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    if "cfg" in ckpt:
        ckpt_cfg = OmegaConf.create(ckpt["cfg"])
        cfg.world_model.obs_dim = ckpt_cfg.world_model.obs_dim
        cfg.world_model.act_dim = ckpt_cfg.world_model.act_dim
        print(f"  [dims from checkpoint]  obs_dim={cfg.world_model.obs_dim}  act_dim={cfg.world_model.act_dim}")
    else:
        print("  [WARN] checkpoint has no embedded cfg — dims may be wrong if config_snapshot has nulls")

    print(f"\n{'='*60}")
    print(f"  ThesisWM Diagnostic Report")
    print(f"  run_dir:    {run_dir}")
    print(f"  checkpoint: {ckpt_name}")
    print(f"  device:     {args.device}")
    print(f"{'='*60}")

    # ── build trainer (no env) ─────────────────────────────────────────────────
    trainer = Trainer(cfg, build_env=False)
    trainer.checkpointer.load_into(trainer, ckpt)
    env_step = trainer.state.env_step
    print(f"\n  env_step = {env_step:,}   updates = {trainer.state.updates:,}")
    print(f"  replay buffer size = {len(trainer.replay):,}")

    if len(trainer.replay) < int(cfg.replay.seq_len) + 2:
        print("\n[WARN] Replay buffer too small for a batch — cannot run forward passes.")
        return

    device = trainer.device

    # ── world model forward ────────────────────────────────────────────────────
    _hdr("WORLD MODEL")
    batch = trainer.replay.sample_sequences(int(cfg.replay.batch_size), int(cfg.replay.seq_len))
    obs      = torch.as_tensor(batch["obs"],      dtype=torch.float32, device=device)
    actions  = torch.as_tensor(batch["actions"],  dtype=torch.float32, device=device)
    rewards  = torch.as_tensor(batch["rewards"],  dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=device)

    kl_beta    = float(cfg.world_model.kl_beta)
    free_nats  = float(getattr(cfg.world_model, "free_nats", 1.0))
    kl_balance = float(getattr(cfg.world_model, "kl_balance", 0.8))

    wm = trainer.world_model_ensemble.models[0]
    with torch.no_grad():
        states, post_dists, prior_dists = wm.rssm.observe_sequence(obs, actions, device=device, sample=True)

        # KL per step
        kls_raw, kls_clamped = [], []
        for q, p in zip(post_dists, prior_dists):
            s1 = torch.exp(q.log_std); s2 = torch.exp(p.log_std)
            kl_d = (p.log_std - q.log_std + (s1**2 + (q.mean - p.mean)**2) / (2*s2**2 + 1e-8) - 0.5)
            kl_sum = kl_d.sum(dim=-1)
            kls_raw.append(kl_sum)
            kls_clamped.append(torch.clamp(kl_sum, min=free_nats))

        kl_raw    = torch.stack(kls_raw, dim=1)     # [B, T]
        kl_clamped = torch.stack(kls_clamped, dim=1)

        # Posterior z statistics
        z_all = torch.stack([s.z for s in states], dim=1).reshape(-1, states[0].z.shape[-1])
        print(f"\n  free_nats={free_nats}  kl_beta={kl_beta}  kl_balance={kl_balance}")
        _stats("KL raw (sum over dims)", kl_raw)
        _stats("KL clamped (free bits applied)", kl_clamped)
        print(f"  Fraction of timesteps where KL>free_nats: "
              f"{(kl_raw > free_nats).float().mean().item():.2%}  "
              f"← should be >50% for healthy encoding")

        _stats("posterior z", z_all)
        _stats("posterior z mean (per dim avg)", torch.stack([d.mean for d in post_dists]).mean(dim=(0,1)).unsqueeze(0))
        _stats("posterior z std (per dim avg)",  torch.exp(torch.stack([d.log_std for d in post_dists])).mean(dim=(0,1)).unsqueeze(0))
        _stats("prior z std (per dim avg)",      torch.exp(torch.stack([d.log_std for d in prior_dists])).mean(dim=(0,1)).unsqueeze(0))

        # Obs reconstruction
        T = len(states)
        h_nexts_list = [states[t].h for t in range(1, T)]
        h_T = wm.rssm.deter_step(states[-1].h, states[-1].z, actions[:, -1])
        h_nexts = torch.stack(h_nexts_list + [h_T], dim=1)
        h_flat = h_nexts.reshape(-1, h_nexts.shape[-1])
        prior_params = wm.rssm.prior_net(h_flat)
        mean_p, std_param = torch.chunk(prior_params, 2, dim=-1)
        std_p = F.softplus(std_param) + wm.rssm.min_std
        z_nexts = mean_p + torch.randn_like(mean_p) * std_p
        flat_next = torch.cat([h_flat, z_nexts], dim=-1)
        obs_pred = wm.predict_obs(flat_next).reshape(next_obs.shape)
        rew_pred = wm.predict_reward(flat_next).reshape(rewards.shape)

        _stats("obs reconstruction error", obs_pred - next_obs)
        _stats("obs MSE per step", ((obs_pred - next_obs)**2).mean(dim=-1))
        _stats("reward prediction", rew_pred)
        _stats("reward target", rewards)
        _stats("reward pred error", rew_pred - rewards)

    # ── actor-critic forward ────────────────────────────────────────────────────
    _hdr("ACTOR-CRITIC")
    B  = int(cfg.imagination.rollout_batch)
    ctx_len = int(getattr(cfg.imagination, "context_len", cfg.replay.seq_len))
    seq = trainer.replay.sample_sequences(B, ctx_len)
    obs_seq = torch.as_tensor(seq["obs"],     dtype=torch.float32, device=device)
    act_seq = torch.as_tensor(seq["actions"], dtype=torch.float32, device=device)

    H = int(cfg.imagination.horizon_fixed) if cfg.method.name.startswith("fixed_h") else \
        int(cfg.imagination.horizons[-1])   # use longest horizon for worst-case check
    gamma = float(cfg.agent.discount)
    lam   = float(cfg.agent.lambda_)

    wm.eval()
    for param in wm.parameters():
        param.requires_grad_(False)

    with torch.no_grad():
        states_ctx, _, _ = wm.rssm.observe_sequence(obs_seq, act_seq, device=device, sample=False)
        s = states_ctx[-1]

        feats, rews, logps, conts = [], [], [], []
        for _ in range(H):
            feat = wm.features(s)
            feats.append(feat)
            a, logp = trainer.agent.actor.sample(feat)
            logps.append(logp)
            s, _ = wm.rssm.imagine_step(s, a, sample=True)
            feat_next = wm.features(s)
            rews.append(wm.predict_reward(feat_next))
            conts.append(torch.sigmoid(wm.predict_continue_logit(feat_next)).clamp(0, 1))

        feat_last = wm.features(s)
        feats_t   = torch.stack(feats, dim=1)
        rewards_t = torch.stack(rews,  dim=1)
        logps_t   = torch.stack(logps, dim=1)
        conts_t   = torch.stack(conts, dim=1)

        flat_feats = feats_t.reshape(-1, feats_t.shape[-1])
        values_pred = trainer.agent.critic(flat_feats).reshape(B, H)
        v_last  = trainer.agent.critic(feat_last)
        values_ext = torch.cat([values_pred, v_last.unsqueeze(1)], dim=1)
        discounts = gamma * conts_t

        from thesiswm.training.imagination import lambda_returns
        target = lambda_returns(rewards=rewards_t, values=values_ext, discounts=discounts, lambda_=lam)
        advantage = target - values_pred

        print(f"\n  H={H}  gamma={gamma}  lambda={lam}")
        _stats("imagined reward",     rewards_t)
        _stats("lambda-return target", target)
        _stats("value prediction",    values_pred)
        _stats("advantage (tgt-val)", advantage)
        _stats("log_prob",            logps_t)

        # Policy distribution health
        import math as _math
        mean_t, log_std_t = trainer.agent.actor.forward(flat_feats)
        std_t = torch.exp(log_std_t)
        _stats("policy mean (pre-tanh)", mean_t)
        _stats("policy std",             std_t)
        _stats("policy log_std",         log_std_t)

        # Gaussian (pre-tanh) entropy — always positive, healthy range 1–5 nats for 3-dim actions.
        # This is what the trainer optimises; it cannot go negative unlike log_prob-based entropy.
        gaussian_entropy = (0.5 * (1.0 + _math.log(2.0 * _math.pi)) + log_std_t).sum(dim=-1).mean()
        # Log-prob entropy — can be NEGATIVE when |tanh(mean)| → 1 (saturation).
        # Negative values here indicate the policy is stuck at the action limits.
        logp_entropy = (-logps_t).mean()
        print(f"  {'Gaussian entropy (pre-tanh, nats)':<42} {float(gaussian_entropy):.4f}"
              f"  ← healthy: >1; bad: <0.5")
        print(f"  {'Log-prob entropy E[-log π] (nats)':<42} {float(logp_entropy):.4f}"
              f"  ← NEGATIVE means policy is SATURATED at action limits")

        # Check for saturation
        actions_t = torch.tanh(mean_t)
        saturated = (actions_t.abs() > 0.98).float().mean()
        pretanh_large = (mean_t.abs() > 2.0).float().mean()
        print(f"  {'saturated actions |a|>0.98':<42} {float(saturated):.2%}"
              f"  ← should be <5%")
        print(f"  {'large pre-tanh mean |μ|>2.0':<42} {float(pretanh_large):.2%}"
              f"  ← should be <10%; causes vanishing tanh gradient")

    wm.train()
    for param in wm.parameters():
        param.requires_grad_(True)

    # ── gradient norms (one real backward pass) ────────────────────────────────
    _hdr("GRADIENT NORMS (one real update step)")
    wm_loss, kl_loss, obs_loss, rew_loss, sigreg_loss = update_world_model(
        ensemble=trainer.world_model_ensemble,
        replay=trainer.replay,
        wm_opt=trainer.wm_opt,
        sigreg=trainer.sigreg,
        cfg=cfg,
        device=device,
        use_amp=trainer.use_amp,
    )
    print(f"\n  wm_loss={wm_loss:.4f}  kl={kl_loss:.4f}  obs={obs_loss:.4f}  "
          f"rew={rew_loss:.4f}  sigreg={sigreg_loss:.4f}")

    actor_loss, critic_loss, h_used, unc, _ = update_actor_critic(
        ensemble=trainer.world_model_ensemble,
        agent=trainer.agent,
        target_critic=trainer.target_critic,
        replay=trainer.replay,
        actor_opt=trainer.actor_opt,
        critic_opt=trainer.critic_opt,
        return_scale=trainer._return_scale,
        cfg=cfg,
        device=device,
        use_amp=trainer.use_amp,
        tb_fn=lambda tag, val, step: None,  # no-op logger for diagnostics
        step=trainer.state.env_step,
        critic_ema=trainer._critic_ema,
    )
    print(f"  actor_loss={actor_loss:.4f}  critic_loss={critic_loss:.6f}  "
          f"horizon_used={h_used}  uncertainty={unc:.4f}")

    # ── summary verdict ────────────────────────────────────────────────────────
    _hdr("VERDICT")
    kl_healthy     = (kl_raw > free_nats).float().mean().item() > 0.30
    gauss_ent_ok   = float(gaussian_entropy) > 1.0
    logp_ent_ok    = float(logp_entropy) > -2.0   # very negative = definitely saturated
    sat_ok         = float(saturated) < 0.10
    pretanh_ok     = float(pretanh_large) < 0.10
    rew_pred_ok    = float(((rew_pred - rewards)**2).mean().sqrt()) < 1.0
    print(f"  KL encoding active (>30% steps above free_nats):   {'OK' if kl_healthy   else 'PROBLEM — posterior may be collapsed'}")
    print(f"  Gaussian entropy > 1 nat:                          {'OK' if gauss_ent_ok else 'PROBLEM — std collapsed, low exploration'}")
    print(f"  Log-prob entropy > -2 (not heavily saturated):     {'OK' if logp_ent_ok  else 'PROBLEM — policy stuck at action limits'}")
    print(f"  Saturated actions < 10%:                           {'OK' if sat_ok       else 'PROBLEM — actions always near ±1'}")
    print(f"  Large pre-tanh mean < 10%:                         {'OK' if pretanh_ok   else 'PROBLEM — vanishing tanh gradient on mean'}")
    print(f"  Reward RMSE < 1.0:                                 {'OK' if rew_pred_ok  else 'PROBLEM — reward prediction inaccurate'}")
    print()


if __name__ == "__main__":
    main()
