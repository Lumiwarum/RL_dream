"""DreamerV3 agent with optional ensemble world models and adaptive horizon.

Stripped from r2dreamer (NM512/r2dreamer): removed R2-Dreamer / InfoNCE /
DreamerPro representation losses.  Added:
  - WorldModel class (encoder + RSSM + decoder + reward/cont heads)
  - WorldModelEnsemble (N independent WorldModel instances)
  - select_horizon() utility (adaptive imagination horizon)
  - EMA tracking of prior_pred_loss for the horizon signal
"""
import copy
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

import dreamer.networks as networks
import dreamer.rssm as rssm_module
import dreamer.tools as tools
from dreamer.optim import LaProp, clip_grad_agc_
from dreamer.tools import to_f32


# ---------------------------------------------------------------------------
# Adaptive horizon selector
# ---------------------------------------------------------------------------

def select_horizon(ema_obs_loss: float, config) -> int:
    """Return imagination horizon H based on WM quality signal.

    If adaptive is disabled, returns config.model.imag_horizon.
    Otherwise selects from config.adaptive.horizons based on ema_obs_loss.
    """
    if not config.adaptive.enabled:
        return int(config.model.imag_horizon)
    h_low, h_mid, h_high = [int(h) for h in config.adaptive.horizons]
    if ema_obs_loss > float(config.adaptive.thresh_high):
        return h_low
    if ema_obs_loss > float(config.adaptive.thresh_mid):
        return h_mid
    return h_high


# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    """Single world model: encoder + RSSM + decoder + reward + cont heads."""

    def __init__(self, obs_space, act_space, config):
        super().__init__()
        act_dim = act_space.n if hasattr(act_space, "n") else int(sum(act_space.shape))
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}

        self.encoder = networks.MultiEncoder(config.encoder, shapes)
        self.embed_size = self.encoder.out_dim

        self.rssm = rssm_module.RSSM(config.rssm, self.embed_size, act_dim)
        self.feat_size = self.rssm.feat_size

        self.decoder = networks.MultiDecoder(
            config.decoder,
            self.rssm._deter,
            self.rssm.flat_stoch,
            shapes,
        )
        self.reward = networks.MLPHead(config.reward, self.feat_size)
        self.cont = networks.MLPHead(config.cont, self.feat_size)

    def observe(self, data, initial):
        """Posterior rollout.  Returns (post_stoch, post_deter, post_logit, prior_logit, embed)."""
        embed = self.encoder(data)
        post_stoch, post_deter, post_logit = self.rssm.observe(
            embed, data["action"], initial, data["is_first"]
        )
        _, prior_logit = self.rssm.prior(post_deter)
        return post_stoch, post_deter, post_logit, prior_logit, embed


# ---------------------------------------------------------------------------
# Dreamer agent
# ---------------------------------------------------------------------------

class Dreamer(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.device = torch.device(config.device)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.return_ema = networks.ReturnEMA(device=self.device)
        self.act_dim = act_space.n if hasattr(act_space, "n") else int(sum(act_space.shape))

        # EMA state for prior-prediction loss (adaptive horizon signal)
        self._ema_obs_loss: float = float(config.adaptive.ema_init)
        self._ema_alpha: float = float(config.adaptive.ema_alpha)
        self._config = config

        # ── Ensemble of world models ────────────────────────────────────────
        self._n_wms = int(config.model.ensemble_size)
        self.wms = nn.ModuleList([
            WorldModel(obs_space, act_space, config) for _ in range(self._n_wms)
        ])

        # ── Actor-critic ────────────────────────────────────────────────────
        feat_size = self.wms[0].feat_size

        config.actor.shape = (
            (act_space.n,) if hasattr(act_space, "n") else tuple(map(int, act_space.shape))
        )
        self.act_discrete = False
        if hasattr(act_space, "multi_discrete"):
            config.actor.dist = config.actor.dist.multi_disc
            self.act_discrete = True
        elif hasattr(act_space, "discrete"):
            config.actor.dist = config.actor.dist.disc
            self.act_discrete = True
        else:
            config.actor.dist = config.actor.dist.cont

        self.actor = networks.MLPHead(config.actor, feat_size)
        self.value = networks.MLPHead(config.critic, feat_size)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value = copy.deepcopy(self.value)
        for p in self._slow_value.parameters():
            p.requires_grad_(False)
        self._slow_value_updates = 0

        # ── Loss scales ─────────────────────────────────────────────────────
        self._loss_scales = dict(config.loss_scales)
        recon_scale = self._loss_scales.pop("recon")
        # Set per-key decoder scales
        for k in self.wms[0].decoder.all_keys:
            self._loss_scales[k] = recon_scale

        self._log_grads = bool(config.log_grads)

        # ── Single optimizer for all params ─────────────────────────────────
        self._named_params = OrderedDict()
        for i, wm in enumerate(self.wms):
            for name, p in wm.named_parameters():
                self._named_params[f"wm{i}.{name}"] = p
        for name, p in self.actor.named_parameters():
            self._named_params[f"actor.{name}"] = p
        for name, p in self.value.named_parameters():
            self._named_params[f"value.{name}"] = p

        total_params = sum(p.numel() for p in self._named_params.values())
        print(f"Total optimiser parameters: {total_params:,}")

        def _agc(params):
            clip_grad_agc_(params, float(config.agc), float(config.pmin), foreach=True)

        self._agc = _agc
        self._optimizer = LaProp(
            self._named_params.values(),
            lr=float(config.lr),
            betas=(float(config.beta1), float(config.beta2)),
            eps=float(config.eps),
        )
        self._scaler = GradScaler()

        def lr_lambda(step):
            warmup = int(config.warmup)
            if warmup:
                return min(1.0, (step + 1) / warmup)
            return 1.0

        self._scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        self.train()
        self.clone_and_freeze()

        if bool(config.compile):
            print("Compiling _cal_grad with torch.compile...")
            self._cal_grad = torch.compile(self._cal_grad, mode="reduce-overhead")

    # ── Slow target ──────────────────────────────────────────────────────────

    def _update_slow_target(self):
        if self._slow_value_updates % self.slow_target_update == 0:
            mix = self.slow_target_fraction
            with torch.no_grad():
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1

    # ── EMA ─────────────────────────────────────────────────────────────────

    def _update_ema(self, prior_pred_loss_value: float) -> float:
        a = self._ema_alpha
        self._ema_obs_loss = (1 - a) * self._ema_obs_loss + a * prior_pred_loss_value
        return self._ema_obs_loss

    # ── Train/eval mode ──────────────────────────────────────────────────────

    def train(self, mode=True):
        super().train(mode)
        self._slow_value.train(False)
        return self

    # ── Frozen copies ────────────────────────────────────────────────────────

    def clone_and_freeze(self):
        """Create read-only frozen copies of all WM RSSM/reward heads + actor/value."""

        def _freeze_copy(src_module):
            frozen = copy.deepcopy(src_module)
            for p_orig, p_frozen in zip(src_module.parameters(), frozen.parameters()):
                p_frozen.data = p_orig.data
                p_frozen.requires_grad_(False)
            return frozen

        # Per-WM frozen RSSM + reward for imagination
        self._frozen_wm_rssms = [_freeze_copy(wm.rssm) for wm in self.wms]
        self._frozen_reward_heads = [_freeze_copy(wm.reward) for wm in self.wms]

        # Primary WM (index 0) frozen encoder + cont for act() and replay
        self._frozen_encoder = _freeze_copy(self.wms[0].encoder)
        self._frozen_rssm = self._frozen_wm_rssms[0]   # alias
        self._frozen_reward = self._frozen_reward_heads[0]  # alias
        self._frozen_cont = _freeze_copy(self.wms[0].cont)

        # Actor / value
        self._frozen_actor = _freeze_copy(self.actor)
        self._frozen_value = _freeze_copy(self.value)
        self._frozen_slow_value = _freeze_copy(self._slow_value)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.clone_and_freeze()
        return self

    # ── Policy inference ─────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, obs, state, eval: bool = False):
        """Single inference step; uses WM[0] frozen encoder/RSSM."""
        torch.compiler.cudagraph_mark_step_begin()
        p_obs = self.preprocess(obs)
        embed = self._frozen_encoder(p_obs)
        prev_stoch, prev_deter, prev_action = (
            state["stoch"], state["deter"], state["prev_action"]
        )
        stoch, deter, _ = self._frozen_rssm.obs_step(
            prev_stoch, prev_deter, prev_action, embed, obs["is_first"]
        )
        feat = self._frozen_rssm.get_feat(stoch, deter)
        action_dist = self._frozen_actor(feat)
        action = action_dist.mode if eval else action_dist.rsample()
        return action, TensorDict(
            {"stoch": stoch, "deter": deter, "prev_action": action},
            batch_size=state.batch_size,
        )

    @torch.no_grad()
    def get_initial_state(self, B: int):
        stoch, deter = self.wms[0].rssm.initial(B)
        action = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.device)
        return TensorDict(
            {"stoch": stoch, "deter": deter, "prev_action": action}, batch_size=(B,)
        )

    # ── Training step ────────────────────────────────────────────────────────

    def update(self, replay_buffer):
        """Sample a batch, compute losses, step optimizer, return metrics dict."""
        data, index, initial = replay_buffer.sample()
        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        self._update_slow_target()

        metrics = {}
        with autocast(device_type=self.device.type, dtype=torch.float16):
            (stoch, deter), mets = self._cal_grad(p_data, initial)

        self._scaler.unscale_(self._optimizer)
        if self._log_grads:
            grads = [p.grad for p in self._named_params.values() if p.grad is not None]
            mets["opt/grad_norm"] = tools.compute_global_norm(grads)
            mets["opt/grad_rms"] = tools.compute_rms(grads)

        self._agc(self._named_params.values())
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._scheduler.step()
        self._optimizer.zero_grad(set_to_none=True)

        mets["opt/lr"] = self._scheduler.get_lr()[0]
        mets["opt/grad_scale"] = self._scaler.get_scale()
        metrics.update(mets)

        # Update replay buffer latents
        replay_buffer.update(index, stoch.detach(), deter.detach())

        # EMA update for prior_pred_loss
        if "wm/prior_pred_loss" in metrics:
            raw = float(tools.to_np(metrics["wm/prior_pred_loss"]))
            ema = self._update_ema(raw)
            metrics["wm/ema_obs_loss"] = ema

        return metrics

    # ── Core gradient computation ─────────────────────────────────────────────

    def _cal_grad(self, data, initial):
        """Compute all losses and accumulate gradients.

        Single backward pass covering:
          1. WM0 posterior rollout → KL(dyn+rep) + decoder recon + reward + cont losses
          2. Extra ensemble WMs (indices 1..N-1) — same losses, independently
          3. Prior-prediction loss (no-grad) — horizon quality signal
          4. Imagination rollout from detached posteriors → lambda-return advantage
          5. Policy gradient (REINFORCE-style: logprob weighted by advantage)
          6. Value regression toward lambda returns (imagine feats + replay feats)

        Total loss: AC_loss + mean(WM_losses_over_ensemble)
        """
        losses = {}
        metrics = {}
        B, T = data.shape

        # ── Primary WM (index 0): posterior rollout ──────────────────────────
        wm0 = self.wms[0]
        post_stoch, post_deter, post_logit, prior_logit, _ = wm0.observe(data, initial)

        dyn_loss, rep_loss = wm0.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = torch.mean(dyn_loss)
        losses["rep"] = torch.mean(rep_loss)

        feat = wm0.rssm.get_feat(post_stoch, post_deter)
        recon_losses = {
            k: torch.mean(-d.log_prob(data[k]))
            for k, d in wm0.decoder(post_stoch, post_deter).items()
        }
        losses.update(recon_losses)

        cont_target = 1.0 - to_f32(data["is_terminal"])
        losses["rew"] = torch.mean(-wm0.reward(feat).log_prob(to_f32(data["reward"])))
        losses["con"] = torch.mean(-wm0.cont(feat).log_prob(cont_target))

        metrics["dyn_entropy"] = torch.mean(wm0.rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(wm0.rssm.get_dist(post_logit).entropy())

        # ── Extra ensemble WMs: independent reconstruction + KL ─────────────
        extra_wm_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        ensemble_posts = [(post_stoch, post_deter)]
        for i, wm in enumerate(self.wms[1:], start=1):
            ps, pd, pl, prl, _ = wm.observe(data, initial)
            ensemble_posts.append((ps, pd))
            dl, rl = wm.rssm.kl_loss(pl, prl, self.kl_free)
            fi = wm.rssm.get_feat(ps, pd)
            wl = torch.mean(dl) + 0.1 * torch.mean(rl)
            for k, d in wm.decoder(ps, pd).items():
                wl = wl + torch.mean(-d.log_prob(data[k]))
            wl = wl + torch.mean(-wm.reward(fi).log_prob(to_f32(data["reward"])))
            wl = wl + torch.mean(-wm.cont(fi).log_prob(cont_target))
            extra_wm_loss = extra_wm_loss + wl
            metrics[f"wm{i}/loss"] = wl.detach()

        # ── Prior-prediction loss (no grad) — horizon signal ─────────────────
        with torch.no_grad():
            prior_stoch_mode = wm0.rssm.get_dist(prior_logit).mode
            prior_recon = wm0.decoder(prior_stoch_mode, post_deter)
            obs_key = wm0.decoder.all_keys[0]
            prior_pred_loss_val = torch.mean(-prior_recon[obs_key].log_prob(data[obs_key]))
            metrics["wm/prior_pred_loss"] = prior_pred_loss_val

        # ── Imagination rollout ───────────────────────────────────────────────
        H = select_horizon(self._ema_obs_loss, self._config)
        metrics["imagine/horizon"] = torch.tensor(float(H), device=self.device)

        start = (
            post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
            post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
        )
        imag_feat, imag_action, wm_idx = self._imagine(start, H + 1)
        imag_feat = imag_feat.detach()
        imag_action = imag_action.detach()

        if self._config.model.ensemble_avg_rewards and self._n_wms > 1:
            ensemble_starts = [
                (
                    ps.reshape(-1, *ps.shape[2:]).detach(),
                    pd.reshape(-1, *pd.shape[2:]).detach(),
                )
                for ps, pd in ensemble_posts
            ]
            imag_reward = self._get_ensemble_reward(imag_feat, imag_action, ensemble_starts, wm_idx)
        else:
            imag_reward = self._get_reward_for_wm(imag_feat, wm_idx)
        imag_cont = self._frozen_cont(imag_feat).mean
        imag_value = self._frozen_value(imag_feat).mode()
        imag_slow_value = self._frozen_slow_value(imag_feat).mode()

        disc = 1 - 1 / self.horizon
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont
        ret = self._lambda_return(last, term, imag_reward, imag_value, imag_value, disc, self.lamb)
        ret_offset, ret_scale = self.return_ema(ret)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        policy = self.actor(imag_feat)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = torch.mean(
            weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy)
        )

        imag_value_dist = self.value(imag_feat)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], dim=1)
        losses["value"] = torch.mean(
            weight[:, :-1].detach()
            * (
                -imag_value_dist.log_prob(tar_padded.detach())
                - imag_value_dist.log_prob(imag_slow_value.detach())
            )[:, :-1].unsqueeze(-1)
        )

        # ── Replay-based value learning ───────────────────────────────────────
        last_r = to_f32(data["is_last"])
        term_r = to_f32(data["is_terminal"])
        reward_r = to_f32(data["reward"])
        feat_r = wm0.rssm.get_feat(post_stoch, post_deter)
        boot = ret[:, 0].reshape(B, T, 1)
        value_r = self._frozen_value(feat_r).mode()
        slow_value_r = self._frozen_slow_value(feat_r).mode()
        weight_r = 1.0 - last_r
        ret_r = self._lambda_return(last_r, term_r, reward_r, value_r, boot, disc, self.lamb)
        ret_padded_r = torch.cat([ret_r, 0 * ret_r[:, -1:]], dim=1)
        value_dist_r = self.value(feat_r)
        losses["repval"] = torch.mean(
            weight_r[:, :-1]
            * (
                -value_dist_r.log_prob(ret_padded_r.detach())
                - value_dist_r.log_prob(slow_value_r.detach())
            )[:, :-1].unsqueeze(-1)
        )

        # ── Compute total loss and backprop ───────────────────────────────────
        wm_keys = ["dyn", "rep", "rew", "con", *wm0.decoder.all_keys]
        wm0_loss = sum(losses[k] * self._loss_scales[k] for k in wm_keys)
        ac_loss = sum(
            losses[k] * self._loss_scales[k]
            for k in ("policy", "value", "repval")
        )
        total_loss = ac_loss + (wm0_loss + extra_wm_loss) / self._n_wms
        self._scaler.scale(total_loss).backward()

        # ── Logging ──────────────────────────────────────────────────────────
        metrics.update({f"loss/{k}": v for k, v in losses.items()})
        metrics["opt/loss"] = total_loss.detach()
        metrics["ret"] = torch.mean(ret)
        metrics["ret_005"] = self.return_ema.ema_vals[0]
        metrics["ret_095"] = self.return_ema.ema_vals[1]
        metrics["adv"] = torch.mean(adv)
        metrics["adv_std"] = torch.std(adv)
        metrics["con"] = torch.mean(imag_cont)
        metrics["rew"] = torch.mean(imag_reward)
        metrics["val"] = torch.mean(imag_value)
        metrics["tar"] = torch.mean(ret)
        metrics["weight"] = torch.mean(weight)
        metrics["action_entropy"] = torch.mean(entropy)
        metrics.update(tools.tensorstats(imag_action, "action"))

        return (post_stoch, post_deter), metrics

    # ── Imagination helpers ───────────────────────────────────────────────────

    @torch.no_grad()
    def _imagine(self, start, imag_horizon: int):
        """Roll out the policy in latent space.  Randomly selects one WM for dynamics."""
        wm_idx = random.randrange(self._n_wms)
        frozen_rssm = self._frozen_wm_rssms[wm_idx]

        feats, actions = [], []
        stoch, deter = start
        for _ in range(imag_horizon):
            feat = frozen_rssm.get_feat(stoch, deter)
            action = self._frozen_actor(feat).rsample()
            feats.append(feat)
            actions.append(action)
            stoch, deter = frozen_rssm.img_step(stoch, deter, action)
        return torch.stack(feats, dim=1), torch.stack(actions, dim=1), wm_idx

    def _get_reward_for_wm(self, feat, wm_idx: int):
        """Compute reward from the reward head that matches the imagined latent."""
        return self._frozen_reward_heads[wm_idx](feat).mode()

    @torch.no_grad()
    def _get_ensemble_reward(self, selected_feat, actions, starts, selected_wm_idx: int):
        """Average reward predictions across all ensemble members.

        Each WM rolls out the *same* action sequence through its own RSSM dynamics,
        so reward head i sees features from WM i's own latent space rather than
        features from the selected WM used for the imagination rollout.

        The selected WM's features are reused directly (already computed).
        Other WMs recompute their own feature trajectories via imagine_with_action.
        """
        rewards = []
        for i, (stoch, deter) in enumerate(starts):
            if i == selected_wm_idx:
                feat = selected_feat
            else:
                stochs, deters = self._frozen_wm_rssms[i].imagine_with_action(
                    stoch, deter, actions[:, :-1]
                )
                feat0 = self._frozen_wm_rssms[i].get_feat(stoch, deter).unsqueeze(1)
                feat_rest = self._frozen_wm_rssms[i].get_feat(stochs, deters)
                feat = torch.cat([feat0, feat_rest], dim=1)
            rewards.append(self._frozen_reward_heads[i](feat).mode())
        return torch.stack(rewards, dim=0).mean(dim=0)

    @torch.no_grad()
    def _lambda_return(self, last, term, reward, value, boot, disc, lamb):
        """TD-lambda return mixing Monte-Carlo and 1-step TD targets.

        lamb=1  → full discounted Monte-Carlo return
        lamb=0  → pure 1-step Bellman backup

        `last`   marks episode ends (stop discounting across them).
        `term`   marks true terminations (zero future value).
        `boot`   is the bootstrapped value used at the sequence boundary.
        """
        assert last.shape == term.shape == reward.shape == value.shape == boot.shape
        live = (1 - to_f32(term))[:, 1:] * disc
        cont = (1 - to_f32(last))[:, 1:] * lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        out = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], dim=1)

    @torch.no_grad()
    def preprocess(self, data):
        if "image" in data:
            data["image"] = to_f32(data["image"]) / 255.0
        return data
