from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import DiagGaussian, MLP


@dataclass
class RSSMState:
    h: torch.Tensor     # deterministic hidden [B, deter_dim]
    z: torch.Tensor     # stochastic latent [B, latent_dim]


class RSSM(nn.Module):
    """
    A compact RSSM-like latent dynamics model for state-vector observations.

    - Posterior: q(z_t | h_t, obs_t)  (inference model)
    - Prior:     p(z_t | h_t)        (dynamics prior after GRU update)
    - Deterministic transition: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])

    This is "RSSM-lite" but works well for research prototyping.
    """
    def __init__(self, obs_dim: int, act_dim: int, latent_dim: int, deter_dim: int, hidden_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.deter_dim = deter_dim
        self.hidden_dim = hidden_dim

        # Dreamer-style: embed (z, a) before GRU; helps stability vs feeding raw concat.
        self.inp_net = MLP(latent_dim + act_dim, deter_dim, hidden_dim=hidden_dim, num_layers=2)
        self.gru = nn.GRUCell(deter_dim, deter_dim)
        self.h_norm = nn.LayerNorm(deter_dim)

        # std parameterization closer to Dreamer: std = softplus(x) + min_std
        self.min_std = 0.1

        # Prior and posterior parameter nets (diag Gaussian)
        self.prior_net = MLP(deter_dim, 2 * latent_dim, hidden_dim=hidden_dim, num_layers=2)
        self.post_net = MLP(deter_dim + obs_dim, 2 * latent_dim, hidden_dim=hidden_dim, num_layers=2)

    def init_state(self, batch_size: int, device: torch.device) -> RSSMState:
        h = torch.zeros((batch_size, self.deter_dim), device=device)
        z = torch.zeros((batch_size, self.latent_dim), device=device)
        return RSSMState(h=h, z=z)

    def _dist_from_params(self, params: torch.Tensor) -> DiagGaussian:
        mean, std_param = torch.chunk(params, 2, dim=-1)
        std = F.softplus(std_param) + self.min_std
        log_std = torch.log(std)
        return DiagGaussian(mean=mean, log_std=log_std)

    def prior(self, h: torch.Tensor) -> DiagGaussian:
        return self._dist_from_params(self.prior_net(h))

    def posterior(self, h: torch.Tensor, obs: torch.Tensor) -> DiagGaussian:
        x = torch.cat([h, obs], dim=-1)
        return self._dist_from_params(self.post_net(x))

    def deter_step(self, h: torch.Tensor, z_prev: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_prev, a_prev], dim=-1)
        x = self.inp_net(x)
        h = self.gru(x, h)
        return self.h_norm(h)

    def observe_sequence(
        self,
        obs: torch.Tensor,        # [B,T,obs_dim]
        actions: torch.Tensor,    # [B,T,act_dim]
        device: torch.device,
        sample: bool = True
    ) -> Tuple[List[RSSMState], List[DiagGaussian], List[DiagGaussian]]:
        """
        Run inference over a real sequence to produce posterior states.

        Returns lists length T:
          states[t] = (h_t, z_t)
          post_dists[t] = q(z_t | h_t, obs_t)
          prior_dists[t] = p(z_t | h_t)
        """
        B, T, _ = obs.shape
        state = self.init_state(B, device=device)

        states: List[RSSMState] = []
        post_dists: List[DiagGaussian] = []
        prior_dists: List[DiagGaussian] = []

        for t in range(T):
            # Update deterministic state using previous z and previous action.
            if t == 0:
                a_prev = torch.zeros((B, self.act_dim), device=device)
            else:
                a_prev = actions[:, t-1]
            state.h = self.deter_step(state.h, state.z, a_prev)
            # Prior from h
            prior_dist = self.prior(state.h)
            # Posterior conditioned on obs_t
            post_dist = self.posterior(state.h, obs[:, t])
            z_t = post_dist.mean if not sample else post_dist.sample()
            state = RSSMState(h=state.h, z=z_t)

            states.append(state)
            post_dists.append(post_dist)
            prior_dists.append(prior_dist)

        return states, post_dists, prior_dists

    def imagine_step(
        self,
        state: RSSMState,
        action: torch.Tensor,
        sample: bool = True,
    ) -> Tuple[RSSMState, DiagGaussian]:
        """
        Latent imagination: given current state and action, produce next state using the prior only.
        """
        h_next = self.deter_step(state.h, state.z, action)
        prior_dist = self.prior(h_next)
        z_next = prior_dist.sample() if sample else prior_dist.mean
        return RSSMState(h=h_next, z=z_next), prior_dist


class WorldModel(nn.Module):
    """
    World model = RSSM + prediction heads:
      - next_obs (or obs) prediction from (h,z)
      - reward prediction from (h,z)
      - optional done prediction

    Loss:
      L = MSE(obs_pred, obs_target) + MSE(rew_pred, rew) + beta * KL(q(z|h,obs) || p(z|h))
    """
    def __init__(self, obs_dim: int, act_dim: int, latent_dim: int, deter_dim: int, hidden_dim: int):
        super().__init__()
        self.rssm = RSSM(obs_dim, act_dim, latent_dim, deter_dim, hidden_dim)

        feat_dim = deter_dim + latent_dim
        self.obs_head = MLP(feat_dim, obs_dim, hidden_dim=hidden_dim, num_layers=2)
        self.rew_head = MLP(feat_dim, 1, hidden_dim=hidden_dim, num_layers=2)
        self.cont_head = MLP(feat_dim, 1, hidden_dim=hidden_dim, num_layers=2)
        # Prior: episodes usually continue. sigmoid(+3) ≈ 0.95 — without this bias the default
        # sigmoid(0)=0.5 means the WM immediately learns low cont_prob from early falling-robot
        # data, collapsing imagination discounts and preventing the actor from seeing long-horizon reward.
        nn.init.constant_(self.cont_head.net[-1].bias, 3.0)

        # Predict next_obs given current latent feature; in this simple setup we predict obs_t directly.
        # For stability on MuJoCo state vectors, predicting obs_t works fine with proper sequence setup.
        
    def predict_continue_logit(self, feat: torch.Tensor) -> torch.Tensor:
        return self.cont_head(feat).squeeze(-1)  # logits


    def features(self, state: RSSMState) -> torch.Tensor:
        return torch.cat([state.h, state.z], dim=-1)

    def predict_obs(self, feat: torch.Tensor) -> torch.Tensor:
        return self.obs_head(feat)

    def predict_reward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.rew_head(feat).squeeze(-1)


class EnsembleWorldModel(nn.Module):
    """
    Small ensemble of independent world models.
    Used for uncertainty estimation (disagreement).
    """
    def __init__(self, n: int, obs_dim: int, act_dim: int, latent_dim: int, deter_dim: int, hidden_dim: int):
        super().__init__()
        self.n = int(n)
        self.models = nn.ModuleList(
            [WorldModel(obs_dim, act_dim, latent_dim, deter_dim, hidden_dim) for _ in range(self.n)]
        )

    def forward(self):
        raise NotImplementedError("Call member models explicitly.")

    @torch.no_grad()
    def disagreement(
        self,
        obs: torch.Tensor,       # [B, obs_dim]
        action: torch.Tensor,    # [B, act_dim]
        metric: Literal["latent_mean_l2", "next_obs_mean_l2", "gaussian_kl"],
        device: torch.device,
    ) -> torch.Tensor:
        """
        One-step disagreement score per sample (shape [B]).
        We compute a posterior from obs, imagine one step with action, and compare ensemble predictions.

        Metrics:
          - latent_mean_l2: L2 distance between ensemble prior means for z_{t+1}
          - next_obs_mean_l2: L2 distance between ensemble predicted obs means at t+1
          - gaussian_kl: symmetric KL between two predicted next-latent Gaussians (for n=2)
        """
        assert self.n >= 2, "Need at least 2 models for disagreement."
        B = obs.shape[0]

        # For each model: infer current state from obs (single-step), then imagine one step with action.
        next_z_means = []
        next_z_dists = []
        next_obs_means = []

        for wm in self.models:
            # Build a single-step pseudo sequence: actions has shape [B,1,act_dim], obs [B,1,obs_dim]
            obs_seq = obs.unsqueeze(1)
            act_seq = action.unsqueeze(1)
            states, post_dists, prior_dists = wm.rssm.observe_sequence(obs_seq, act_seq, device=device, sample=False)
            s0 = states[-1]
            s1, prior1 = wm.rssm.imagine_step(s0, action)
            next_z_means.append(prior1.mean)
            next_z_dists.append(prior1)

            feat1 = wm.features(s1)
            next_obs_means.append(wm.predict_obs(feat1))

        if metric == "latent_mean_l2":
            # True pairwise average: all (i,j) pairs, each member weighted equally.
            # Comparing only vs member-0 gives member-0 double weight for N≥3.
            d, count = 0.0, 0
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    d = d + torch.norm(next_z_means[i] - next_z_means[j], dim=-1)
                    count += 1
            return d / float(count)

        if metric == "next_obs_mean_l2":
            d, count = 0.0, 0
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    d = d + torch.norm(next_obs_means[i] - next_obs_means[j], dim=-1)
                    count += 1
            return d / float(count)

        if metric == "gaussian_kl":
            if self.n == 2:
                # Fast path for n=2
                p = next_z_dists[0]
                q = next_z_dists[1]
                kl_pq = p.kl_div(q)
                kl_qp = q.kl_div(p)
                return 0.5 * (kl_pq + kl_qp)
            else:
                # Average all pairwise KLs for n > 2
                total_kl = 0.0
                count = 0
                for i in range(self.n):
                    for j in range(i + 1, self.n):
                        p = next_z_dists[i]
                        q = next_z_dists[j]
                        kl_pq = p.kl_div(q)
                        kl_qp = q.kl_div(p)
                        total_kl += 0.5 * (kl_pq + kl_qp)
                        count += 1
                return total_kl / count

        raise ValueError(f"Unknown metric: {metric}")
