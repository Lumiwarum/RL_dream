# Research Context — Master Thesis

**Topic:** Adaptive Imagination Horizon in World Model Reinforcement Learning  
**Date:** 2026-04-28  
**Base algorithm:** DreamerV3 (Hafner et al. 2023)  
**Implementation base:** NM512/r2dreamer (PyTorch DreamerV3 reproduction)

---

## 1. Core Thesis Idea

Standard DreamerV3 uses a **fixed imagination horizon** H (default 15 steps) for actor-critic training. This is suboptimal: when the world model (WM) is still inaccurate early in training, long rollouts compound WM errors and produce garbage training signal for the actor. When the WM has converged, short rollouts leave reward credit assignment on the table.

**Hypothesis:** Dynamically adjusting H based on the current quality of the world model improves policy learning stability and/or final performance compared to a fixed H baseline.

### Novel Contribution 1 — Ensemble World Model
Train N=2 independent world models on the same replay buffer. During imagination, the actor sees reward predictions from all ensemble members (averaged by default, or pessimistic minimum). This has two effects:
- The actor must earn reward in both models simultaneously, reducing exploitation of individual WM biases
- Ensemble disagreement gives a secondary uncertainty signal (used only as a logging diagnostic in the final design; the EMA signal below proved more reliable as the horizon selector)

### Novel Contribution 2 — Adaptive Imagination Horizon
Select H ∈ {H_low, H_mid, H_high} per training batch based on the **EMA of the world model's prior prediction loss**:

```
prior_pred_loss = MSE( obs_decoder(h_t, z_prior_t), symlog(obs_t) )
```

This measures how well the WM predicts observations using **only its prior** (no real obs), without gradients. High value = WM is unreliable in open-loop = actor should use short H to avoid compounding errors.

```
ema_obs_loss > thresh_high  →  H = H_low   (e.g. 10) — WM still learning
ema_obs_loss > thresh_mid   →  H = H_mid   (e.g. 15)
ema_obs_loss < thresh_mid   →  H = H_high  (e.g. 20) — WM has converged
```

The EMA is initialised at a high value (e.g. 2.0) so the first horizon is always H_low (cautious start). As training proceeds and the WM improves, the EMA naturally decays, triggering horizon increases.

---

## 2. Experimental Design

### Environments
- **InvertedPendulum-v5** (4-dim obs, 1-dim act) — fast, simple, proof-of-concept
- **Hopper-v4** (11-dim obs, 3-dim act) — medium complexity, main benchmark
- **Walker2d-v4** (17-dim obs, 6-dim act) — harder, secondary benchmark

### Method Variants
| Name | Horizon | WM | Description |
|---|---|---|---|
| `fixed_h10` | 10 (fixed) | N=1 | Short horizon baseline |
| `fixed_h15` | 15 (fixed) | N=1 | DreamerV3 default baseline |
| `fixed_h20` | 20 (fixed) | N=1 | Long horizon baseline |
| `fixed_h20_ens` | 20 (fixed) | N=2 | Ensemble ablation — isolates ensemble effect |
| `adaptive` | {10,15,20} adaptive | N=1 | Adaptive horizon ablation — isolates horizon effect |
| `adaptive_ens` | {10,15,20} adaptive | N=2 | Full method |

### Control Study (2×2 Factorial on Hopper-v4)
Resolves the confound of changing two things at once:
- Factor A: Horizon (fixed H=20 vs adaptive {10,15,20})
- Factor B: WM (single N=1 vs ensemble N=2)

Allows attributing any performance difference to either ensemble alone, adaptive horizon alone, or the combination.

### Experiment Schedule
- Smoke: 100k steps, 2 seeds (fast sanity check before committing to full runs)
- Full: 500k steps, 3 seeds (main comparison)
- Long: 1M steps, 2 seeds (definitive, run only when 500k is ambiguous)

---

## 3. Key Experimental Findings

### What Worked

**EMA prior_pred_loss as horizon signal** — much more useful than ensemble disagreement. Disagreement stabilises to a fixed value within ~500 WM updates and locks to one horizon for the whole run. The EMA obs_loss continues to decrease as the WM trains, giving genuine within-run horizon switching (H=10 early → H=20 late).

**Categorical latents** — discrete latent space (32 slots × 32 classes, DreamerV3 default) prevents the actor from exploiting smooth WM gradients. With Gaussian continuous latents, the pathwise gradient through the WM causes the actor to find imagined trajectories that maximise reward prediction but don't generalise to the real environment.

**REINFORCE actor gradient** — applying REINFORCE (score-function estimator) instead of reparameterisation through the WM dynamics prevents gradient exploitation. The actor sees detached WM features; gradient flows only through the log_prob w.r.t. actor parameters. This is DreamerV3's default (`ac_grads=False` in the JAX code).

**Symlog observation encoding** — normalising inputs with symlog(x) = sign(x) * log(|x|+1) is essential for MuJoCo state observations, which have very different scales across dimensions.

**WM decoder on posterior features** — the reconstruction target must be obs_t decoded from (h_t, z_posterior_t), not from prior features or next-step features. This is the standard VAE objective.

**Reward/continue on transition features** — reward[t] and terminated[t] are consequences of action[t], so they must be predicted from features that include action[t]. The correct features are (h_next, z_next_post) where h_next = GRU(h_t, z_t, a_t).

### What Did Not Work

**Pathwise actor gradient** — allows the actor to exploit WM biases via reparameterisation. Actor gradient norm grows (p95 ≈ 33), policy std collapses to the floor, and eval return stagnates despite high imagined returns.

**cont_disc_floor > 0** — clamping the continuation discount to a minimum (e.g. 0.9) erases the WM's termination signal. The actor cannot learn to avoid episode-ending states.

**Rollback mechanism** — restoring actor/critic to a past checkpoint while the WM has moved on causes immediate re-degradation. The WM and replay buffer cannot be rolled back consistently, so restoring the actor alone doesn't help.

**Ensemble disagreement as horizon signal** — see above (stabilises too quickly, no within-run dynamics).

**Very short horizons (H=5)** — the bootstrap fraction (proportion of value coming from the last step's value function) is ~77% at H=5. This saturates the advantage normalisation scale and the actor gradient approaches zero.

**Gaussian continuous latents** — smooth latent space allows gradient exploitation. Categorical latents solve this.

### Observed Training Dynamics

Healthy training on InvertedPendulum with DreamerV3:
- WM obs_loss starts ~1.5, drops to <0.1 within 20k steps
- EMA obs_loss: starts high (~2), decays to ~0.03–0.08 by 50k
- With adaptive H: H=10 early (WM learning), H=20 after ~30–50k steps (WM converged)
- eval/return reaches 1000 (max) by 100–200k steps

### Calibrated Thresholds (per Environment)

**InvertedPendulum-v5:**
```
obs_loss_thresh_high = 0.30  →  H=10 while WM is learning
obs_loss_thresh_mid  = 0.08  →  H=15 in mid-training
                               H=20 once WM has converged
```

**Hopper-v4:**
```
obs_loss_thresh_high = 0.27
obs_loss_thresh_mid  = 0.20
horizons = [10, 15, 20]
```

**Walker2d-v4:**
```
obs_loss_thresh_high = 1.80
obs_loss_thresh_mid  = 1.30
horizons = [10, 15, 20]
# Walker2d obs_loss is much higher because 17-dim obs + complex dynamics
```

---

## 4. Why Moving to r2dreamer Baseline

The previous custom implementation accumulated:
- Multiple architectural changes not in DreamerV3 (sigmoid std, SIGReg regulariser, custom KL balance, special cont_disc_floor logic, pretanh regulariser, rollback/freeze mechanism)
- Each fix for one bug introduced another instability
- Difficult to separate thesis contributions from implementation quirks

The r2dreamer codebase is:
- A clean, cited PyTorch DreamerV3 reproduction (runs ~5x faster than naive impl)
- Community-validated, matching DreamerV3 paper results
- Used as a baseline for r2dreamer, DreamerPro, InfoNCE ablations in published work

The migration adds ONLY the two thesis contributions on top of r2dreamer's unmodified DreamerV3 core. Everything else stays exactly as r2dreamer provides it.

---

## 5. What to Measure / Log for the Thesis

### Primary Metrics
- `eval/return_mean` — main performance metric (mean over eval episodes)
- `eval/return_std` — variance across eval episodes

### World Model Health
- `loss/obs` — reconstruction loss (should decrease)
- `loss/obs_prior` — prior prediction loss (should decrease; drives adaptive horizon)
- `loss/kl` — KL divergence (should be > 0; near-zero = posterior collapse)
- `loss/reward` — reward prediction loss
- `wm/ema_obs_loss` — the EMA signal that drives horizon selection

### Actor-Critic Health
- `policy/std_mean` — policy standard deviation (should be > 0.1; near floor = collapsed)
- `grad/actor_norm` — actor gradient norm (should be stable, not growing)
- `value/return_scale` — advantage normalisation scale (should be < scale_max)
- `imagine/horizon_used` — which H was selected (for adaptive runs, should vary)
- `imagine/uncertainty_mean` — ensemble disagreement (secondary signal, always logged)

### Adaptive Horizon Verification
- `wm/ema_obs_loss` vs `imagine/horizon_used`: these should be negatively correlated
  (when ema_obs_loss is high → H is low; when ema_obs_loss is low → H is high)
- Compute early_avg_H (first third of training) vs late_avg_H (last third)
  - Expected: late_avg_H > early_avg_H for adaptive runs

---

## 6. Recommended Hyperparameters (Stable Defaults)

These are DreamerV3 defaults unless noted; all had empirical justification:

```yaml
# World model
world_model:
  ensemble_size: 1           # 1 = baseline, 2 = thesis ensemble
  deter: 512                 # DreamerV3 default for state tasks
  stoch: 32                  # categorical slots
  classes: 32                # classes per slot
  kl_free: 1.0               # free nats (floor on KL loss)
  lr: 3e-4                   # Adam, DreamerV3 default

# Actor-Critic
actor_critic:
  actor_lr: 3e-4             # DreamerV3 default
  critic_lr: 3e-4
  discount: 0.997            # DreamerV3 default
  lambda_: 0.95
  entropy_coef: 3e-4         # DreamerV3 default

# Imagination
imagination:
  horizon: 15                # fixed default; adaptive overrides this
  ensemble_avg_rewards: false  # true = average reward over ensemble members

# Adaptive horizon (disabled by default)
adaptive:
  enabled: false
  horizons: [10, 15, 20]
  ema_alpha: 0.003           # slow EMA — keeps signal high early in training
  ema_init: 2.0              # initialise high so first horizon is always H_low
```

---

## 7. Thesis Writing Notes

### Framing
The thesis should be framed as: "We propose Adaptive Imagination Horizon (AIH) for DreamerV3-style world model RL, where the imagination rollout length H is dynamically adjusted based on the world model's open-loop prediction quality."

### Claims to Support Experimentally
1. **Fixed H is suboptimal**: compare fixed_h10 vs fixed_h15 vs fixed_h20 — show that different environments need different H values (no single H is best everywhere)
2. **Adaptive H finds the right H automatically**: show that adaptive runs select H≈H_low early, H≈H_high late, matching the empirically best fixed H at each training phase
3. **Ensemble adds complementary benefit**: the 2×2 control study shows whether ensemble alone, adaptive H alone, or both together drive any performance improvement

### Expected Outcome
- Adaptive H should match or exceed the best fixed H on both Hopper and Pendulum
- The adaptive H signal should be visually interpretable (EMA decaying, H increasing) — good for a figure
- The 2×2 factorial gives a clean attribution table for the thesis
