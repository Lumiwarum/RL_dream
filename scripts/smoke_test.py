import os

import numpy as np
import torch
from omegaconf import OmegaConf

from thesiswm.envs.make_env import make_env
from thesiswm.models.rssm import EnsembleWorldModel
from thesiswm.data.replay_buffer import ReplayBuffer


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")

    env_id = "InvertedPendulum-v4"
    env = make_env(env_id, seed=0, render_mode=None)
    obs, _ = env.reset(seed=0)
    obs = np.asarray(obs, dtype=np.float32)
    act_dim = int(env.action_space.shape[0])
    obs_dim = int(obs.shape[0])

    # Step once
    a = env.action_space.sample().astype(np.float32)
    next_obs, r, d, tr, _ = env.step(a)
    next_obs = np.asarray(next_obs, dtype=np.float32)
    assert next_obs.shape == obs.shape

    # Replay buffer add/sample
    rb = ReplayBuffer(capacity=1000, obs_dim=obs_dim, act_dim=act_dim)
    for _ in range(200):
        a = env.action_space.sample().astype(np.float32)
        nobs, r, d, tr, _ = env.step(a)
        rb.add(obs, a, float(r), bool(d), bool(tr), np.asarray(nobs, dtype=np.float32))
        obs = np.asarray(nobs, dtype=np.float32)
        if d or tr:
            obs, _ = env.reset(seed=0)

    batch = rb.sample_sequences(batch_size=8, seq_len=16)
    assert batch["obs"].shape == (8, 16, obs_dim)

    # World model forward shapes
    device = torch.device("cpu")
    ens = EnsembleWorldModel(n=2, obs_dim=obs_dim, act_dim=act_dim, latent_dim=32, deter_dim=128, hidden_dim=256).to(device)
    obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    act_t = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
    wm0 = ens.models[0]
    states, post, prior = wm0.rssm.observe_sequence(obs_t, act_t, device=device, True)
    feat0 = wm0.features(states[0])
    assert feat0.shape[-1] == (128 + 32)

    # Disagreement metric
    o0 = torch.as_tensor(rb.sample_batch(16)["obs"], dtype=torch.float32, device=device)
    a0 = torch.as_tensor(rb.sample_batch(16)["actions"], dtype=torch.float32, device=device)
    u = ens.disagreement(o0, a0, metric="next_obs_mean_l2", device=device)
    assert u.shape == (16,)

    env.close()
    print("Smoke test OK")


if __name__ == "__main__":
    main()
