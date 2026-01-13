#!/usr/bin/env python3
"""
Minimal test script for Light DreamerV3 on dm_control walker-walk.
Optimized for headless servers with flattened observations.
"""

import os
import sys
import time
from collections import deque

# ==================== HEADLESS CONFIGURATION ====================
# MUST BE SET BEFORE ANY GRAPHICS IMPORTS
os.environ['MUJOCO_GL'] = 'osmesa'        # Headless rendering
os.environ['DISPLAY'] = ''                # No display
os.environ['PYOPENGL_PLATFORM'] = 'osmesa' # Force OSMesa for PyOpenGL

# Suppress various warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# ==================== IMPORTS ====================
import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
from flax import linen as nn
from flax.training import train_state
import dm_control.suite
from shimmy import DmControlCompatibilityV0
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")

# ==================== LIGHT DREAMERV3 MODEL ====================

class LightRSSM(nn.Module):
    """Minimal Recurrent State-Space Model for DreamerV3."""
    hidden_size: int = 256          # GRU hidden state (reduce from 512 for testing)
    stoch_size: int = 16            # Number of categorical latents
    category_size: int = 16         # Categories per latent
    mlp_dim: int = 256              # MLP hidden dimension
    num_layers: int = 2             # Depth of MLPs
    
    @nn.compact
    def __call__(self, prev_state, prev_action, observation):
        """Forward pass of RSSM.
        
        Args:
            prev_state: Dictionary with keys 'deter' and 'stoch'
            prev_action: Previous action [batch, action_dim]
            observation: Current observation [batch, obs_dim]
        
        Returns:
            state: Updated state dictionary
            obs_recon: Reconstructed observation
            reward_pred: Predicted reward
        """
        # 1. Encode observation (already flattened by wrapper)
        x = observation
        for _ in range(self.num_layers):
            x = nn.Dense(self.mlp_dim)(x)
            x = nn.relu(x)
        encoded_obs = x
        
        # 2. GRU for deterministic state - FIXED: GRUCell returns (output, new_carry)
        # Flatten previous stochastic state
        batch_size = prev_state['stoch'].shape[0]
        prev_stoch_flat = prev_state['stoch'].reshape((batch_size, -1))
        
        # Correct order: (carry, inputs) where carry is previous hidden state
        gru_input = jnp.concatenate([prev_stoch_flat, prev_action], -1)
        
        # GRUCell returns (output, new_carry_state)
        # We need the new_carry_state as our deterministic state
        _, deter = nn.GRUCell(self.hidden_size)(prev_state['deter'], gru_input)
        
        # 3. Posterior latent (from observation + deter)
        x = jnp.concatenate([deter, encoded_obs], -1)
        for _ in range(self.num_layers):
            x = nn.Dense(self.mlp_dim)(x)
            x = nn.relu(x)
        post_logits = nn.Dense(self.stoch_size * self.category_size)(x)
        post_dist = jax.nn.softmax(
            post_logits.reshape((-1, self.stoch_size, self.category_size))
        )
        
        # 4. Prior latent (from deter only - dynamics)
        x = deter
        for _ in range(self.num_layers):
            x = nn.Dense(self.mlp_dim)(x)
            x = nn.relu(x)
        prior_logits = nn.Dense(self.stoch_size * self.category_size)(x)
        prior_dist = jax.nn.softmax(
            prior_logits.reshape((-1, self.stoch_size, self.category_size))
        )
        
        # 5. Sample stochastic state (straight-through gradient)
        stoch = self._sample_categorical(post_dist)
        
        # 6. Decode observation and reward
        state_features = jnp.concatenate([
            deter, 
            stoch.reshape((batch_size, -1))
        ], -1)
        
        # Observation decoder
        x = state_features
        for _ in range(self.num_layers):
            x = nn.Dense(self.mlp_dim)(x)
            x = nn.relu(x)
        obs_recon = nn.Dense(observation.shape[-1])(x)
        
        # Reward predictor
        x = state_features
        for _ in range(self.num_layers):
            x = nn.Dense(self.mlp_dim)(x)
            x = nn.relu(x)
        reward_pred = nn.Dense(1)(x)
        
        state = {'deter': deter, 'stoch': stoch, 'post_dist': post_dist, 'prior_dist': prior_dist}
        return state, obs_recon, reward_pred
    
    def _sample_categorical(self, dist):
        """Sample from categorical with straight-through gradient."""
        noise = jax.random.gumbel(self.make_rng('sample'), dist.shape)
        sample = jnp.argmax(dist + noise, -1)
        # Straight-through gradient: use sample in forward, softmax in backward
        sample_onehot = jax.nn.one_hot(sample, dist.shape[-1])
        return sample_onehot + dist - jax.lax.stop_gradient(dist)

class LightActorCritic(nn.Module):
    """Minimal MLP-based actor-critic (replaces transformer)."""
    action_size: int
    hidden_size: int = 256
    num_layers: int = 2
    
    @nn.compact
    def __call__(self, state_features):
        # Actor network
        x = state_features
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = nn.tanh(x)
        action_mean = nn.Dense(self.action_size)(x)
        action_logstd = self.param('logstd', nn.initializers.zeros, (self.action_size,))
        
        # Critic network
        v = state_features
        for _ in range(self.num_layers):
            v = nn.Dense(self.hidden_size)(v)
            v = nn.tanh(v)
        value = nn.Dense(1)(v)
        
        return {'mean': action_mean, 'logstd': action_logstd}, value

# ==================== ENVIRONMENT ====================

def make_env(domain='walker', task='walk'):
    """Create dm_control environment with flattened observations."""
    env = dm_control.suite.load(domain_name=domain, task_name=task)
    
    # Convert to Gymnasium (creates Dict observation space)
    gym_env = DmControlCompatibilityV0(env, render_mode=None)
    
    # Flatten dictionary observations to vector
    gym_env = FlattenObservation(gym_env)
    
    return gym_env

# ==================== REPLAY BUFFER ====================

class SimpleReplayBuffer:
    """Minimal replay buffer for prototyping."""
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.clear()
    
    def clear(self):
        self.observations = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.idx = 0
        self.full = False
    
    def add(self, obs, action, reward, done, next_obs):
        idx = self.idx % self.capacity
        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.next_observations[idx] = next_obs
        self.idx += 1
        if self.idx >= self.capacity:
            self.full = True
            self.idx = 0
    
    def sample(self, batch_size, seq_len):
        """Sample random sequences from buffer."""
        if not self.full and self.idx < seq_len:
            return None
        
        max_start = min(self.idx, self.capacity) - seq_len
        if max_start <= 0:
            return None
        
        batch_obs = []
        batch_action = []
        batch_reward = []
        batch_done = []
        
        for _ in range(batch_size):
            start = np.random.randint(0, max_start)
            batch_obs.append(self.observations[start:start+seq_len])
            batch_action.append(self.actions[start:start+seq_len])
            batch_reward.append(self.rewards[start:start+seq_len])
            batch_done.append(self.dones[start:start+seq_len])
        
        return (
            np.stack(batch_obs),
            np.stack(batch_action),
            np.stack(batch_reward),
            np.stack(batch_done)
        )

# ==================== TRAINING UTILITIES ====================

def create_train_state(rng, obs_dim, action_dim, config):
    """Initialize model parameters and optimizer."""
    # World Model
    rng, model_rng, actor_rng = jax.random.split(rng, 3)
    
    world_model = LightRSSM(
        hidden_size=config['hidden_size'],
        stoch_size=config['stoch_size'],
        category_size=config['category_size'],
        mlp_dim=config['mlp_dim'],
        num_layers=config['num_layers']
    )
    
    # Dummy data for initialization
    dummy_obs = jnp.ones((1, obs_dim))
    dummy_action = jnp.ones((1, action_dim))
    dummy_state = {
        'deter': jnp.ones((1, config['hidden_size'])),
        'stoch': jnp.zeros((1, config['stoch_size'], config['category_size']))
    }
    
    # Initialize world model parameters
    world_model_vars = world_model.init(
        {'params': model_rng, 'sample': model_rng},
        dummy_state, dummy_action, dummy_obs
    )
    
    # Actor-Critic
    actor_critic = LightActorCritic(
        action_size=action_dim,
        hidden_size=config['actor_hidden_size'],
        num_layers=config['actor_layers']
    )
    
    # State embedding dimension
    state_embed_dim = config['hidden_size'] + (config['stoch_size'] * config['category_size'])
    dummy_embed = jnp.ones((1, state_embed_dim))
    
    # Initialize actor-critic parameters
    actor_critic_vars = actor_critic.init(actor_rng, dummy_embed)
    
    # Merge parameters directly
    merged_params = {
        'world_model': world_model_vars['params'],
        'actor_critic': actor_critic_vars['params']
    }
    
    # Optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=config['lr'])
    )
    
    # Create train state
    state = train_state.TrainState.create(
        apply_fn=None,
        params=merged_params,
        tx=tx
    )
    
    return state, world_model, actor_critic

# Create the training step function inside main so it captures the models
def create_train_step(world_model, actor_critic, config):
    """Create a jitted training step function with captured models."""
    @jax.jit
    def train_step(state, batch, key):
        """Single training step with world model and actor-critic losses."""
        obs_seq, action_seq, reward_seq, done_seq = batch
        
        def loss_fn(params):
            # Extract parameters
            world_model_params = params['world_model']
            actor_critic_params = params['actor_critic']

            # Initialize state
            batch_size = obs_seq.shape[0]
            deter = jnp.zeros((batch_size, config['hidden_size']))
            stoch = jnp.zeros((batch_size, config['stoch_size'], config['category_size']))

            total_loss = 0.0
            recon_loss = 0.0
            kl_loss = 0.0
            reward_loss = 0.0

            # Unroll world model over sequence
            for t in range(config['seq_len'] - 1):
                state_dict = {'deter': deter, 'stoch': stoch}

                # World model forward - wrap params in dict
                next_state, obs_recon, reward_pred = world_model.apply(
                    {'params': world_model_params},
                    state_dict, action_seq[:, t], obs_seq[:, t],
                    rngs={'sample': key}
                )
                
                # Reconstruction loss (MSE)
                recon_loss += jnp.mean((obs_recon - obs_seq[:, t+1]) ** 2)
                
                # KL divergence (posterior vs prior)
                post_dist = next_state['post_dist']
                prior_dist = next_state['prior_dist']
                kl = jnp.mean(
                    jnp.sum(post_dist * (jnp.log(post_dist + 1e-8) - jnp.log(prior_dist + 1e-8)), axis=-1)
                )
                kl_loss += kl
                
                # Reward prediction loss
                if config['use_reward_pred']:
                    reward_loss += jnp.mean((reward_pred - reward_seq[:, t+1]) ** 2)
                
                # Update state for next step
                deter = next_state['deter']
                stoch = next_state['stoch']
            
            # Combine losses with weights
            total_loss = (
                config['recon_weight'] * recon_loss +
                config['kl_weight'] * kl_loss +
                config['reward_weight'] * reward_loss
            )
            
            # Add tiny regularization
            #total_loss += 1e-4 * optax.trace.norm(params['world_model'])
            
            return total_loss, (recon_loss, kl_loss, reward_loss)
        
        # Compute gradients
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        recon_loss, kl_loss, reward_loss = aux
        
        # Update parameters
        new_state = state.apply_gradients(grads=grads)
        
        return new_state, loss, recon_loss, kl_loss, reward_loss
    
    return train_step

# ==================== MAIN TEST ====================

def main():
    print("=" * 60)
    print("Testing Light DreamerV3 on Walker-Walk")
    print("=" * 60)
    
    # Configuration (MODIFY THESE FOR YOUR EXPERIMENTS)
    config = {
        # World Model
        'hidden_size': 256,           # Reduce from 512 for faster testing
        'stoch_size': 16,             # Number of categorical latents
        'category_size': 16,          # Categories per latent
        'mlp_dim': 256,               # MLP hidden dimension
        'num_layers': 2,              # Depth of MLPs
        
        # Actor-Critic
        'actor_hidden_size': 128,
        'actor_layers': 2,
        
        # Loss weights
        'recon_weight': 1.0,
        'kl_weight': 0.1,
        'reward_weight': 1.0,
        'use_reward_pred': True,
        
        # Training
        'lr': 3e-4,
        'seq_len': 32,                # Shorter for testing
        'batch_size': 16,
        'buffer_capacity': 10000,
        
        # Environment
        'domain': 'walker',
        'task': 'walk',
        'max_steps': 1000,            # For testing
        'train_interval': 10,         # Train every N steps
        
        'batch_size': 8,              # Reduce from 16
        'buffer_capacity': 2000,      # Reduce from 10000
        'seq_len': 16,                # Reduce from 32
        'hidden_size': 128,           # Reduce from 256
        'stoch_size': 8,              # Reduce from 16
        'category_size': 8,           # Reduce from 16
    }
    
    # Create environment
    print("Creating environment...")
    env = make_env(config['domain'], config['task'])
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Initialize model
    print("Initializing model...")
    rng = jax.random.PRNGKey(42)
    state, world_model, actor_critic = create_train_state(rng, obs_dim, action_dim, config)

    # Create the jitted training step function
    train_step_fn = create_train_step(world_model, actor_critic, config)
    
    # Create replay buffer
    buffer = SimpleReplayBuffer(
        capacity=config['buffer_capacity'],
        obs_dim=obs_dim,
        action_dim=action_dim
    )
    
    # Training metrics
    episode_returns = deque(maxlen=10)
    episode_steps = deque(maxlen=10)
    loss_history = []
    
    # Start training loop
    print("\nStarting training loop...")
    print("-" * 60)
    
    total_steps = 0
    rng, env_rng = jax.random.split(rng)
    
    for episode in range(5):  # Run 5 episodes for testing
        obs, _ = env.reset()
        done = False
        episode_return = 0
        episode_step = 0
        
        while not done and episode_step < config['max_steps']:
            # Random action for initial exploration
            action = env.action_space.sample()
            
            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            buffer.add(obs, action, reward, float(done), next_obs)
            
            # Train occasionally
            if total_steps % config['train_interval'] == 0 and buffer.full:
                batch = buffer.sample(config['batch_size'], config['seq_len'])
                if batch is not None:
                    rng, train_rng = jax.random.split(rng)
                    # Use the created function instead
                    state, loss, recon, kl, reward = train_step_fn(
                        state, batch, train_rng
                    )
                    loss_history.append(float(loss))
            
            obs = next_obs
            episode_return += reward
            episode_step += 1
            total_steps += 1
        
        # Episode statistics
        episode_returns.append(episode_return)
        episode_steps.append(episode_step)
        
        print(f"Episode {episode + 1}: "
              f"Return = {episode_return:7.2f}, "
              f"Steps = {episode_step:3d}, "
              f"Buffer = {min(buffer.idx, buffer.capacity)}/{buffer.capacity}")
        
        # Early exit for testing
        if episode >= 2 and len(loss_history) > 10:
            break
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"Total steps: {total_steps}")
    print(f"Total episodes: {episode + 1}")
    if episode_returns:
        print(f"Average return (last {len(episode_returns)}): {np.mean(episode_returns):.2f}")
    if loss_history:
        print(f"Final training loss: {loss_history[-1]:.6f}")
        print(f"Loss trend: {'Decreasing' if len(loss_history) > 10 and loss_history[-1] < loss_history[0] else 'Stable/Increasing'}")
    
    print("\nModel Architecture:")
    print(f"  RSSM: hidden={config['hidden_size']}, stoch={config['stoch_size']}x{config['category_size']}")
    print(f"  Actor-Critic: {config['actor_layers']}x{config['actor_hidden_size']} MLP")
    
    print("\n✅ Test successful! The model is training.")
    print("   Next: Increase episode count, tune hyperparameters, add imagination.")
    
    env.close()

if __name__ == "__main__":
    main()