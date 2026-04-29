"""Online training loop — adapted from r2dreamer (NM512/r2dreamer).

Changes vs. r2dreamer:
  - Config keys mapped to the new config.yaml structure (trainer sub-config).
  - Removed video prediction (state-based tasks only).
  - Eval logs eval/return_mean and eval/return_std.
  - Training logs include wm/ema_obs_loss and imagine/horizon from agent metrics.
"""
import torch
from omegaconf import OmegaConf

from dreamer import tools


class OnlineTrainer:
    def __init__(
        self,
        config,
        replay_buffer,
        logger,
        logdir,
        train_envs,
        eval_envs,
        full_config=None,
    ):
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.logdir = logdir
        self.full_config = full_config
        self.steps = int(config.steps)
        self.pretrain = int(config.pretrain)
        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episodes)
        self.batch_length = int(config.batch_length)
        self._action_repeat = int(config.action_repeat)
        self.update_count = 0
        self.best_eval_return = float("-inf")

        batch_steps = int(config.batch_size * config.batch_length)
        self._updates_needed = tools.Every(
            batch_steps / config.train_ratio * config.action_repeat
        )
        self._should_pretrain = tools.Once()
        self._should_log = tools.Every(int(config.log_every))
        self._should_eval = tools.Every(self.eval_every)
        checkpoint_every = int(getattr(config, "checkpoint_every", 0))
        self._should_checkpoint = tools.Every(checkpoint_every) if checkpoint_every > 0 else None
        self.save_latest = bool(getattr(config, "save_latest", True))
        self.save_best = bool(getattr(config, "save_best", True))
        self.save_periodic = bool(getattr(config, "save_periodic", False))
        self.best_min_delta = float(getattr(config, "best_min_delta", 0.0))

        ckpt_subdir = "checkpoints"
        if full_config is not None and hasattr(full_config, "paths"):
            ckpt_subdir = str(getattr(full_config.paths, "ckpt_subdir", ckpt_subdir))
        self.ckpt_dir = self.logdir / ckpt_subdir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Checkpoints ─────────────────────────────────────────────────────────

    def _checkpoint_payload(self, agent, step: int, eval_return=None):
        cfg = None
        if self.full_config is not None:
            cfg = OmegaConf.to_container(self.full_config, resolve=True)
        return {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            "step": int(step),
            "updates": int(self.update_count),
            "eval_return": None if eval_return is None else float(eval_return),
            "best_eval_return": float(self.best_eval_return),
            "cfg": cfg,
        }

    def save_checkpoint(self, agent, step: int, tag: str, eval_return=None):
        path = self.ckpt_dir / f"{tag}.pt"
        tmp = self.ckpt_dir / f".{tag}.tmp.pt"
        torch.save(self._checkpoint_payload(agent, step, eval_return), tmp)
        tmp.replace(path)
        print(f"Saved checkpoint: {path}")
        return path

    # ── Evaluation ────────────────────────────────────────────────────────────

    def eval(self, agent, train_step):
        """Run deterministic evaluation episodes and log metrics.

        Runs until every eval env has completed at least one episode.
        Returns mean return across eval envs and saves latest/best checkpoints.
        """
        print("Evaluating...")
        envs = self.eval_envs
        agent.eval()

        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        once_done = torch.zeros(envs.env_num, dtype=torch.bool, device=agent.device)
        steps = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        agent_state = agent.get_initial_state(envs.env_num)
        act = agent_state["prev_action"].clone()

        while not once_done.all():
            steps += ~done * ~once_done
            act_cpu = act.detach().to("cpu")
            done_cpu = done.detach().to("cpu")
            trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)
            trans = trans_cpu.to(agent.device, non_blocking=True)
            done = done_cpu.to(agent.device)
            trans["action"] = act
            act, agent_state = agent.act(trans, agent_state, eval=True)
            returns += trans["reward"][:, 0] * ~once_done
            once_done |= done

        mean_return = returns.mean()
        self.logger.scalar("eval/return_mean", mean_return)
        self.logger.scalar("eval/return_std", returns.std())
        self.logger.scalar("eval/return_max", returns.max())
        self.logger.scalar("eval/episode_length", steps.to(torch.float32).mean())
        improved = mean_return.item() > self.best_eval_return + self.best_min_delta
        if improved:
            self.best_eval_return = mean_return.item()
        self.logger.scalar("eval/best_return", self.best_eval_return)
        self.logger.write(train_step)
        if self.save_latest:
            self.save_checkpoint(agent, train_step, "latest", mean_return.item())
        if self.save_best and improved:
            self.save_checkpoint(agent, train_step, "best", mean_return.item())
        agent.train()
        return mean_return.item()

    # ── Training loop ────────────────────────────────────────────────────────

    def begin(self, agent):
        """Run the main training loop until `trainer.steps` env steps are reached.

        Interleaves environment collection and gradient updates according to
        `train_ratio` (env steps per update batch).  Runs evaluation at
        `eval_every` steps, logs at `log_every` steps, and saves checkpoints
        periodically.  Does NOT save the final checkpoint — call
        `save_checkpoint(agent, trainer.steps, "final")` after returning.
        """
        envs = self.train_envs
        step = self.replay_buffer.count() * self._action_repeat
        update_count = 0

        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        lengths = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        episode_ids = torch.arange(envs.env_num, dtype=torch.int32, device=agent.device)

        train_metrics = {}
        agent_state = agent.get_initial_state(envs.env_num)
        act = agent_state["prev_action"].clone()

        while step < self.steps:
            # Evaluation
            if self._should_eval(step) and self.eval_episode_num > 0:
                self.eval(agent, step)

            # Log completed episodes
            if done.any():
                for i, d in enumerate(done):
                    if d and lengths[i] > 0:
                        self.logger.scalar("train/episode_return", returns[i])
                        self.logger.scalar("train/episode_length", lengths[i])
                        self.logger.write(step + i)
                        returns[i] = lengths[i] = 0

            step += int((~done).sum()) * self._action_repeat
            lengths += ~done

            # Step environments
            act_cpu = act.detach().to("cpu")
            done_cpu = done.detach().to("cpu")
            trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)
            trans = trans_cpu.to(agent.device, non_blocking=True)
            done = done_cpu.to(agent.device)

            # Policy inference
            act, agent_state = agent.act(trans.clone(), agent_state, eval=False)

            # Store transition
            trans["action"] = act * ~done.unsqueeze(-1)
            trans["stoch"] = agent_state["stoch"]
            trans["deter"] = agent_state["deter"]
            # Keep one replay trajectory per env worker. Early MuJoCo episodes can
            # be shorter than batch_length; RSSM uses is_first to reset within
            # sampled streams, so sampling across episode boundaries is valid.
            trans["episode"] = episode_ids
            self.replay_buffer.add_transition(trans.detach())
            returns += trans["reward"][:, 0]

            # Model updates
            if step // (envs.env_num * self._action_repeat) > self.batch_length + 1:
                if self._should_pretrain():
                    update_num = self.pretrain
                else:
                    update_num = self._updates_needed(step)
                for _ in range(update_num):
                    _metrics = agent.update(self.replay_buffer)
                    train_metrics = _metrics
                update_count += update_num
                self.update_count = update_count

                if self._should_log(step):
                    for name, value in train_metrics.items():
                        value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
                        self.logger.scalar(f"train/{name}", value)
                    self.logger.scalar("train/updates", update_count)
                    self.logger.write(step, fps=True)

            if self._should_checkpoint is not None and self._should_checkpoint(step):
                if self.save_latest:
                    self.save_checkpoint(agent, step, "latest")
                if self.save_periodic:
                    self.save_checkpoint(agent, step, f"step_{step}")
