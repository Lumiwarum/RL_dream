"""Entry point — DreamerV3 thesis implementation."""
import atexit
import pathlib
import sys
import warnings

import hydra
import torch

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    # Add dreamer_impl to path so imports work regardless of cwd
    root = pathlib.Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from dreamer import tools
    from dreamer.agent import Dreamer
    from dreamer.buffer import Buffer
    from dreamer.envs import make_envs
    from dreamer.trainer import OnlineTrainer

    tools.set_seed_everywhere(int(config.seed))

    logdir = pathlib.Path(config.paths.runs_dir) / config.exp_name
    logdir.mkdir(parents=True, exist_ok=True)

    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    tb_dir = logdir / config.paths.tb_subdir
    tb_dir.mkdir(parents=True, exist_ok=True)
    logger = tools.Logger(tb_dir)
    logger.log_hydra_config(config)

    print(f"Logdir: {logdir}")
    print(f"Env: {config.env.id}  |  Ensemble: {config.model.ensemble_size}  |  Adaptive: {config.adaptive.enabled}")

    train_envs, eval_envs, obs_space, act_space = make_envs(config)

    replay = Buffer(config.buffer)

    agent = Dreamer(config, obs_space, act_space).to(config.device)

    trainer = OnlineTrainer(
        config.trainer, replay, logger, logdir, train_envs, eval_envs, full_config=config
    )
    trainer.begin(agent)

    # Save final checkpoint
    trainer.save_checkpoint(agent, trainer.steps, "final")
    print("Saved final checkpoint.")


if __name__ == "__main__":
    main()
