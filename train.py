import os
import sys
from typing import List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from thesiswm.training.trainer import Trainer
from thesiswm.utils.seed import set_global_seeds


def _preprocess_argv(argv: List[str]) -> List[str]:
    """
    Convert user-friendly CLI flags into Hydra overrides.

    Supported:
      --config <name>             -> loads configs/<name>.yaml (via Hydra config_name)
      --total_steps <int>         -> training.total_steps=<int>
      --steps_per_chunk <int>     -> training.steps_per_chunk=<int>
      --resume                    -> training.resume=true
      --no_resume                 -> training.resume=false
    """
    out: List[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--total_steps":
            out.append(f"training.total_steps={int(argv[i+1])}")
            i += 2
        elif a == "--steps_per_chunk":
            out.append(f"training.steps_per_chunk={int(argv[i+1])}")
            i += 2
        elif a == "--resume":
            out.append("training.resume=true")
            i += 1
        elif a == "--no_resume":
            out.append("training.resume=false")
            i += 1
        elif a == "--config":
            # handled outside by passing config_name to hydra.main wrapper
            out.append(a)
            out.append(argv[i+1])
            i += 2
        else:
            out.append(a)
            i += 1
    return out


def _extract_config_name(argv: List[str], default_name: str = "config") -> str:
    if "--config" in argv:
        idx = argv.index("--config")
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return default_name


def main():
    argv = _preprocess_argv(sys.argv[1:])
    config_name = _extract_config_name(argv, default_name="config")
    # remove the --config <name> pair from argv passed to Hydra
    if "--config" in argv:
        idx = argv.index("--config")
        del argv[idx:idx+2]
    sys.argv = [sys.argv[0]] + argv

    # Use a fixed hydra run dir: runs/<exp_name> (so chunked runs keep writing to same folder)
    # The config file also sets hydra.run.dir, but we enforce a safe default here.
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")

    _hydra_entrypoint(config_name=config_name)


def _hydra_entrypoint(config_name: str):
    @hydra.main(version_base=None, config_path="configs", config_name=config_name)
    def _run(cfg: DictConfig) -> None:
        # Ensure float32 obs/action as required
        set_global_seeds(int(cfg.seed), deterministic=bool(cfg.deterministic))

        trainer = Trainer(cfg)
        print()
        trainer.run()

    _run()


if __name__ == "__main__":
    main()
