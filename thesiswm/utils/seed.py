from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seeds(seed: int, deterministic: bool = True) -> None:
    """
    Seed python, numpy, torch (cpu+cuda) and set deterministic flags when possible.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms can throw on some ops; we keep it off by default.
