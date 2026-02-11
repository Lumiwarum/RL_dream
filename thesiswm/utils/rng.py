from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch

def _to_byte_tensor(x: Any) -> torch.ByteTensor:
    """
    Coerce various serialized RNG state formats into torch.ByteTensor.
    Handles: torch.Tensor, numpy array, list of ints/bytes, bytes/bytearray.
    """
    if x is None:
        raise ValueError("RNG state is None")
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.uint8, device="cpu")
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.uint8, copy=False))
    if isinstance(x, (bytes, bytearray)):
        return torch.tensor(list(x), dtype=torch.uint8)
    if isinstance(x, (list, tuple)):
        return torch.tensor(list(x), dtype=torch.uint8)
    # last resort: try torch.tensor
    return torch.tensor(x, dtype=torch.uint8)


def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])

    torch_cpu = _to_byte_tensor(state["torch_cpu"])
    torch.set_rng_state(torch_cpu)

    if torch.cuda.is_available() and state.get("torch_cuda_all") is not None:
        cuda_states = []
        for s in state["torch_cuda_all"]:
            cuda_states.append(_to_byte_tensor(s))
        torch.cuda.set_rng_state_all(cuda_states)
