from __future__ import annotations

import random
from typing import Any, Dict, List

import numpy as np
import torch


RNGState = Dict[str, Any]


def _as_cpu_byte_tensor(value: Any) -> torch.Tensor:
    t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if t.device.type != "cpu":
        t = t.detach().cpu()
    if t.dtype != torch.uint8:
        t = t.to(dtype=torch.uint8)
    return t.contiguous()


def capture_rng_state() -> RNGState:
    state: RNGState = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: RNGState) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_as_cpu_byte_tensor(state["torch_cpu"]))

    cuda_states: List[Any] | None = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_as_cpu_byte_tensor(s) for s in cuda_states])
