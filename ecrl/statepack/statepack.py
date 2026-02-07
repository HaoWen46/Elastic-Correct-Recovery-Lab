from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from ecrl.statepack.rng import capture_rng_state, restore_rng_state


StepState = Dict[str, int]
Payload = Dict[str, Any]


def capture_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    sampler: Any,
    epoch: int,
    global_step: int,
    cursor_step: int,
) -> Payload:
    payload: Payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "sampler": sampler.state_dict(),
        "step_state": {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "cursor_step": int(cursor_step),
        },
        "rng": capture_rng_state(),
    }
    return payload


def save_payload(payload: Payload, path: str | Path) -> None:
    torch.save(payload, str(path))


def load_payload(path: str | Path, map_location: str | torch.device = "cpu") -> Payload:
    return torch.load(str(path), map_location=map_location, weights_only=False)


def restore_state(
    payload: Payload,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    sampler: Any,
) -> StepState:
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])

    scheduler_state = payload.get("scheduler")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    sampler.load_state_dict(payload["sampler"])
    restore_rng_state(payload["rng"])
    return payload["step_state"]
