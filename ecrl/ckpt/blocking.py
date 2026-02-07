from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict

import torch.distributed as dist

from ecrl.ckpt.atomic_writer import write_atomic_checkpoint


CaptureFn = Callable[[], Dict[str, Any]]
MetricsFn = Callable[[Dict[str, Any]], None]


class BlockingPeriodicCheckpointer:
    def __init__(
        self,
        *,
        every_k_steps: int,
        checkpoint_dir: str | Path,
        rank: int,
        world_size: int,
        metrics_logger: MetricsFn | None = None,
    ) -> None:
        self.every_k_steps = int(every_k_steps)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.metrics_logger = metrics_logger

    def maybe_checkpoint(
        self,
        *,
        global_step: int,
        epoch: int,
        capture_fn: CaptureFn,
    ) -> Dict[str, Any] | None:
        if self.every_k_steps <= 0 or global_step % self.every_k_steps != 0:
            return None

        stall_start = time.perf_counter()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        snapshot_start = time.perf_counter()
        payload = capture_fn() if self.rank == 0 else None
        snapshot_time = time.perf_counter() - snapshot_start

        write_time = 0.0
        if self.rank == 0:
            write_start = time.perf_counter()
            write_atomic_checkpoint(
                payload=payload,
                checkpoint_dir=self.checkpoint_dir,
                global_step=global_step,
                epoch=epoch,
                world_size_at_save=self.world_size,
            )
            write_time = time.perf_counter() - write_start

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        stall_time = time.perf_counter() - stall_start

        record = {
            "event": "checkpoint",
            "strategy": "blocking",
            "global_step": int(global_step),
            "epoch": int(epoch),
            "snapshot_time_sec": float(snapshot_time),
            "write_time_sec": float(write_time),
            "stall_time_sec": float(stall_time),
        }
        if self.rank == 0 and self.metrics_logger is not None:
            self.metrics_logger(record)
        return record
