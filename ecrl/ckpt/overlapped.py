from __future__ import annotations

import io
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Any, Callable, Dict

import torch
import torch.distributed as dist

from ecrl.ckpt.atomic_writer import write_atomic_checkpoint_bytes


CaptureFn = Callable[[], Dict[str, Any]]
MetricsFn = Callable[[Dict[str, Any]], None]


def _io_worker(task_q: mp.Queue, done_q: mp.Queue) -> None:
    while True:
        item = task_q.get()
        if item is None:
            break

        start = time.perf_counter()
        write_atomic_checkpoint_bytes(
            payload_bytes=item["payload_bytes"],
            checkpoint_dir=item["checkpoint_dir"],
            global_step=item["global_step"],
            epoch=item["epoch"],
            world_size_at_save=item["world_size_at_save"],
        )
        duration = time.perf_counter() - start
        done_q.put(
            {
                "global_step": int(item["global_step"]),
                "write_time_sec": float(duration),
                "done_time": time.time(),
            }
        )


def _serialize_payload(payload: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


class OverlappedPeriodicCheckpointer:
    def __init__(
        self,
        *,
        every_k_steps: int,
        max_inflight: int,
        checkpoint_dir: str | Path,
        rank: int,
        world_size: int,
        metrics_logger: MetricsFn | None = None,
    ) -> None:
        self.every_k_steps = int(every_k_steps)
        self.max_inflight = max(1, int(max_inflight))
        self.checkpoint_dir = str(Path(checkpoint_dir))
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.metrics_logger = metrics_logger

        self._task_q: mp.Queue | None = None
        self._done_q: mp.Queue | None = None
        self._proc: mp.Process | None = None
        self._inflight: Dict[int, float] = {}

        if self.rank == 0:
            ctx = mp.get_context("spawn")
            self._task_q = ctx.Queue()
            self._done_q = ctx.Queue()
            self._proc = ctx.Process(target=_io_worker, args=(self._task_q, self._done_q), daemon=True)
            self._proc.start()

    def _ensure_worker_alive(self) -> None:
        if self.rank != 0:
            return
        if self._proc is None:
            return
        if self._proc.is_alive():
            return
        inflight_steps = sorted(int(s) for s in self._inflight.keys())
        raise RuntimeError(
            "overlapped checkpoint I/O worker exited unexpectedly "
            f"(exitcode={self._proc.exitcode}); pending_steps={inflight_steps}"
        )

    def _drain_done_nonblocking(self) -> Dict[int, float]:
        completions: Dict[int, float] = {}
        if self.rank != 0 or self._done_q is None:
            return completions

        while True:
            try:
                item = self._done_q.get_nowait()
            except queue.Empty:
                break
            step = int(item["global_step"])
            completions[step] = float(item["write_time_sec"])
            self._inflight.pop(step, None)
        return completions

    def _wait_for_one_completion(self, *, timeout_sec: float | None = None) -> Dict[str, Any] | None:
        if self._done_q is None:
            raise RuntimeError("done queue missing")
        try:
            if timeout_sec is None:
                item = self._done_q.get()
            else:
                item = self._done_q.get(timeout=float(timeout_sec))
        except queue.Empty:
            return None
        step = int(item["global_step"])
        self._inflight.pop(step, None)
        return item

    def maybe_checkpoint(
        self,
        *,
        global_step: int,
        epoch: int,
        capture_fn: CaptureFn,
    ) -> Dict[str, Any] | None:
        if self.every_k_steps <= 0 or global_step % self.every_k_steps != 0:
            if self.rank == 0:
                self._drain_done_nonblocking()
            return None

        stall_start = time.perf_counter()
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        snapshot_start = time.perf_counter()
        payload = capture_fn() if self.rank == 0 else None
        snapshot_time = time.perf_counter() - snapshot_start

        enqueue_latency = 0.0
        backpressure_wait = 0.0
        completed_write_times: Dict[int, float] = {}

        if self.rank == 0:
            if self._task_q is None:
                raise RuntimeError("task queue missing")

            completed_write_times.update(self._drain_done_nonblocking())

            wait_start = time.perf_counter()
            while len(self._inflight) >= self.max_inflight:
                item = self._wait_for_one_completion(timeout_sec=5.0)
                if item is None:
                    self._ensure_worker_alive()
                    if self.metrics_logger is not None:
                        self.metrics_logger(
                            {
                                "event": "checkpoint_backpressure_waiting",
                                "pending_count": int(len(self._inflight)),
                                "pending_steps": [int(s) for s in sorted(self._inflight.keys())[:16]],
                                "wait_time_sec": float(time.perf_counter() - wait_start),
                            }
                        )
                    continue
                completed_write_times[int(item["global_step"])] = float(item["write_time_sec"])
            backpressure_wait = time.perf_counter() - wait_start

            enqueue_start = time.perf_counter()
            payload_bytes = _serialize_payload(payload)
            self._task_q.put(
                {
                    "payload_bytes": payload_bytes,
                    "checkpoint_dir": self.checkpoint_dir,
                    "global_step": int(global_step),
                    "epoch": int(epoch),
                    "world_size_at_save": int(self.world_size),
                }
            )
            self._inflight[int(global_step)] = time.time()
            enqueue_latency = time.perf_counter() - enqueue_start

            completed_write_times.update(self._drain_done_nonblocking())

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        stall_time = time.perf_counter() - stall_start

        record = {
            "event": "checkpoint",
            "strategy": "overlapped",
            "global_step": int(global_step),
            "epoch": int(epoch),
            "snapshot_time_sec": float(snapshot_time),
            "enqueue_latency_sec": float(enqueue_latency),
            "backpressure_wait_sec": float(backpressure_wait),
            "completed_write_times_sec": completed_write_times,
            "stall_time_sec": float(stall_time),
            "inflight_after_enqueue": int(len(self._inflight)) if self.rank == 0 else 0,
        }
        if self.rank == 0 and self.metrics_logger is not None:
            self.metrics_logger(record)
        return record

    def close(self, wait: bool = True) -> None:
        if self.rank != 0:
            return
        if self._task_q is None or self._done_q is None or self._proc is None:
            return

        if wait:
            self.flush(wait=True)

        self._task_q.put(None)
        self._proc.join(timeout=30)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5)

    def flush(self, wait: bool = True) -> None:
        """Drain completed writes and optionally wait for all inflight writes."""
        if self.rank != 0:
            return
        if self._done_q is None:
            return

        completed = self._drain_done_nonblocking()
        if self.metrics_logger is not None:
            for step, write_time in completed.items():
                self.metrics_logger(
                    {
                        "event": "checkpoint_write_complete",
                        "global_step": int(step),
                        "write_time_sec": float(write_time),
                    }
                )

        if not wait:
            return

        wait_start = time.perf_counter()
        while self._inflight:
            item = self._wait_for_one_completion(timeout_sec=5.0)
            if item is None:
                self._ensure_worker_alive()
                if self.metrics_logger is not None:
                    self.metrics_logger(
                        {
                            "event": "checkpoint_flush_waiting",
                            "pending_count": int(len(self._inflight)),
                            "pending_steps": [int(s) for s in sorted(self._inflight.keys())[:16]],
                            "wait_time_sec": float(time.perf_counter() - wait_start),
                        }
                    )
                continue
            if self.metrics_logger is not None:
                self.metrics_logger(
                    {
                        "event": "checkpoint_write_complete",
                        "global_step": int(item["global_step"]),
                        "write_time_sec": float(item["write_time_sec"]),
                    }
                )
