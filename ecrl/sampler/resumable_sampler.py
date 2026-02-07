from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


@dataclass
class EpochGeometry:
    dataset_size: int
    global_batch: int

    @property
    def steps_per_epoch(self) -> int:
        return self.dataset_size // self.global_batch

    @property
    def epoch_samples(self) -> int:
        return self.steps_per_epoch * self.global_batch


class ResumableGlobalBatchSampler(Sampler[int]):
    """Sampler that maps global step windows to rank-local contiguous slices.

    The sampler enforces fixed epoch geometry based on GLOBAL_BATCH, independent of
    world size. It supports resume (cursor_step) and elastic N->M resume by using
    current runtime world_size/rank when slicing each global batch window.
    """

    def __init__(
        self,
        dataset_size: int,
        global_batch: int,
        world_size: int,
        rank: int,
        seed: int,
        epoch: int = 0,
        cursor_step: int = 0,
    ) -> None:
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if global_batch <= 0:
            raise ValueError(f"global_batch must be positive, got {global_batch}")
        if global_batch % world_size != 0:
            raise ValueError(
                f"GLOBAL_BATCH ({global_batch}) must divide world_size ({world_size})"
            )
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank {rank} out of range for world_size {world_size}")

        self.geometry = EpochGeometry(dataset_size=dataset_size, global_batch=global_batch)
        self.world_size = world_size
        self.rank = rank
        self.local_batch = global_batch // world_size
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.cursor_step = int(cursor_step)

        if self.cursor_step < 0 or self.cursor_step > self.steps_per_epoch:
            raise ValueError(
                f"cursor_step must be in [0, {self.steps_per_epoch}], got {self.cursor_step}"
            )

        self._cached_epoch: int | None = None
        self._cached_perm_epoch: List[int] | None = None

    @property
    def dataset_size(self) -> int:
        return self.geometry.dataset_size

    @property
    def global_batch(self) -> int:
        return self.geometry.global_batch

    @property
    def steps_per_epoch(self) -> int:
        return self.geometry.steps_per_epoch

    @property
    def epoch_samples(self) -> int:
        return self.geometry.epoch_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._cached_epoch = None
        self._cached_perm_epoch = None

    def state_dict(self) -> Dict[str, int]:
        return {
            "epoch": int(self.epoch),
            "cursor_step": int(self.cursor_step),
            "seed": int(self.seed),
        }

    def load_state_dict(self, state: Dict[str, int]) -> None:
        self.epoch = int(state["epoch"])
        self.cursor_step = int(state["cursor_step"])
        self.seed = int(state["seed"])
        if self.cursor_step < 0 or self.cursor_step > self.steps_per_epoch:
            raise ValueError(
                f"cursor_step must be in [0, {self.steps_per_epoch}], got {self.cursor_step}"
            )
        self._cached_epoch = None
        self._cached_perm_epoch = None

    def _epoch_seed(self, epoch: int) -> int:
        # Deterministic tuple seed composition.
        return (self.seed * 1_000_003 + int(epoch)) & 0xFFFFFFFFFFFF

    def _perm_epoch(self) -> List[int]:
        if self._cached_epoch == self.epoch and self._cached_perm_epoch is not None:
            return self._cached_perm_epoch

        g = torch.Generator(device="cpu")
        g.manual_seed(self._epoch_seed(self.epoch))
        perm = torch.randperm(self.dataset_size, generator=g).tolist()
        perm_epoch = perm[: self.epoch_samples]

        self._cached_epoch = self.epoch
        self._cached_perm_epoch = perm_epoch
        return perm_epoch

    def local_ids_for(self, epoch: int, cursor_step: int, rank: int, world_size: int) -> List[int]:
        if self.global_batch % world_size != 0:
            raise ValueError(
                f"GLOBAL_BATCH ({self.global_batch}) must divide world_size ({world_size})"
            )
        local_batch = self.global_batch // world_size

        old_epoch = self.epoch
        old_cached_epoch = self._cached_epoch
        old_cached_perm_epoch = self._cached_perm_epoch
        try:
            self.epoch = int(epoch)
            perm_epoch = self._perm_epoch()
        finally:
            self.epoch = old_epoch
            self._cached_epoch = old_cached_epoch
            self._cached_perm_epoch = old_cached_perm_epoch

        start = cursor_step * self.global_batch
        end = (cursor_step + 1) * self.global_batch
        window = perm_epoch[start:end]

        lstart = rank * local_batch
        lend = (rank + 1) * local_batch
        return window[lstart:lend]

    def advance_step(self) -> int:
        """Advance cursor_step after one completed training step.

        Rank 0 is authoritative and broadcasts the updated cursor_step.
        """
        if dist.is_available() and dist.is_initialized():
            backend = dist.get_backend()
            if backend == "nccl":
                t = torch.tensor(
                    [self.cursor_step],
                    dtype=torch.int64,
                    device=torch.device("cuda", torch.cuda.current_device()),
                )
            else:
                t = torch.tensor([self.cursor_step], dtype=torch.int64)
            if dist.get_rank() == 0:
                t += 1
            dist.broadcast(t, src=0)
            self.cursor_step = int(t.item())
        else:
            self.cursor_step += 1

        if self.cursor_step > self.steps_per_epoch:
            raise RuntimeError(
                f"cursor_step overflow: {self.cursor_step} > steps_per_epoch {self.steps_per_epoch}"
            )
        return self.cursor_step

    def __iter__(self) -> Iterator[int]:
        perm_epoch = self._perm_epoch()
        for step in range(self.cursor_step, self.steps_per_epoch):
            start = step * self.global_batch
            end = (step + 1) * self.global_batch
            window = perm_epoch[start:end]

            lstart = self.rank * self.local_batch
            lend = (self.rank + 1) * self.local_batch
            local_ids = window[lstart:lend]
            for idx in local_ids:
                yield idx

    def __len__(self) -> int:
        return (self.steps_per_epoch - self.cursor_step) * self.local_batch
