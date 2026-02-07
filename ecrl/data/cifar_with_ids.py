from __future__ import annotations

from typing import Tuple

from torchvision.datasets import CIFAR10


class CIFAR10WithIDs(CIFAR10):
    """CIFAR10 variant that returns sample IDs (dataset indices)."""

    def __getitem__(self, index: int) -> Tuple[object, int, int]:
        x, y = super().__getitem__(index)
        return x, y, index
