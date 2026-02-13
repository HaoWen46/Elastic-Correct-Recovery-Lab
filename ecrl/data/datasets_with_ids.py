from __future__ import annotations

from typing import Tuple

from torchvision.datasets import CIFAR10, CIFAR100, FakeData, ImageFolder


class CIFAR10WithIDs(CIFAR10):
    """CIFAR10 variant that returns sample IDs (dataset indices)."""

    def __getitem__(self, index: int) -> Tuple[object, int, int]:
        x, y = super().__getitem__(index)
        return x, y, index


class CIFAR100WithIDs(CIFAR100):
    """CIFAR100 variant that returns sample IDs (dataset indices)."""

    def __getitem__(self, index: int) -> Tuple[object, int, int]:
        x, y = super().__getitem__(index)
        return x, y, index


class ImageFolderWithIDs(ImageFolder):
    """ImageFolder variant that returns sample IDs (dataset indices)."""

    def __getitem__(self, index: int) -> Tuple[object, int, int]:
        x, y = super().__getitem__(index)
        return x, y, index


class FakeDataWithIDs(FakeData):
    """FakeData variant that mirrors (x, y, sample_id) contract."""

    def __getitem__(self, index: int) -> Tuple[object, int, int]:
        x, y = super().__getitem__(index)
        return x, y, index
