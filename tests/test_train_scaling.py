from __future__ import annotations

import unittest

import torch

from ecrl.train.ddp_train import _autocast_and_scaler, _build_model, _build_transforms


class TestTrainScalingHooks(unittest.TestCase):
    def test_build_resnet18_with_custom_num_classes(self) -> None:
        model = _build_model(model_name="resnet18", num_classes=100, image_size=32)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 100))

    def test_build_small_cnn_shape(self) -> None:
        model = _build_model(model_name="small_cnn", num_classes=10, image_size=32)
        x = torch.randn(4, 3, 32, 32)
        y = model(x)
        self.assertEqual(tuple(y.shape), (4, 10))

    def test_transforms_support_cifar100_and_imagefolder(self) -> None:
        t1 = _build_transforms(
            dataset_name="cifar100",
            image_size=32,
            train=True,
            use_augmentation=True,
        )
        t2 = _build_transforms(
            dataset_name="imagefolder",
            image_size=224,
            train=False,
            use_augmentation=False,
        )
        self.assertIsNotNone(t1)
        self.assertIsNotNone(t2)

    def test_autocast_scaler_fallback_on_cpu(self) -> None:
        ctx_factory, scaler = _autocast_and_scaler(precision="bf16", device=torch.device("cpu"))
        with ctx_factory():
            x = torch.tensor([1.0])
        self.assertIsNone(scaler)
        self.assertEqual(float(x.item()), 1.0)


if __name__ == "__main__":
    unittest.main()
