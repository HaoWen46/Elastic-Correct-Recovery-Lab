from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import models, transforms

from ecrl.ckpt.atomic_writer import read_latest
from ecrl.ckpt.blocking import BlockingPeriodicCheckpointer
from ecrl.ckpt.overlapped import OverlappedPeriodicCheckpointer
from ecrl.data import CIFAR10WithIDs, CIFAR100WithIDs, FakeDataWithIDs, ImageFolderWithIDs
from ecrl.sampler.resumable_sampler import ResumableGlobalBatchSampler
from ecrl.statepack.statepack import capture_state, load_payload, restore_state


@dataclass
class Runtime:
    rank: int
    world_size: int
    local_rank: int
    is_distributed: bool
    device: torch.device


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", encoding="utf-8")

    def log(self, record: Dict[str, Any]) -> None:
        self._f.write(json.dumps(record, sort_keys=True) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def _parse_fail_steps(raw: str | None, fallback: Iterable[int]) -> List[int]:
    if raw is None:
        return sorted(int(x) for x in fallback)
    raw = raw.strip()
    if not raw:
        return []
    return sorted(int(part) for part in raw.split(",") if part.strip())


def _hash_ids(sample_ids: torch.Tensor) -> str:
    arr = sample_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _init_runtime() -> Runtime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_distributed:
        backend = "nccl" if device.type == "cuda" else "gloo"
        if backend == "nccl":
            # Pin NCCL to the local CUDA device to avoid ambiguous rank->GPU mapping.
            try:
                dist.init_process_group(backend=backend, device_id=local_rank)
            except TypeError:
                # Older torch builds may not support init_process_group(device_id=...).
                dist.init_process_group(backend=backend)
        else:
            dist.init_process_group(backend=backend)

    return Runtime(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_distributed=is_distributed,
        device=device,
    )


def _cleanup_runtime(rt: Runtime) -> None:
    if rt.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _barrier(rt: Runtime) -> None:
    if rt.is_distributed and dist.is_initialized():
        if rt.device.type == "cuda":
            dist.barrier(device_ids=[rt.local_rank])
        else:
            dist.barrier()


def _load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)
_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class SmallCifarNet(nn.Module):
    """A lightweight CNN for fast evaluation-focused runs on 32x32 inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _build_transforms(
    dataset_name: str,
    image_size: int,
    *,
    train: bool,
    use_augmentation: bool,
) -> transforms.Compose:
    if dataset_name == "cifar10":
        ops: List[Any] = []
        if train and use_augmentation:
            ops.extend(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
            ]
        )
        return transforms.Compose(ops)
    if dataset_name == "cifar100":
        ops: List[Any] = []
        if train and use_augmentation:
            ops.extend(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
            ]
        )
        return transforms.Compose(ops)
    if dataset_name == "imagefolder":
        if train and use_augmentation:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )
    if dataset_name == "fake":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
            ]
        )
    raise ValueError(f"unsupported dataset.name: {dataset_name}")


def _prepare_dataset(
    *,
    dataset_cfg: Dict[str, Any],
    rt: Runtime,
    transform: transforms.Compose,
) -> Tuple[Any, int]:
    dataset_name = str(dataset_cfg.get("name", "cifar10")).lower()
    dataset_root = dataset_cfg.get("root", "./data")
    dataset_train = bool(dataset_cfg.get("train", True))
    dataset_download = bool(dataset_cfg.get("download", True))

    if dataset_name == "cifar10":
        if dataset_download and rt.rank == 0:
            CIFAR10WithIDs(
                root=dataset_root,
                train=dataset_train,
                transform=transform,
                download=True,
            )
        _barrier(rt)
        dataset = CIFAR10WithIDs(
            root=dataset_root,
            train=dataset_train,
            transform=transform,
            download=False,
        )
        return dataset, 10

    if dataset_name == "cifar100":
        if dataset_download and rt.rank == 0:
            CIFAR100WithIDs(
                root=dataset_root,
                train=dataset_train,
                transform=transform,
                download=True,
            )
        _barrier(rt)
        dataset = CIFAR100WithIDs(
            root=dataset_root,
            train=dataset_train,
            transform=transform,
            download=False,
        )
        return dataset, 100

    if dataset_name == "imagefolder":
        split_subdir = str(dataset_cfg.get("split_subdir", "train"))
        folder = Path(dataset_root) / split_subdir
        dataset = ImageFolderWithIDs(root=str(folder), transform=transform)
        return dataset, len(dataset.classes)

    if dataset_name == "fake":
        image_size = int(dataset_cfg.get("image_size", 32))
        num_classes = int(dataset_cfg.get("num_classes", 10))
        dataset = FakeDataWithIDs(
            size=int(dataset_cfg.get("size", 50_000)),
            image_size=(3, image_size, image_size),
            num_classes=num_classes,
            transform=transform,
        )
        return dataset, num_classes

    raise ValueError(f"unsupported dataset.name: {dataset_name}")


def _build_model(model_name: str, num_classes: int, image_size: int) -> nn.Module:
    name = model_name.lower()
    if name == "small_cnn":
        if image_size != 32:
            raise ValueError("small_cnn expects image_size=32")
        return SmallCifarNet()
    if name == "resnet18":
        return models.resnet18(weights=None, num_classes=num_classes)
    if name == "resnet34":
        return models.resnet34(weights=None, num_classes=num_classes)
    if name == "resnet50":
        return models.resnet50(weights=None, num_classes=num_classes)
    if name == "efficientnet_b0":
        return models.efficientnet_b0(weights=None, num_classes=num_classes)
    if name == "mobilenet_v3_large":
        return models.mobilenet_v3_large(weights=None, num_classes=num_classes)
    raise ValueError(
        f"unsupported training.model_name: {model_name}. "
        "Choose one of: small_cnn,resnet18,resnet34,resnet50,efficientnet_b0,mobilenet_v3_large"
    )


def _resolve_resume_path(args: argparse.Namespace, checkpoint_dir: Path) -> str | None:
    if args.resume_path:
        return args.resume_path
    if args.resume_latest:
        latest = read_latest(checkpoint_dir)
        if latest is None:
            return None
        return latest["path"]
    return None


def _autocast_and_scaler(
    *,
    precision: str,
    device: torch.device,
) -> Tuple[Any, torch.cuda.amp.GradScaler | None]:
    p = precision.lower()
    if p not in ("fp32", "bf16", "fp16"):
        raise ValueError("training.precision must be one of: fp32,bf16,fp16")

    if device.type != "cuda" or p == "fp32":
        return nullcontext, None
    if p == "bf16":
        return (
            lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            None,
        )
    return (
        lambda: torch.autocast(device_type="cuda", dtype=torch.float16),
        torch.cuda.amp.GradScaler(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ECRL DDP trainer")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--checkpoint-strategy", type=str, choices=["blocking", "overlapped"], default=None)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--target-steps", type=int, default=None)
    parser.add_argument("--fail-steps", type=str, default=None)
    parser.add_argument("--disable-failure", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--max-inflight", type=int, default=None)
    args = parser.parse_args()

    rt = _init_runtime()
    try:
        cfg = _load_config(args.config)

        seed = int(args.seed if args.seed is not None else cfg.get("seed", 1337))
        _seed_everything(seed)

        training_cfg = cfg.get("training", {})
        ckpt_cfg = cfg.get("checkpoint", {})
        failure_cfg = cfg.get("failure", {})
        dataset_cfg = cfg.get("dataset", {})

        global_batch = int(training_cfg["global_batch"])
        if global_batch % rt.world_size != 0:
            raise ValueError(
                f"GLOBAL_BATCH ({global_batch}) must divide world_size ({rt.world_size})"
            )
        local_batch = global_batch // rt.world_size

        target_steps = int(
            args.target_steps if args.target_steps is not None else training_cfg["max_steps"]
        )

        strategy = args.checkpoint_strategy or ckpt_cfg.get("strategy", "blocking")
        every_k = int(args.checkpoint_every or ckpt_cfg.get("every_k_steps", 50))
        max_inflight = int(args.max_inflight or ckpt_cfg.get("max_inflight", 4))

        failure_enabled = bool(failure_cfg.get("enabled", False)) and not args.disable_failure
        fail_steps = set(_parse_fail_steps(args.fail_steps, failure_cfg.get("steps", [])))
        dataset_name = str(dataset_cfg.get("name", "cifar10")).lower()
        dataset_train = bool(dataset_cfg.get("train", True))
        use_augmentation = bool(training_cfg.get("use_augmentation", False))
        image_size = int(dataset_cfg.get("image_size", 32 if dataset_name in ("cifar10", "cifar100", "fake") else 224))
        transform = _build_transforms(
            dataset_name=dataset_name,
            image_size=image_size,
            train=dataset_train,
            use_augmentation=use_augmentation,
        )
        dataset, num_classes = _prepare_dataset(dataset_cfg=dataset_cfg, rt=rt, transform=transform)

        dataset_size = len(dataset)
        if "size" in dataset_cfg and int(dataset_cfg["size"]) != dataset_size:
            raise ValueError(
                f"dataset size mismatch: config={dataset_cfg['size']} actual={dataset_size}"
            )

        sampler = ResumableGlobalBatchSampler(
            dataset_size=dataset_size,
            global_batch=global_batch,
            world_size=rt.world_size,
            rank=rt.rank,
            seed=seed,
            epoch=0,
            cursor_step=0,
        )
        steps_per_epoch = sampler.steps_per_epoch

        model_name = str(training_cfg.get("model_name", "small_cnn"))
        model = _build_model(model_name=model_name, num_classes=num_classes, image_size=image_size).to(rt.device)
        if rt.is_distributed:
            ddp_model = DDP(
                model,
                device_ids=[rt.local_rank] if rt.device.type == "cuda" else None,
                output_device=rt.local_rank if rt.device.type == "cuda" else None,
            )
        else:
            ddp_model = DDP(model) if dist.is_initialized() else model

        optimizer = optim.SGD(
            ddp_model.parameters(),
            lr=float(training_cfg.get("lr", 0.1)),
            momentum=float(training_cfg.get("momentum", 0.9)),
            weight_decay=float(training_cfg.get("weight_decay", 5e-4)),
        )
        scheduler_name = str(training_cfg.get("scheduler", "none")).lower()
        if scheduler_name == "none":
            scheduler = None
        elif scheduler_name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(training_cfg.get("scheduler_t_max", target_steps)),
                eta_min=float(training_cfg.get("scheduler_eta_min", 0.0)),
            )
        else:
            raise ValueError("training.scheduler must be one of: none,cosine")

        precision = str(training_cfg.get("precision", "fp32")).lower()
        autocast_ctx, grad_scaler = _autocast_and_scaler(precision=precision, device=rt.device)
        max_grad_norm = float(training_cfg.get("max_grad_norm", 0.0))
        criterion = nn.CrossEntropyLoss().to(rt.device)

        run_root = Path(args.results_dir)
        logs_dir = run_root / "logs" / args.run_id
        ckpt_dir = run_root / "checkpoints" / args.run_id
        rank_log = JsonlLogger(logs_dir / f"rank{rt.rank}.jsonl")
        debug_log = JsonlLogger(logs_dir / "debug_rank0_ids.jsonl") if rt.rank == 0 else None
        ckpt_metrics_log = JsonlLogger(logs_dir / "checkpoint_rank0.jsonl") if rt.rank == 0 else None

        def ckpt_metrics_logger(record: Dict[str, Any]) -> None:
            if ckpt_metrics_log is not None:
                rec = dict(record)
                rec["time"] = time.time()
                rec["run_id"] = args.run_id
                rec["rank"] = rt.rank
                ckpt_metrics_log.log(rec)

        if strategy == "blocking":
            checkpointer: Any = BlockingPeriodicCheckpointer(
                every_k_steps=every_k,
                checkpoint_dir=ckpt_dir,
                rank=rt.rank,
                world_size=rt.world_size,
                metrics_logger=ckpt_metrics_logger,
            )
        elif strategy == "overlapped":
            checkpointer = OverlappedPeriodicCheckpointer(
                every_k_steps=every_k,
                max_inflight=max_inflight,
                checkpoint_dir=ckpt_dir,
                rank=rt.rank,
                world_size=rt.world_size,
                metrics_logger=ckpt_metrics_logger,
            )
        else:
            raise ValueError(f"unknown checkpoint strategy: {strategy}")

        epoch = 0
        global_step = 0

        resume_path = _resolve_resume_path(args, ckpt_dir)
        if resume_path:
            payload = load_payload(resume_path, map_location=rt.device)
            step_state = restore_state(
                payload,
                model=ddp_model.module if isinstance(ddp_model, DDP) else ddp_model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
            )
            epoch = int(step_state["epoch"])
            global_step = int(step_state["global_step"])
            sampler.cursor_step = int(step_state["cursor_step"])

        _barrier(rt)

        debug_every = int(training_cfg.get("debug_every", 100))

        while global_step < target_steps:
            if sampler.cursor_step >= steps_per_epoch:
                epoch += 1
                sampler.cursor_step = 0

            sampler.set_epoch(epoch)
            if sampler.cursor_step < 0 or sampler.cursor_step > steps_per_epoch:
                raise RuntimeError(
                    f"invalid cursor_step {sampler.cursor_step} for steps_per_epoch {steps_per_epoch}"
                )

            dataloader = DataLoader(
                dataset,
                batch_size=local_batch,
                sampler=sampler,
                num_workers=0,
                drop_last=True,
                pin_memory=rt.device.type == "cuda",
            )
            data_iter = iter(dataloader)

            start_cursor = sampler.cursor_step
            for step_in_epoch in range(steps_per_epoch):
                if step_in_epoch < start_cursor:
                    continue
                if global_step >= target_steps:
                    break

                x, y, sample_ids = next(data_iter)
                x = x.to(rt.device, non_blocking=True)
                y = y.to(rt.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with autocast_ctx():
                    logits = ddp_model(x)
                    loss = criterion(logits, y)

                if grad_scaler is not None:
                    grad_scaler.scale(loss).backward()
                    if max_grad_norm > 0.0:
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_grad_norm)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    if max_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_grad_norm)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                consumed_cursor = sampler.cursor_step
                sampler.advance_step()
                global_step += 1

                step_record = {
                    "time": time.time(),
                    "run_id": args.run_id,
                    "rank": rt.rank,
                    "world_size": rt.world_size,
                    "epoch": epoch,
                    "global_step": global_step,
                    "cursor_step": consumed_cursor,
                    "loss": float(loss.detach().cpu().item()),
                    "sample_ids_hash": _hash_ids(sample_ids),
                    "sample_ids_count": int(sample_ids.numel()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "precision": precision,
                }
                rank_log.log(step_record)

                if rt.rank == 0 and debug_log is not None and global_step % debug_every == 0:
                    debug_log.log(
                        {
                            "time": time.time(),
                            "run_id": args.run_id,
                            "epoch": epoch,
                            "global_step": global_step,
                            "cursor_step": consumed_cursor,
                            "sample_ids": [int(v) for v in sample_ids.detach().cpu().tolist()],
                        }
                    )

                checkpointer.maybe_checkpoint(
                    global_step=global_step,
                    epoch=epoch,
                    capture_fn=lambda: capture_state(
                        model=ddp_model.module if isinstance(ddp_model, DDP) else ddp_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        sampler=sampler,
                        epoch=epoch,
                        global_step=global_step,
                        cursor_step=sampler.cursor_step,
                    ),
                )

                if failure_enabled and global_step in fail_steps:
                    # Ensure asynchronous checkpoint writes are durably persisted before
                    # an injected crash at this step boundary.
                    if hasattr(checkpointer, "flush"):
                        checkpointer.flush(wait=True)
                    _barrier(rt)
                    if rt.rank == 0:
                        os._exit(137)

            if sampler.cursor_step >= steps_per_epoch:
                epoch += 1
                sampler.cursor_step = 0

        if hasattr(checkpointer, "close"):
            checkpointer.close(wait=True)

        if rt.rank == 0:
            final_payload = capture_state(
                model=ddp_model.module if isinstance(ddp_model, DDP) else ddp_model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                epoch=epoch,
                global_step=global_step,
                cursor_step=sampler.cursor_step,
            )
            from ecrl.ckpt.atomic_writer import write_atomic_checkpoint

            write_atomic_checkpoint(
                payload=final_payload,
                checkpoint_dir=ckpt_dir,
                global_step=global_step,
                epoch=epoch,
                world_size_at_save=rt.world_size,
            )

        _barrier(rt)

        rank_log.close()
        if debug_log is not None:
            debug_log.close()
        if ckpt_metrics_log is not None:
            ckpt_metrics_log.close()

    finally:
        _cleanup_runtime(rt)


if __name__ == "__main__":
    main()
