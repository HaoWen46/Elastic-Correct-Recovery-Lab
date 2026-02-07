from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import torch


def _fsync_file(path: Path) -> None:
    with path.open("rb") as f:
        os.fsync(f.fileno())


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def write_atomic_checkpoint(
    *,
    payload: Dict[str, Any],
    checkpoint_dir: str | Path,
    global_step: int,
    epoch: int,
    world_size_at_save: int,
) -> Dict[str, Any]:
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = f"step_{global_step:08d}.pt"
    final_path = ckpt_dir / ckpt_name
    fd, tmp_path = tempfile.mkstemp(prefix=f".{ckpt_name}.", dir=str(ckpt_dir))

    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
        _fsync_dir(ckpt_dir)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    latest = {
        "path": str(final_path),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "world_size_at_save": int(world_size_at_save),
        "timestamp": int(time.time()),
    }
    _atomic_write_json(ckpt_dir / "latest.json", latest)
    return latest


def write_atomic_checkpoint_bytes(
    *,
    payload_bytes: bytes,
    checkpoint_dir: str | Path,
    global_step: int,
    epoch: int,
    world_size_at_save: int,
) -> Dict[str, Any]:
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = f"step_{global_step:08d}.pt"
    final_path = ckpt_dir / ckpt_name
    fd, tmp_path = tempfile.mkstemp(prefix=f".{ckpt_name}.", dir=str(ckpt_dir))

    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
        _fsync_dir(ckpt_dir)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    latest = {
        "path": str(final_path),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "world_size_at_save": int(world_size_at_save),
        "timestamp": int(time.time()),
    }
    _atomic_write_json(ckpt_dir / "latest.json", latest)
    return latest


def read_latest(checkpoint_dir: str | Path) -> Dict[str, Any] | None:
    path = Path(checkpoint_dir) / "latest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
