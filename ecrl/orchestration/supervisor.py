from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from ecrl.ckpt.atomic_writer import read_latest


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _run_cmd(cmd: List[str], *, output_log_path: Path) -> int:
    output_log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    with output_log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"$ {shlex.join(cmd)}\n")
        logf.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
        proc.stdout.close()
        return int(proc.wait())


def _decode_return_code(ret: int) -> Dict[str, Any]:
    if ret < 0:
        sig = int(-ret)
        try:
            sig_name = signal.Signals(sig).name
        except ValueError:
            sig_name = f"SIG{sig}"
        return {
            "kind": "signal",
            "signal": sig,
            "signal_name": sig_name,
        }
    return {
        "kind": "exit",
        "exit_code": int(ret),
    }


def _tail_text(path: Path, *, max_bytes: int = 64 * 1024) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes), os.SEEK_SET)
        return f.read().decode("utf-8", errors="replace")


def _infer_failure_hint(ret: int, *, attempt_log_path: Path) -> str | None:
    if ret == 0:
        return None
    tail = _tail_text(attempt_log_path)
    if not tail:
        return None
    if "SignalException" in tail and "signal: 1" in tail:
        return "torchrun_received_sighup"
    if "Received 1 death signal" in tail:
        return "elastic_agent_received_sighup"
    if "CUDA out of memory" in tail:
        return "cuda_oom"
    if "RNG state must be a torch.ByteTensor" in tail:
        return "rng_state_type_mismatch"
    if "Address already in use" in tail or "EADDRINUSE" in tail:
        return "master_port_in_use"
    return None


def _checkpoint_snapshot(ckpt_dir: Path, latest: Dict[str, Any] | None) -> Dict[str, Any]:
    latest_path = str(latest.get("path")) if latest is not None else None
    latest_path_exists = bool(latest_path) and Path(latest_path).exists()
    checkpoint_count = sum(1 for _ in ckpt_dir.glob("step_*.pt")) if ckpt_dir.exists() else 0
    return {
        "ckpt_dir": str(ckpt_dir),
        "latest_json_path": str(ckpt_dir / "latest.json"),
        "latest_exists": latest is not None,
        "latest_path": latest_path,
        "latest_path_exists": latest_path_exists,
        "checkpoint_file_count": checkpoint_count,
    }


def _build_cmd(args: argparse.Namespace, resume_latest: bool) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--master_addr",
        args.master_addr,
        "--master_port",
        str(args.master_port),
        "-m",
        "ecrl.train.ddp_train",
        "--config",
        args.config,
        "--run-id",
        args.run_id,
        "--results-dir",
        args.results_dir,
        "--target-steps",
        str(args.target_steps),
        "--checkpoint-strategy",
        args.checkpoint_strategy,
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--max-inflight",
        str(args.max_inflight),
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])

    if args.fail_steps:
        cmd.extend(["--fail-steps", args.fail_steps])
    if args.disable_failure:
        cmd.append("--disable-failure")
    if resume_latest:
        cmd.append("--resume-latest")

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="ECRL restart supervisor")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--target-steps", type=int, required=True)
    parser.add_argument("--checkpoint-strategy", type=str, choices=["blocking", "overlapped"], default="blocking")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--max-inflight", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fail-steps", type=str, default="")
    parser.add_argument("--disable-failure", action="store_true")
    parser.add_argument("--max-restarts", type=int, default=20)
    parser.add_argument("--max-restarts-without-checkpoint", type=int, default=3)
    parser.add_argument("--restart-delay-sec", type=float, default=1.0)
    parser.add_argument("--start-resume-latest", action="store_true")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    ckpt_dir = results_root / "checkpoints" / args.run_id
    logs_dir = results_root / "logs" / args.run_id
    logs_dir.mkdir(parents=True, exist_ok=True)
    supervisor_path = logs_dir / "supervisor.json"
    attempts_path = logs_dir / "supervisor_attempts.jsonl"

    attempts = 0
    restarts = 0
    restarts_without_checkpoint_total = 0
    restarts_without_checkpoint_streak = 0
    resume_latest = bool(args.start_resume_latest)
    start_time = time.time()

    while True:
        attempts += 1
        cmd = _build_cmd(args, resume_latest=resume_latest)
        attempt_log_path = logs_dir / f"attempt_{attempts:03d}.log"
        attempt_start = time.time()
        _append_jsonl(
            attempts_path,
            {
                "time": attempt_start,
                "event": "attempt_start",
                "attempt": attempts,
                "resume_latest": resume_latest,
                "cmd": cmd,
                "cmd_shell": shlex.join(cmd),
                "attempt_log_path": str(attempt_log_path),
            },
        )
        ret = _run_cmd(cmd, output_log_path=attempt_log_path)

        latest = read_latest(ckpt_dir)
        latest_step = int(latest["global_step"]) if latest is not None else 0
        ret_info = _decode_return_code(ret)
        failure_hint = _infer_failure_hint(ret, attempt_log_path=attempt_log_path)
        ckpt_info = _checkpoint_snapshot(ckpt_dir, latest)
        latest_path_exists = bool(ckpt_info["latest_path_exists"])
        attempt_duration = time.time() - attempt_start

        status: Dict[str, Any] = {
            "time": time.time(),
            "attempt": attempts,
            "attempts": attempts,
            "restarts": restarts,
            "restarts_without_checkpoint_total": restarts_without_checkpoint_total,
            "restarts_without_checkpoint_streak": restarts_without_checkpoint_streak,
            "last_return_code": ret,
            "last_return": ret_info,
            "failure_hint": failure_hint,
            "latest_step": latest_step,
            "target_steps": int(args.target_steps),
            "resume_latest": resume_latest,
            "attempt_duration_sec": attempt_duration,
            "attempt_log_path": str(attempt_log_path),
            "checkpoint_snapshot": ckpt_info,
        }
        supervisor_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
        _append_jsonl(
            attempts_path,
            {
                **status,
                "event": "attempt_end",
            },
        )

        if ret == 0:
            break

        if latest is not None and not latest_path_exists:
            raise RuntimeError(
                "latest checkpoint pointer exists but target checkpoint file is missing: "
                f"path={ckpt_info['latest_path']}"
            )

        if latest is None:
            restarts_without_checkpoint_total += 1
            restarts_without_checkpoint_streak += 1
            if restarts_without_checkpoint_streak > args.max_restarts_without_checkpoint:
                raise RuntimeError(
                    "training failed without any checkpoint and exceeded "
                    f"max_restarts_without_checkpoint={args.max_restarts_without_checkpoint}. "
                    f"last_return={ret_info}, attempt_log={attempt_log_path}, "
                    f"latest_json={ckpt_info['latest_json_path']}"
                )
            restarts += 1
            resume_latest = False
            _append_jsonl(
                attempts_path,
                {
                    "time": time.time(),
                    "event": "restart",
                    "reason": "no_checkpoint_yet",
                    "attempt": attempts,
                    "restarts": restarts,
                    "restarts_without_checkpoint_total": restarts_without_checkpoint_total,
                    "restarts_without_checkpoint_streak": restarts_without_checkpoint_streak,
                    "resume_latest": resume_latest,
                },
            )
            time.sleep(args.restart_delay_sec)
            continue

        restarts_without_checkpoint_streak = 0

        if latest_step >= args.target_steps:
            break

        restarts += 1
        if restarts > args.max_restarts:
            raise RuntimeError(
                f"exceeded max restarts ({args.max_restarts}); last_step={latest_step}"
            )

        resume_latest = True
        _append_jsonl(
            attempts_path,
            {
                "time": time.time(),
                "event": "restart",
                "reason": "resume_latest",
                "attempt": attempts,
                "restarts": restarts,
                "restarts_without_checkpoint_total": restarts_without_checkpoint_total,
                "restarts_without_checkpoint_streak": restarts_without_checkpoint_streak,
                "resume_latest": resume_latest,
                "latest_step": latest_step,
            },
        )
        time.sleep(args.restart_delay_sec)

    end_summary = {
        "time": time.time(),
        "attempts": attempts,
        "restarts": restarts,
        "restarts_without_checkpoint_total": restarts_without_checkpoint_total,
        "restarts_without_checkpoint_streak": restarts_without_checkpoint_streak,
        "duration_sec": time.time() - start_time,
        "target_steps": int(args.target_steps),
        "status": "completed",
    }
    supervisor_path.write_text(json.dumps(end_summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
