from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from ecrl.ckpt.atomic_writer import read_latest


def _run_cmd(cmd: List[str]) -> int:
    return subprocess.call(cmd)


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
    parser.add_argument("--fail-steps", type=str, default="")
    parser.add_argument("--disable-failure", action="store_true")
    parser.add_argument("--max-restarts", type=int, default=20)
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

    attempts = 0
    restarts = 0
    resume_latest = bool(args.start_resume_latest)
    start_time = time.time()

    while True:
        attempts += 1
        cmd = _build_cmd(args, resume_latest=resume_latest)
        ret = _run_cmd(cmd)

        latest = read_latest(ckpt_dir)
        latest_step = int(latest["global_step"]) if latest is not None else 0

        status: Dict[str, Any] = {
            "time": time.time(),
            "attempts": attempts,
            "restarts": restarts,
            "last_return_code": ret,
            "latest_step": latest_step,
            "target_steps": int(args.target_steps),
            "resume_latest": resume_latest,
        }
        supervisor_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")

        if ret == 0:
            break

        if latest is None:
            raise RuntimeError(
                f"training failed with return code {ret}, and no latest checkpoint was found"
            )

        if latest_step >= args.target_steps:
            break

        restarts += 1
        if restarts > args.max_restarts:
            raise RuntimeError(
                f"exceeded max restarts ({args.max_restarts}); last_step={latest_step}"
            )

        resume_latest = True
        time.sleep(args.restart_delay_sec)

    end_summary = {
        "time": time.time(),
        "attempts": attempts,
        "restarts": restarts,
        "duration_sec": time.time() - start_time,
        "target_steps": int(args.target_steps),
        "status": "completed",
    }
    supervisor_path.write_text(json.dumps(end_summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
