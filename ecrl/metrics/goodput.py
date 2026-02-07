from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="ECRL goodput metric")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--target-steps", type=int, default=None)
    args = parser.parse_args()

    logs_dir = Path(args.results_dir) / "logs" / args.run_id
    rank0 = _load_jsonl(logs_dir / "rank0.jsonl")
    ckpt = _load_jsonl(logs_dir / "checkpoint_rank0.jsonl")

    if not rank0:
        raise RuntimeError(f"no rank0 logs found: {logs_dir / 'rank0.jsonl'}")

    t0 = float(rank0[0]["time"])
    t1 = float(rank0[-1]["time"])
    wall = max(t1 - t0, 1e-9)

    committed_by_step: Dict[int, Dict[str, Any]] = {}
    for rec in rank0:
        committed_by_step[int(rec["global_step"])] = rec

    unique_steps = sorted(committed_by_step)
    final_step = unique_steps[-1] if unique_steps else 0
    useful_steps = int(args.target_steps if args.target_steps is not None else final_step)
    useful_steps = min(useful_steps, final_step)

    replayed_steps = len(rank0) - len(unique_steps)

    detected_restarts = 0
    prev = None
    for rec in rank0:
        cur = int(rec["global_step"])
        if prev is not None and cur < prev:
            detected_restarts += 1
        prev = cur

    snapshot_total = 0.0
    write_total = 0.0
    enqueue_total = 0.0
    backpressure_total = 0.0
    stall_total = 0.0
    for rec in ckpt:
        snapshot_total += float(rec.get("snapshot_time_sec", 0.0))
        write_total += float(rec.get("write_time_sec", 0.0))
        enqueue_total += float(rec.get("enqueue_latency_sec", 0.0))
        backpressure_total += float(rec.get("backpressure_wait_sec", 0.0))
        stall_total += float(rec.get("stall_time_sec", 0.0))

    restarts = detected_restarts
    supervisor_path = logs_dir / "supervisor.json"
    if supervisor_path.exists():
        supervisor = json.loads(supervisor_path.read_text(encoding="utf-8"))
        restarts = int(supervisor.get("restarts", restarts))

    out = {
        "run_id": args.run_id,
        "wall_clock_time_sec": wall,
        "useful_steps": useful_steps,
        "final_step": final_step,
        "goodput_steps_per_sec": useful_steps / wall,
        "restarts": restarts,
        "replayed_steps": replayed_steps,
        "checkpoint": {
            "num_events": len(ckpt),
            "snapshot_time_sec_total": snapshot_total,
            "write_time_sec_total": write_total,
            "enqueue_latency_sec_total": enqueue_total,
            "backpressure_wait_sec_total": backpressure_total,
            "stall_time_sec_total": stall_total,
        },
    }

    if supervisor_path.exists():
        out["supervisor"] = supervisor

    out_dir = Path(args.results_dir) / "metrics" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "goodput.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
