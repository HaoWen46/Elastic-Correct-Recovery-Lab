from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from ecrl.sampler.resumable_sampler import ResumableGlobalBatchSampler


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


def _hash_ids(ids: List[int]) -> str:
    import numpy as np

    arr = np.asarray(ids, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _load_run_records(logs_dir: Path) -> Dict[Tuple[int, int], Dict[str, Any]]:
    # Keep the last record per (rank, global_step) to represent committed trajectory.
    by_rank_step: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for rank_path in sorted(logs_dir.glob("rank*.jsonl")):
        for rec in _load_jsonl(rank_path):
            key = (int(rec["rank"]), int(rec["global_step"]))
            by_rank_step[key] = rec
    return by_rank_step


def _perm_counter(
    sampler: ResumableGlobalBatchSampler,
    *,
    epoch: int,
    steps_in_epoch: int,
) -> Counter[int]:
    c: Counter[int] = Counter()
    for cursor in range(steps_in_epoch):
        ids = sampler.local_ids_for(
            epoch=epoch,
            cursor_step=cursor,
            rank=0,
            world_size=1,
        )
        c.update(ids)
    return c


def main() -> None:
    parser = argparse.ArgumentParser(description="ECRL correctness checker")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--target-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    global_batch = int(cfg["training"]["global_batch"])
    dataset_size = int(cfg.get("dataset", {}).get("size", 50_000))
    seed = int(cfg.get("seed", 1337))

    logs_dir = Path(args.results_dir) / "logs" / args.run_id
    records = _load_run_records(logs_dir)

    by_step: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for (_rank, global_step), rec in records.items():
        by_step[int(global_step)].append(rec)

    sampler_ref = ResumableGlobalBatchSampler(
        dataset_size=dataset_size,
        global_batch=global_batch,
        world_size=1,
        rank=0,
        seed=seed,
        epoch=0,
        cursor_step=0,
    )
    steps_per_epoch = sampler_ref.steps_per_epoch

    failures: List[str] = []

    unique_steps = sorted(by_step.keys())
    if unique_steps:
        expected_steps = set(range(1, max(unique_steps) + 1))
        missing_steps = sorted(expected_steps - set(unique_steps))
        if missing_steps:
            failures.append(f"missing global steps: {missing_steps[:10]} (total={len(missing_steps)})")

    if args.target_steps is not None and unique_steps and unique_steps[-1] != args.target_steps:
        failures.append(
            f"final global_step mismatch: observed={unique_steps[-1]} expected={args.target_steps}"
        )

    epoch_observed: Dict[int, Counter[int]] = defaultdict(Counter)
    epoch_step_count: Dict[int, int] = defaultdict(int)

    for step in unique_steps:
        step_records = by_step[step]
        world_sizes = {int(r["world_size"]) for r in step_records}
        if len(world_sizes) != 1:
            failures.append(f"step {step}: inconsistent world_size across ranks: {world_sizes}")
            continue
        world_size = next(iter(world_sizes))

        ranks = sorted(int(r["rank"]) for r in step_records)
        expected_ranks = list(range(world_size))
        if ranks != expected_ranks:
            failures.append(f"step {step}: ranks {ranks} != expected {expected_ranks}")
            continue

        expected_epoch = (step - 1) // steps_per_epoch
        expected_cursor = (step - 1) % steps_per_epoch
        epoch_step_count[expected_epoch] += 1

        for rec in step_records:
            rank = int(rec["rank"])
            epoch = int(rec["epoch"])
            cursor = int(rec["cursor_step"])

            if epoch != expected_epoch or cursor != expected_cursor:
                failures.append(
                    f"step {step} rank {rank}: epoch/cursor mismatch "
                    f"observed=({epoch},{cursor}) expected=({expected_epoch},{expected_cursor})"
                )
                continue

            if global_batch % world_size != 0:
                failures.append(
                    f"step {step}: GLOBAL_BATCH ({global_batch}) not divisible by world_size ({world_size})"
                )
                continue

            expected_ids = sampler_ref.local_ids_for(
                epoch=epoch,
                cursor_step=cursor,
                rank=rank,
                world_size=world_size,
            )
            expected_hash = _hash_ids(expected_ids)
            expected_count = len(expected_ids)

            obs_hash = rec["sample_ids_hash"]
            obs_count = int(rec["sample_ids_count"])
            if obs_hash != expected_hash:
                failures.append(
                    f"step {step} rank {rank}: sample_ids_hash mismatch "
                    f"obs={obs_hash[:12]} exp={expected_hash[:12]}"
                )
                continue
            if obs_count != expected_count:
                failures.append(
                    f"step {step} rank {rank}: sample_ids_count mismatch obs={obs_count} exp={expected_count}"
                )
                continue

            epoch_observed[epoch].update(expected_ids)

    epoch_report: Dict[str, Dict[str, int]] = {}
    for epoch, obs_counter in sorted(epoch_observed.items()):
        expected_counter = _perm_counter(
            sampler_ref,
            epoch=epoch,
            steps_in_epoch=epoch_step_count.get(epoch, 0),
        )

        duplicates = sum(max(0, v - expected_counter.get(k, 0)) for k, v in obs_counter.items())
        missing = sum(max(0, expected_counter.get(k, 0) - v) for k, v in obs_counter.items())
        # Include keys only in expected.
        missing += sum(v for k, v in expected_counter.items() if k not in obs_counter)
        extra = duplicates

        epoch_report[str(epoch)] = {
            "steps": int(epoch_step_count.get(epoch, 0)),
            "duplicates": int(duplicates),
            "missing": int(missing),
            "extra": int(extra),
        }

        if duplicates != 0 or missing != 0 or extra != 0:
            failures.append(
                f"epoch {epoch}: duplicates={duplicates} missing={missing} extra={extra}"
            )

    out = {
        "run_id": args.run_id,
        "passed": len(failures) == 0,
        "num_failures": len(failures),
        "failures": failures,
        "epochs": epoch_report,
        "steps_per_epoch": int(steps_per_epoch),
        "global_batch": int(global_batch),
    }

    out_dir = Path(args.results_dir) / "metrics" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "correctness.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
