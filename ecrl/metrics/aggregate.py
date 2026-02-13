from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std_ci95(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    mean = float(np.mean(arr)) if n else float("nan")
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    ci95 = 1.96 * sem
    return {"n": n, "mean": mean, "std": std, "ci95": ci95}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ECRL metrics across run IDs")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--run-ids", type=str, required=True, help="comma-separated run IDs")
    parser.add_argument("--label", type=str, default="aggregate")
    parser.add_argument("--output", type=str, default=None, help="optional JSON output path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    run_ids = [x.strip() for x in args.run_ids.split(",") if x.strip()]
    if not run_ids:
        raise ValueError("run-ids cannot be empty")

    records: List[Dict[str, Any]] = []
    for run_id in run_ids:
        base = results_dir / "metrics" / run_id
        cpath = base / "correctness.json"
        gpath = base / "goodput.json"
        if not cpath.exists() or not gpath.exists():
            raise FileNotFoundError(f"missing metrics for run_id={run_id}: {base}")
        c = _load_json(cpath)
        g = _load_json(gpath)
        records.append(
            {
                "run_id": run_id,
                "correctness_passed": bool(c["passed"]),
                "goodput_steps_per_sec": float(g["goodput_steps_per_sec"]),
                "wall_clock_time_sec": float(g["wall_clock_time_sec"]),
                "restarts": int(g["restarts"]),
                "replayed_steps": int(g["replayed_steps"]),
                "stall_time_sec_total": float(g["checkpoint"]["stall_time_sec_total"]),
                "snapshot_time_sec_total": float(g["checkpoint"]["snapshot_time_sec_total"]),
                "write_time_sec_total": float(g["checkpoint"]["write_time_sec_total"]),
            }
        )

    all_pass = all(r["correctness_passed"] for r in records)
    pass_rate = float(np.mean([1.0 if r["correctness_passed"] else 0.0 for r in records]))

    out = {
        "label": args.label,
        "run_ids": run_ids,
        "num_runs": len(run_ids),
        "correctness": {
            "all_passed": all_pass,
            "pass_rate": pass_rate,
        },
        "goodput_steps_per_sec": _mean_std_ci95([r["goodput_steps_per_sec"] for r in records]),
        "wall_clock_time_sec": _mean_std_ci95([r["wall_clock_time_sec"] for r in records]),
        "restarts": _mean_std_ci95([float(r["restarts"]) for r in records]),
        "replayed_steps": _mean_std_ci95([float(r["replayed_steps"]) for r in records]),
        "stall_time_sec_total": _mean_std_ci95([r["stall_time_sec_total"] for r in records]),
        "snapshot_time_sec_total": _mean_std_ci95([r["snapshot_time_sec_total"] for r in records]),
        "write_time_sec_total": _mean_std_ci95([r["write_time_sec_total"] for r in records]),
        "runs": records,
    }

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = results_dir / "metrics" / "_aggregate"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
