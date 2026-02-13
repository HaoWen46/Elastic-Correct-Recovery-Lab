from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    if arr.size == 1:
        return {"n": 1, "mean": float(arr[0]), "std": 0.0}
    return {"n": int(arr.size), "mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1))}


def _parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _load_variant_divergence(
    *,
    results_dir: Path,
    reference_runs: List[str],
    candidate_runs: List[str],
) -> Dict[str, Any]:
    if len(reference_runs) != len(candidate_runs):
        raise ValueError("reference_runs and candidate_runs must have the same length")

    loss_mean_abs_diffs: List[float] = []
    step_to_l2: Dict[str, List[float]] = {}

    for ref_run, cand_run in zip(reference_runs, candidate_runs):
        path = results_dir / "metrics" / cand_run / f"divergence_vs_{ref_run}.json"
        if not path.exists():
            continue
        d = _load_json(path)
        loss = d.get("loss_divergence", {}).get("mean_abs_diff")
        if loss is not None:
            loss_mean_abs_diffs.append(float(loss))

        for step, rec in d.get("steps", {}).items():
            if not rec.get("available", False):
                continue
            l2 = rec.get("l2_distance")
            if l2 is None:
                continue
            step_to_l2.setdefault(str(step), []).append(float(l2))

    return {
        "loss_mean_abs_diff": _mean_std(loss_mean_abs_diffs),
        "l2_by_step": {step: _mean_std(vals) for step, vals in sorted(step_to_l2.items(), key=lambda x: int(x[0]))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publishable markdown report from ECRL aggregates")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-prefix", type=str, default="exp3_publishable")
    parser.add_argument("--reference-runs", type=str, required=True)
    parser.add_argument("--blocking-runs", type=str, required=True)
    parser.add_argument("--overlapped-runs", type=str, required=True)
    parser.add_argument("--aggregate-reference-label", type=str, default="exp3_reference")
    parser.add_argument("--aggregate-blocking-label", type=str, default="exp3_failure_blocking")
    parser.add_argument("--aggregate-overlapped-label", type=str, default="exp3_failure_overlapped")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    reference_runs = _parse_csv(args.reference_runs)
    blocking_runs = _parse_csv(args.blocking_runs)
    overlapped_runs = _parse_csv(args.overlapped_runs)
    if not (reference_runs and blocking_runs and overlapped_runs):
        raise ValueError("run lists cannot be empty")

    agg_ref = _load_json(results_dir / "metrics" / "_aggregate" / f"{args.aggregate_reference_label}.json")
    agg_blk = _load_json(results_dir / "metrics" / "_aggregate" / f"{args.aggregate_blocking_label}.json")
    agg_ovl = _load_json(results_dir / "metrics" / "_aggregate" / f"{args.aggregate_overlapped_label}.json")

    div_blk = _load_variant_divergence(
        results_dir=results_dir,
        reference_runs=reference_runs,
        candidate_runs=blocking_runs,
    )
    div_ovl = _load_variant_divergence(
        results_dir=results_dir,
        reference_runs=reference_runs,
        candidate_runs=overlapped_runs,
    )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "num_seeds": len(reference_runs),
        "variants": {
            "reference": agg_ref,
            "failure_blocking": agg_blk,
            "failure_overlapped": agg_ovl,
        },
        "divergence": {
            "failure_blocking": div_blk,
            "failure_overlapped": div_ovl,
        },
    }

    json_path = report_dir / f"{args.output_prefix}.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    def _variant_row(label: str, data: Dict[str, Any]) -> str:
        g = data["goodput_steps_per_sec"]
        r = data["restarts"]
        c = data["correctness"]
        return (
            f"| {label} | {c['pass_rate']:.2f} | "
            f"{g['mean']:.4f} +/- {g['std']:.4f} | {g['ci95']:.4f} | "
            f"{r['mean']:.2f} +/- {r['std']:.2f} |"
        )

    lines: List[str] = []
    lines.append(f"# Publishable Report: {args.output_prefix}")
    lines.append("")
    lines.append(f"Generated at: {summary['generated_at']}")
    lines.append(f"Seeds: {len(reference_runs)}")
    lines.append("")
    lines.append("## Aggregate Performance")
    lines.append("")
    lines.append("| Variant | Correctness Pass Rate | Goodput Mean +/- Std | Goodput CI95 | Restarts Mean +/- Std |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(_variant_row("Reference", agg_ref))
    lines.append(_variant_row("Failure + Blocking", agg_blk))
    lines.append(_variant_row("Failure + Overlapped", agg_ovl))
    lines.append("")
    lines.append("## Divergence (Across Seeds)")
    lines.append("")
    lines.append("| Variant | Mean Abs Loss Diff (mean +/- std) |")
    lines.append("|---|---:|")
    lines.append(
        f"| Failure + Blocking | "
        f"{div_blk['loss_mean_abs_diff']['mean']:.6f} +/- {div_blk['loss_mean_abs_diff']['std']:.6f} |"
    )
    lines.append(
        f"| Failure + Overlapped | "
        f"{div_ovl['loss_mean_abs_diff']['mean']:.6f} +/- {div_ovl['loss_mean_abs_diff']['std']:.6f} |"
    )
    lines.append("")

    all_steps = sorted(set(div_blk["l2_by_step"].keys()) | set(div_ovl["l2_by_step"].keys()), key=lambda x: int(x))
    if all_steps:
        lines.append("### Parameter L2 by Step")
        lines.append("")
        lines.append("| Step | Blocking L2 (mean +/- std) | Overlapped L2 (mean +/- std) |")
        lines.append("|---:|---:|---:|")
        for step in all_steps:
            b = div_blk["l2_by_step"].get(step, {"mean": float("nan"), "std": float("nan")})
            o = div_ovl["l2_by_step"].get(step, {"mean": float("nan"), "std": float("nan")})
            lines.append(
                f"| {step} | {b['mean']:.6f} +/- {b['std']:.6f} | {o['mean']:.6f} +/- {o['std']:.6f} |"
            )
        lines.append("")

    md_path = report_dir / f"{args.output_prefix}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"markdown": str(md_path), "json": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
