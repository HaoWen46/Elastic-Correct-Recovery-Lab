from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


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


def _load_loss_curve(results_dir: Path, run_id: str) -> tuple[list[int], list[float]]:
    recs = _load_jsonl(results_dir / "logs" / run_id / "rank0.jsonl")
    by_step: Dict[int, float] = {}
    for rec in recs:
        by_step[int(rec["global_step"])] = float(rec["loss"])
    steps = sorted(by_step)
    losses = [by_step[s] for s in steps]
    return steps, losses


def _load_goodput(results_dir: Path, run_id: str) -> float | None:
    p = results_dir / "metrics" / run_id / "goodput.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return float(data["goodput_steps_per_sec"])


def main() -> None:
    parser = argparse.ArgumentParser(description="ECRL plotting")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--run-ids", type=str, required=True, help="comma-separated")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    run_ids = [x.strip() for x in args.run_ids.split(",") if x.strip()]

    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    for run_id in run_ids:
        steps, losses = _load_loss_curve(results_dir, run_id)
        if steps:
            plt.plot(steps, losses, label=run_id)
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.title("Loss Curves (Rank 0, committed trajectory)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "loss_curves.png", dpi=150)
    plt.close()

    goods: List[float] = []
    names: List[str] = []
    for run_id in run_ids:
        g = _load_goodput(results_dir, run_id)
        if g is not None:
            names.append(run_id)
            goods.append(g)

    if names:
        plt.figure(figsize=(10, 5))
        plt.bar(names, goods)
        plt.ylabel("Goodput (steps/sec)")
        plt.title("Goodput Comparison")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(plot_dir / "goodput.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
