from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _digest_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    import hashlib

    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        h.update(key.encode("utf-8"))
        t = state_dict[key].detach().cpu().contiguous()
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _l2_distance(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    keys = sorted(set(a.keys()) & set(b.keys()))
    if not keys:
        return float("nan")
    total = 0.0
    for k in keys:
        da = a[k].detach().cpu().float().reshape(-1)
        db = b[k].detach().cpu().float().reshape(-1)
        if da.numel() != db.numel():
            raise ValueError(f"shape mismatch for key={k}: {tuple(da.shape)} vs {tuple(db.shape)}")
        total += float(torch.sum((da - db) ** 2).item())
    return float(np.sqrt(total))


def _load_rank0_losses(path: Path) -> Dict[int, float]:
    if not path.exists():
        return {}
    by_step: Dict[int, float] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            by_step[int(rec["global_step"])] = float(rec["loss"])
    return by_step


def main() -> None:
    parser = argparse.ArgumentParser(description="ECRL divergence metric")
    parser.add_argument("--reference-run", type=str, required=True)
    parser.add_argument("--candidate-run", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--steps", type=str, default="200,400,800")
    args = parser.parse_args()

    steps = [int(x.strip()) for x in args.steps.split(",") if x.strip()]

    ckpt_root = Path(args.results_dir) / "checkpoints"
    out_steps: Dict[str, Any] = {}
    for step in steps:
        ref_path = ckpt_root / args.reference_run / f"step_{step:08d}.pt"
        cand_path = ckpt_root / args.candidate_run / f"step_{step:08d}.pt"

        if not ref_path.exists() or not cand_path.exists():
            out_steps[str(step)] = {
                "available": False,
                "reference_path": str(ref_path),
                "candidate_path": str(cand_path),
            }
            continue

        ref_payload = _load_checkpoint(ref_path)
        cand_payload = _load_checkpoint(cand_path)
        ref_model = ref_payload["model"]
        cand_model = cand_payload["model"]

        out_steps[str(step)] = {
            "available": True,
            "l2_distance": _l2_distance(ref_model, cand_model),
            "reference_digest": _digest_state_dict(ref_model),
            "candidate_digest": _digest_state_dict(cand_model),
        }

    ref_losses = _load_rank0_losses(Path(args.results_dir) / "logs" / args.reference_run / "rank0.jsonl")
    cand_losses = _load_rank0_losses(Path(args.results_dir) / "logs" / args.candidate_run / "rank0.jsonl")

    common_steps = sorted(set(ref_losses) & set(cand_losses))
    if common_steps:
        diffs = np.asarray([abs(ref_losses[s] - cand_losses[s]) for s in common_steps], dtype=np.float64)
        x = np.asarray(common_steps, dtype=np.float64)
        if hasattr(np, "trapezoid"):
            auc = float(np.trapezoid(diffs, x=x))
        else:
            auc = float(np.trapz(diffs, x=x))
        loss_stats = {
            "num_common_steps": len(common_steps),
            "max_abs_diff": float(np.max(diffs)),
            "mean_abs_diff": float(np.mean(diffs)),
            "auc_abs_diff": auc,
        }
    else:
        loss_stats = {
            "num_common_steps": 0,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "auc_abs_diff": None,
        }

    out = {
        "reference_run": args.reference_run,
        "candidate_run": args.candidate_run,
        "steps": out_steps,
        "loss_divergence": loss_stats,
    }

    out_dir = Path(args.results_dir) / "metrics" / args.candidate_run
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"divergence_vs_{args.reference_run}.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
