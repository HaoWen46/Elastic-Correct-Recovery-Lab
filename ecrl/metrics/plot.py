from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

VARIANT_ORDER = ["reference", "failure_blocking", "failure_overlapped", "other"]
VARIANT_DISPLAY = {
    "reference": "Reference",
    "failure_blocking": "Failure+Blocking",
    "failure_overlapped": "Failure+Overlapped",
    "other": "Other",
}
VARIANT_COLOR = {
    "reference": "#4C78A8",
    "failure_blocking": "#F58518",
    "failure_overlapped": "#54A24B",
    "other": "#B279A2",
}
PAPER_VARIANT_COLOR = {
    # Grayscale-safe palette for print / B&W export.
    "reference": "#111111",
    "failure_blocking": "#5C5C5C",
    "failure_overlapped": "#A2A2A2",
    "other": "#7A7A7A",
}
PAPER_VARIANT_HATCH = {
    "reference": "",
    "failure_blocking": "///",
    "failure_overlapped": "xxx",
    "other": "...",
}
LINESTYLES = ["-", "--", "-.", ":"]


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


def _set_theme(*, paper_mode: bool) -> None:
    if paper_mode:
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.edgecolor": "#6B7280",
                "axes.labelcolor": "#111111",
                "axes.titlecolor": "#111111",
                "xtick.color": "#222222",
                "ytick.color": "#222222",
                "grid.color": "#DDDDDD",
                "grid.linestyle": "-",
                "grid.linewidth": 0.7,
                "font.family": "DejaVu Serif",
                "font.size": 11.0,
                "axes.titlesize": 12.5,
                "axes.labelsize": 11.5,
                "legend.frameon": False,
                "savefig.facecolor": "white",
                "savefig.edgecolor": "white",
            }
        )
        return
    plt.rcParams.update(
        {
            "figure.facecolor": "#FBFCFF",
            "axes.facecolor": "#FBFCFF",
            "axes.edgecolor": "#CED4DA",
            "axes.labelcolor": "#212529",
            "axes.titlecolor": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "grid.color": "#E5E7EB",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": False,
        }
    )


def _variant_of(run_id: str) -> str:
    if "failure_overlapped" in run_id:
        return "failure_overlapped"
    if "failure_blocking" in run_id:
        return "failure_blocking"
    if "reference" in run_id:
        return "reference"
    return "other"


def _seed_of(run_id: str) -> str:
    m = re.search(r"_s(\d+)_", run_id)
    return m.group(1) if m else "?"


def _label_of(run_id: str, *, include_seed: bool = True) -> str:
    variant = _variant_of(run_id)
    seed = _seed_of(run_id)
    if include_seed:
        return f"{VARIANT_DISPLAY[variant]} (s{seed})"
    return VARIANT_DISPLAY[variant]


def _ema(values: List[float], alpha: float = 0.12) -> List[float]:
    if not values:
        return []
    out: List[float] = [float(values[0])]
    for v in values[1:]:
        out.append(alpha * float(v) + (1.0 - alpha) * out[-1])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="ECRL plotting")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--run-ids", type=str, required=True, help="comma-separated")
    parser.add_argument("--output-prefix", type=str, default="")
    parser.add_argument("--paper-mode", action="store_true", help="Use paper-friendly style (grayscale-safe).")
    parser.add_argument("--save-pdf", action="store_true", help="Save PDF copies in addition to PNG.")
    parser.add_argument("--loss-ema-alpha", type=float, default=0.12)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    run_ids = [x.strip() for x in args.run_ids.split(",") if x.strip()]
    suffix = f"_{args.output_prefix}" if args.output_prefix else ""

    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    _set_theme(paper_mode=args.paper_mode)
    variant_color = PAPER_VARIANT_COLOR if args.paper_mode else VARIANT_COLOR

    # Stable ordering: by variant, then seed, then run id
    run_ids = sorted(run_ids, key=lambda r: (VARIANT_ORDER.index(_variant_of(r)), _seed_of(r), r))
    seed_order = sorted({_seed_of(r) for r in run_ids})
    seed_to_linestyle = {s: LINESTYLES[i % len(LINESTYLES)] for i, s in enumerate(seed_order)}
    include_seed = len(seed_order) > 1

    loss_figsize = (6.8, 3.9) if args.paper_mode else (11.5, 6.1)
    loss_dpi = 220 if args.paper_mode else 150
    fig, ax = plt.subplots(figsize=loss_figsize, dpi=loss_dpi)
    for run_id in run_ids:
        steps, losses = _load_loss_curve(results_dir, run_id)
        if steps:
            variant = _variant_of(run_id)
            color = variant_color[variant]
            linestyle = seed_to_linestyle[_seed_of(run_id)]
            smooth = _ema(losses, alpha=float(args.loss_ema_alpha))
            # Keep raw trajectory faint for transparency, highlight EMA for readability.
            ax.plot(steps, losses, color=color, linewidth=1.0, alpha=0.18)
            ax.plot(
                steps,
                smooth,
                color=color,
                linestyle=linestyle,
                linewidth=2.2 if not args.paper_mode else 2.0,
                label=_label_of(run_id, include_seed=include_seed),
            )
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Trajectories")
    ax.grid(True, axis="y")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ncol = 1 if not args.paper_mode else min(3, len(handles))
        loc = "upper right" if not args.paper_mode else "upper center"
        ax.legend(loc=loc, ncol=ncol)
    fig.tight_layout()
    loss_png = plot_dir / f"loss_curves{suffix}.png"
    fig.savefig(loss_png, dpi=170 if not args.paper_mode else 260)
    if args.save_pdf:
        fig.savefig(plot_dir / f"loss_curves{suffix}.pdf")
    plt.close(fig)

    goods: List[float] = []
    names: List[str] = []
    bar_colors: List[str] = []
    bar_variants: List[str] = []
    for run_id in run_ids:
        g = _load_goodput(results_dir, run_id)
        if g is not None:
            variant = _variant_of(run_id)
            names.append(_label_of(run_id, include_seed=include_seed))
            goods.append(g)
            bar_colors.append(variant_color[variant])
            bar_variants.append(variant)

    if names:
        ordered = sorted(zip(goods, names, bar_colors, bar_variants), key=lambda x: x[0], reverse=True)
        goods = [x[0] for x in ordered]
        names = [x[1] for x in ordered]
        bar_colors = [x[2] for x in ordered]
        bar_variants = [x[3] for x in ordered]

        goodput_figsize = (6.8, 3.8) if args.paper_mode else (11.5, 6.1)
        fig, ax = plt.subplots(figsize=goodput_figsize, dpi=loss_dpi)
        bars = ax.barh(names, goods, color=bar_colors, edgecolor="#1F2937", linewidth=0.4)
        if args.paper_mode:
            for bar, variant in zip(bars, bar_variants):
                bar.set_hatch(PAPER_VARIANT_HATCH[variant])
        ax.invert_yaxis()
        ax.set_xlabel("Goodput (steps/sec)")
        ax.set_title("Goodput Comparison (Higher is Better)")
        ax.grid(True, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        max_g = max(goods) if goods else 1.0
        ax.set_xlim(0.0, max_g * 1.18)
        for b, g in zip(bars, goods):
            ax.text(
                b.get_width() + max_g * 0.012,
                b.get_y() + b.get_height() / 2.0,
                f"{g:.2f}",
                va="center",
                ha="left",
                fontsize=9.5 if not args.paper_mode else 9.0,
                color="#111827",
            )
        fig.tight_layout()
        goodput_png = plot_dir / f"goodput{suffix}.png"
        fig.savefig(goodput_png, dpi=170 if not args.paper_mode else 260)
        if args.save_pdf:
            fig.savefig(plot_dir / f"goodput{suffix}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
