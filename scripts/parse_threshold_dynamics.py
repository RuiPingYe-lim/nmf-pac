#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Parse training logs and visualize threshold dynamics."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

PAPER_STYLE = {
    "figure_size": (8, 4.8),
    "base_fontsize": 10.5,
    "axes_labelsize": 10.5,
    "axes_titlesize": 11,
    "legend_fontsize": 9,
    "tick_labelsize": 9,
    "title_linespacing": 1.15,
    "title_pad": 8,
    "tight_pad": 1.0,
}


ROUND_HEADER_RE = re.compile(r"^=+\s*Round\s+(\d+)/(\d+).*$")
THR_START_RE = re.compile(
    r"^\[Thr\]\[start\]\s+tau_global=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\|\s*"
    r"tau_cls=\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]"
)
THR_END_RE = re.compile(
    r"^\[Thr\]\[end\]\s+tau_global=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\|\s*"
    r"tau_cls=\[\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]"
)
SELECTED_RE = re.compile(r"^\[A-stage\]\s+selected target pseudo samples:\s*(\d+)")
AUC_RE = re.compile(
    r"^\[Round\s*(\d+)\]\s+tgt_test:\s+AUC=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?).*$"
)


@dataclass
class RoundRecord:
    round: int
    tau_global_start: Optional[float] = None
    tau_cls0_start: Optional[float] = None
    tau_cls1_start: Optional[float] = None
    tau_global_end: Optional[float] = None
    tau_cls0_end: Optional[float] = None
    tau_cls1_end: Optional[float] = None
    selected_target_pseudo_samples: Optional[int] = None
    tgt_test_auc: Optional[float] = None


def _get_or_create(records: Dict[int, RoundRecord], rnd: int) -> RoundRecord:
    if rnd not in records:
        records[rnd] = RoundRecord(round=rnd)
    return records[rnd]


def parse_log(log_path: Path) -> List[RoundRecord]:
    if not log_path.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_path}")

    records: Dict[int, RoundRecord] = {}
    current_round: Optional[int] = None

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            m = ROUND_HEADER_RE.match(line)
            if m:
                current_round = int(m.group(1))
                _get_or_create(records, current_round)
                continue

            m = THR_START_RE.match(line)
            if m and current_round is not None:
                rec = _get_or_create(records, current_round)
                rec.tau_global_start = float(m.group(1))
                rec.tau_cls0_start = float(m.group(2))
                rec.tau_cls1_start = float(m.group(3))
                continue

            m = THR_END_RE.match(line)
            if m and current_round is not None:
                rec = _get_or_create(records, current_round)
                rec.tau_global_end = float(m.group(1))
                rec.tau_cls0_end = float(m.group(2))
                rec.tau_cls1_end = float(m.group(3))
                continue

            m = SELECTED_RE.match(line)
            if m and current_round is not None:
                rec = _get_or_create(records, current_round)
                rec.selected_target_pseudo_samples = int(m.group(1))
                continue

            m = AUC_RE.match(line)
            if m:
                auc_round = int(m.group(1))
                rec = _get_or_create(records, auc_round)
                rec.tgt_test_auc = float(m.group(2))
                continue

            # Keep parser tolerant: ignore unrecognized lines.
            _ = line_num

    parsed = sorted(records.values(), key=lambda x: x.round)
    if not parsed:
        raise ValueError("No round records were parsed from the log.")

    return parsed


def save_csv(records: List[RoundRecord], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "round",
        "tau_global_start",
        "tau_cls0_start",
        "tau_cls1_start",
        "tau_global_end",
        "tau_cls0_end",
        "tau_cls1_end",
        "selected_target_pseudo_samples",
        "tgt_test_auc",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))


def _setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": PAPER_STYLE["figure_size"],
            "font.size": PAPER_STYLE["base_fontsize"],
            "axes.labelsize": PAPER_STYLE["axes_labelsize"],
            "axes.titlesize": PAPER_STYLE["axes_titlesize"],
            "legend.fontsize": PAPER_STYLE["legend_fontsize"],
            "xtick.labelsize": PAPER_STYLE["tick_labelsize"],
            "ytick.labelsize": PAPER_STYLE["tick_labelsize"],
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.0,
            "savefig.bbox": "tight",
        }
    )


def plot_threshold_evolution(records: List[RoundRecord], out_dir: Path, tau_min: float = 0.60, warmup_rounds: int = 0) -> None:
    rounds = [r.round for r in records]
    tau_global_end = [r.tau_global_end for r in records]
    tau_cls0_end = [r.tau_cls0_end for r in records]
    tau_cls1_end = [r.tau_cls1_end for r in records]

    # Thresholds shown here are from each round after Stage A update.
    fig, ax = plt.subplots()
    if warmup_rounds > 0:
        ax.axvspan(0, warmup_rounds, color="gray", alpha=0.08, label="Warm-up")
    ax.plot(rounds, tau_global_end, marker="o", markersize=4, linestyle="-", label="Global")
    ax.plot(rounds, tau_cls0_end, marker="s", markersize=4, linestyle="--", label="Class 0")
    ax.plot(rounds, tau_cls1_end, marker="^", markersize=4, linestyle="-.", label="Class 1")
    ax.axhline(float(tau_min), color="gray", linestyle=":", linewidth=1.0, label=r"$\tau_{\min}$")

    ax.set_xlabel("Round")
    ax.set_ylabel("Threshold")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout(pad=PAPER_STYLE["tight_pad"])

    png_path = out_dir / "threshold_evolution.png"
    pdf_path = out_dir / "threshold_evolution.pdf"
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)


def _ema(values: List[Optional[float]], alpha: float = 0.3) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    state: Optional[float] = None
    for value in values:
        if value is None:
            out.append(None)
            continue
        state = float(value) if state is None else float(alpha) * float(value) + (1.0 - float(alpha)) * state
        out.append(state)
    return out


def plot_selected_vs_auc(records: List[RoundRecord], out_dir: Path, target_pool_size: Optional[int] = None) -> None:
    rounds = [r.round for r in records]
    selected = [r.selected_target_pseudo_samples for r in records]
    selected_ema = _ema(selected, alpha=0.3)
    auc = [r.tgt_test_auc for r in records]

    fig, ax1 = plt.subplots()
    color_left = "#1f77b4"  # blue
    color_right = "#ff7f0e"  # orange

    ax1.plot(
        rounds,
        selected,
        color=color_left,
        marker="o",
        markersize=3,
        linestyle="-",
        alpha=0.28,
        linewidth=1.0,
        label="Selected raw",
    )
    ln1 = ax1.plot(
        rounds,
        selected_ema,
        color=color_left,
        marker="o",
        markersize=4,
        linestyle="-",
        linewidth=2.2,
        label="Selected (EMA)",
    )
    if target_pool_size is not None and int(target_pool_size) > 0:
        ax1.axhline(int(target_pool_size), color=color_left, linestyle=":", linewidth=1.0, alpha=0.45, label="Target pool size")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Selected Target Pseudo Samples", color=color_left)
    ax1.tick_params(axis="y", labelcolor=color_left)
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    ax2 = ax1.twinx()
    ln2 = ax2.plot(
        rounds,
        auc,
        color=color_right,
        marker="s",
        markersize=4,
        linestyle="--",
        label="Target-domain AUC",
    )
    ax2.set_ylabel("Target Test AUC", color=color_right)
    ax2.tick_params(axis="y", labelcolor=color_right)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    lines = handles1 + handles2
    labels = labels1 + labels2
    ax1.legend(
        lines,
        labels,
        frameon=True,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
    )
    fig.tight_layout(pad=PAPER_STYLE["tight_pad"])

    png_path = out_dir / "selected_vs_tgt_auc.png"
    pdf_path = out_dir / "selected_vs_tgt_auc.pdf"
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse threshold dynamics from training log and generate plots."
    )
    parser.add_argument("--log_path", type=str, required=True, help="Path to training log file")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--tau_min", type=float, default=0.60, help="Reference lower threshold line shown in Fig. 3")
    parser.add_argument("--warmup_rounds", type=int, default=0, help="Optional warm-up span shown in Fig. 3")
    parser.add_argument("--target_pool_size", type=int, default=640, help="Optional denominator/reference line for selected pseudo labels")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    log_path = Path(args.log_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    try:
        records = parse_log(log_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "parsed_threshold_dynamics.csv"
        save_csv(records, csv_path)

        _setup_plot_style()
        plot_threshold_evolution(records, out_dir, tau_min=float(args.tau_min), warmup_rounds=int(args.warmup_rounds))
        plot_selected_vs_auc(records, out_dir, target_pool_size=int(args.target_pool_size) if args.target_pool_size > 0 else None)

        print(f"Parsed rounds: {len(records)}")
        print(f"CSV saved to: {csv_path}")
        print(f"Figures saved to: {out_dir}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
