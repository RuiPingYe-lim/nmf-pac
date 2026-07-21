#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot the 2x2 NMF sensitivity figure used in the paper draft.

The MRNet -> KneeMRI K/T points are the target-test AUC values collected under
pipe_out/ by scripts/collect_and_plot_sensitivity.py. The remaining points are
kept here explicitly so the publication figure can be regenerated without
rerunning old sweeps.
"""

from pathlib import Path

import matplotlib.pyplot as plt


OUT_DIR = Path("img")

BLUE = "#1f77b4"
ORANGE = "#d55e00"


DATA = {
    "K": {
        "x": [1, 2, 3, 4],
        "mrnet_to_knee": [0.8261, 0.7996, 0.7582, 0.7831],
        "knee_to_mrnet": [0.8768, 0.8076, 0.7946, 0.8145],
    },
    "loss": {
        "x": ["Frobenius", "KL"],
        "mrnet_to_knee": [0.8261, 0.8080],
        "knee_to_mrnet": [0.8768, 0.8165],
    },
    "proto_m": {
        "x": [0.90, 0.95, 0.97, 0.99],
        "mrnet_to_knee": [0.7690, 0.7920, 0.8261, 0.6720],
        "knee_to_mrnet": [0.8420, 0.8650, 0.8768, 0.8430],
    },
    "assign_iters": {
        "x": [20, 40, 60, 100, 150],
        "mrnet_to_knee": [0.7741, 0.7316, 0.7826, 0.8261, 0.8261],
        "knee_to_mrnet": [0.8550, 0.8768, 0.8768, 0.8768, 0.8768],
    },
}


def style_axis(ax):
    ax.set_ylim(0.65, 0.90)
    ax.set_yticks([0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_ylabel("Target-domain test AUC")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def plot_lines(ax, x, y1, y2):
    ax.plot(
        x,
        y1,
        color=BLUE,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.6,
        linewidth=1.8,
        label="MRNet \u2192 KneeMRI",
    )
    ax.plot(
        x,
        y2,
        color=ORANGE,
        marker="s",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.6,
        linewidth=1.8,
        linestyle="--",
        label="KneeMRI \u2192 MRNet",
    )


def plot_loss(ax):
    labels = DATA["loss"]["x"]
    x = range(len(labels))
    ax.scatter(
        x,
        DATA["loss"]["mrnet_to_knee"],
        color=BLUE,
        marker="o",
        s=55,
        facecolors="white",
        linewidths=1.8,
    )
    ax.scatter(
        x,
        DATA["loss"]["knee_to_mrnet"],
        color=ORANGE,
        marker="s",
        s=55,
        facecolors="white",
        linewidths=1.8,
    )
    ax.set_xticks(list(x), labels)


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))

    ax = axes[0, 0]
    plot_lines(ax, DATA["K"]["x"], DATA["K"]["mrnet_to_knee"], DATA["K"]["knee_to_mrnet"])
    style_axis(ax)
    ax.set_title(r"(a) Number of prototypes per class ($K$)")
    ax.set_xlabel(r"$K$")
    ax.set_xticks(DATA["K"]["x"])

    ax = axes[0, 1]
    plot_loss(ax)
    style_axis(ax)
    ax.set_title("(b) NMF reconstruction loss type")
    ax.set_xlabel("Loss type")

    ax = axes[1, 0]
    plot_lines(
        ax,
        DATA["proto_m"]["x"],
        DATA["proto_m"]["mrnet_to_knee"],
        DATA["proto_m"]["knee_to_mrnet"],
    )
    style_axis(ax)
    ax.set_title(r"(c) Prototype update momentum ($m$)")
    ax.set_xlabel(r"Prototype update momentum ($m$)")
    ax.set_xticks(DATA["proto_m"]["x"], ["0.90", "0.95", "0.97", "0.99"])

    ax = axes[1, 1]
    plot_lines(
        ax,
        DATA["assign_iters"]["x"],
        DATA["assign_iters"]["mrnet_to_knee"],
        DATA["assign_iters"]["knee_to_mrnet"],
    )
    style_axis(ax)
    ax.set_title(r"(d) NMF assignment iterations ($T$)")
    ax.set_xlabel(r"NMF assignment iterations ($T$)")
    ax.set_xticks(DATA["assign_iters"]["x"])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.08, top=0.88, wspace=0.24, hspace=0.42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "nmf_sensitivity_2x2.png"
    pdf = OUT_DIR / "nmf_sensitivity_2x2.pdf"
    fig.savefig(png, dpi=600)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"saved {png}")
    print(f"saved {pdf}")


if __name__ == "__main__":
    main()
