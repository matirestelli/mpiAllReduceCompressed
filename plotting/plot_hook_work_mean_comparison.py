#!/usr/bin/env python3
"""
Compare mean hook work time for one model.

This plots:
    mean(epoch hook_work_mean_ms)

So if a log has 20 epochs, this script computes the mean of the 20
per-epoch hook work means.

Built-in/no-hook logs are skipped because they have no HOOK_WORK_SUMMARY.

Run from:
    ddp-allreduce-eval-framework/plotting/

Example:
    python plot_hook_work_mean_comparison.py \
        --results-root experiments_results_lr001_GlobalBS128 \
        --model-dir wideresnet \
        --model-title Wide_ResNet50_2 \
        --gpus 4 \
        --out wide_resnet50_2_4gpu_hook_work_mean.pdf

Across all available GPU counts:
    python plot_hook_work_mean_comparison.py \
        --results-root experiments_results_lr001_GlobalBS128 \
        --model-dir wideresnet \
        --model-title Wide_ResNet50_2 \
        --out wide_resnet50_2_all_gpus_hook_work_mean.pdf
"""

import argparse
import matplotlib.pyplot as plt

from ddp_plot_common import (
    HOOK_COLORS,
    HOOK_X,
    add_grouped_hook_legend,
    apply_paper_style,
    find_logs,
    hook_sort_key,
    parse_log,
    pretty_hook_from_filename,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="experiments_results_lr001_GlobalBS128")
    p.add_argument("--model-dir", default="wideresnet")
    p.add_argument("--model-title", default="Wide_ResNet50_2")
    p.add_argument("--gpus", type=int, default=None)
    p.add_argument("--run", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    apply_paper_style(plt)

    logs = find_logs(args.results_root, args.model_dir, gpus=args.gpus, run=args.run)
    if not logs:
        raise SystemExit("No logs found.")

    names, values = [], []

    for log in logs:
        stats = parse_log(log)
        value = stats["hook_work_mean_ms"]
        if value is None:
            continue
        names.append(pretty_hook_from_filename(log))
        values.append(value)

    if not values:
        raise SystemExit(
            "No hook work timing found. Built-in logs have no hook work timing."
        )

    rows = sorted(zip(names, values), key=lambda row: hook_sort_key(row[0]))
    names = [row[0] for row in rows]
    values = [row[1] for row in rows]

    can_group = all(name in HOOK_X for name in names)
    x_positions = [HOOK_X[name] for name in names] if can_group else range(len(names))
    colors = [HOOK_COLORS.get(name, "#999999") for name in names]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    bars = ax.bar(
        x_positions,
        values,
        width=0.38 if can_group else 0.48,
        color=colors,
        edgecolor="black",
        linewidth=0.35,
    )

    gpu_text = f" - {args.gpus} GPUs" if args.gpus else ""
    ax.set_title(
        f"{args.model_title}{gpu_text}",
        fontsize=21,
        fontweight="normal",
        pad=7,
    )
    ax.set_xlabel("Communication Hook", fontsize=19)
    ax.set_ylabel("Mean Hook Work Time (ms)", fontsize=20)
    ax.grid(axis="y", alpha=0.25, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=15)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    if can_group:
        add_grouped_hook_legend(ax, fontsize=10.5, y=-0.08)
        ax.set_xlim(-0.45, 7.15)
    else:
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(names, rotation=28, ha="right", fontsize=10)

    ax.set_ylim(0, max(values) * 1.18)

    gpu_name = f"_{args.gpus}gpu" if args.gpus else ""
    out = args.out or f"{args.model_dir}{gpu_name}_hook_work_mean.pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
