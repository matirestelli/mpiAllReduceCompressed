#!/usr/bin/env python3
"""
Plot scaling for one model: compare hooks across all discovered GPU counts.

Run from:
    ddp-allreduce-eval-framework/plotting/

Example:
    python plot_hook_scaling.py \
        --results-root experiments_results_lr001_GlobalBS128 \
        --model-dir wideresnet \
        --model-title Wide_ResNet50_2 \
        --metric epoch_mean \
        --out wide_resnet50_2_scaling_epoch_mean.pdf

Communication timing scaling:
    python plot_hook_scaling.py \
        --results-root experiments_results_lr001_GlobalBS128 \
        --model-dir wideresnet \
        --model-title Wide_ResNet50_2 \
        --metric hook_tail_total_ms \
        --out wide_resnet50_2_scaling_tail_total.pdf

For LocalBS128 results:
    python plot_hook_scaling.py \
        --results-root experiments_results_lr001_LocalBS128 \
        --model-dir wideresnet \
        --model-title Wide_ResNet50_2
        
For resnext_101_32x8d
    python plot_hook_scaling.py \
    --results-root experiments_results_lr001_GlobalBS128 \
    --model-dir resnext101_32x8d \
    --model-title ResNeXt101_32x8d \
    --metric epoch_mean \
    --out resnext101_32x8d_scaling_epoch_mean.pdf
    
    For resnext_101_32x8d
    python plot_hook_scaling.py \
    --results-root experiments_results_lr001_LocalBS128 \
    --model-dir resnext101_32x8d \
    --model-title ResNeXt101_32x8d \
    --metric epoch_mean \
    --out resnext101_32x8d_scaling_epoch_mean.pdf
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from ddp_plot_common import (
    HOOK_COLORS,
    HOOK_ORDER,
    HOOK_X,
    add_grouped_hook_legend,
    apply_paper_style,
    find_logs,
    hook_sort_key,
    parse_log,
    metric_value,
    pretty_hook_from_filename,
    parse_gpu_count,
)

METRICS = [
    "epoch_mean",
    "epoch_min",
    "epoch_max",
    "total_time",
    "best_val_acc",
    "hook_work_mean_ms",
    "hook_tail_mean_ms",
    "hook_work_total_ms",
    "hook_tail_total_ms",
]

YLABELS = {
    "epoch_mean": "Training time / epoch (s)",
    "epoch_min": "Min epoch time (s)",
    "epoch_max": "Max epoch time (s)",
    "total_time": "Total training time (s)",
    "best_val_acc": "Best validation accuracy (%)",
    "hook_work_mean_ms": "Mean hook work time (ms)",
    "hook_tail_mean_ms": "Mean exposed tail time (ms)",
    "hook_work_total_ms": "Mean accumulated hook work / epoch (ms)",
    "hook_tail_total_ms": "Mean accumulated exposed tail / epoch (ms)",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="experiments_results_lr001_GlobalBS128")
    p.add_argument("--model-dir", default="wideresnet")
    p.add_argument("--model-title", default="Wide_ResNet50_2")
    p.add_argument("--run", default=None)
    p.add_argument("--metric", choices=METRICS, default="epoch_mean")
    p.add_argument("--skip-first-epoch", type=int, default=1)
    p.add_argument(
        "--allow-missing-hooks",
        action="store_true",
        help="Plot hooks even when they are not present at every discovered GPU count.",
    )
    p.add_argument("--out", default=None)
    args = p.parse_args()

    apply_paper_style(plt)

    logs = find_logs(args.results_root, args.model_dir, run=args.run)
    if not logs:
        raise SystemExit("No logs found.")

    data = {}
    hooks = []
    gpus_seen = set()

    for log in logs:
        gpus = parse_gpu_count(log)
        if gpus is None:
            continue

        hook = pretty_hook_from_filename(log)
        if hook not in HOOK_ORDER:
            print(f"Skipping unsupported hook label for scaling plot: {hook} ({log})")
            continue

        stats = parse_log(log)
        value = metric_value(stats, args.metric, args.skip_first_epoch)
        if value is None:
            continue

        if (gpus, hook) in data:
            print(f"Skipping duplicate scaling value for {gpus} GPUs / {hook}: {log}")
            continue

        data[(gpus, hook)] = value
        gpus_seen.add(gpus)
        if hook not in hooks:
            hooks.append(hook)

    hooks = sorted(hooks, key=hook_sort_key)
    gpus_list = sorted(gpus_seen)
    if not args.allow_missing_hooks:
        complete_hooks = []
        for hook in hooks:
            missing_gpus = [g for g in gpus_list if (g, hook) not in data]
            if missing_gpus:
                print(
                    "Skipping incomplete hook for scaling plot: "
                    f"{hook} missing at GPUs {missing_gpus}"
                )
                continue
            complete_hooks.append(hook)
        hooks = complete_hooks

    if not hooks:
        raise SystemExit("No complete hook series found for the selected results.")

    x = np.arange(len(gpus_list)) * 1.65
    width = 0.085
    present_hook_x = [HOOK_X[hook] for hook in hooks if hook in HOOK_X]
    hook_center = (
        (min(present_hook_x) + max(present_hook_x)) / 2 if present_hook_x else 0.0
    )
    family_scale = 0.15

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    for i, hook in enumerate(hooks):
        if hook in HOOK_X:
            offset = (HOOK_X[hook] - hook_center) * family_scale
        else:
            offset = (i - (len(hooks) - 1) / 2) * width
        values = [data.get((g, hook), np.nan) for g in gpus_list]
        ax.bar(
            x + offset,
            values,
            width,
            label=hook,
            color=HOOK_COLORS.get(hook, "#999999"),
            edgecolor="black",
            linewidth=0.35,
        )

    ax.set_title(args.model_title, fontsize=21, fontweight="normal", pad=7)
    ax.set_xlabel("GPUs", fontsize=19)
    ax.set_ylabel(YLABELS[args.metric], fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in gpus_list], fontsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(axis="y", alpha=0.25, linewidth=0.55)
    ax.set_axisbelow(True)
    add_grouped_hook_legend(ax, fontsize=10.5, y=-0.10)

    out = args.out or f"{args.model_dir}_scaling_{args.metric}.pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
