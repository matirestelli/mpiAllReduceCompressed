#!/usr/bin/env python3
"""
Scaling plot: one model, many hooks, many GPU counts.

Example:
python plot_hook_scaling.py \
  --model Wide_ResNet50_2 \
  --metric epoch_mean \
  --out wide_resnet50_2_hook_scaling.png \
  "4:Baseline=logs/builtin_4gpu.out" \
  "8:Baseline=logs/builtin_8gpu.out" \
  "16:Baseline=logs/builtin_16gpu.out" \
  "4:Ring+ZFP(rate:10)=logs/ring_zfp_rate10_4gpu.out" \
  "8:Ring+ZFP(rate:10)=logs/ring_zfp_rate10_8gpu.out" \
  "16:Ring+ZFP(rate:10)=logs/ring_zfp_rate10_16gpu.out"
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EPOCH_RE = re.compile(r"\[Epoch\s+(\d+)\].*?\|\s*Time:\s*([0-9.eE+-]+)s")
SUMMARY_RE = re.compile(
    r"Epoch times:\s*mean=([0-9.eE+-]+)s\s+min=([0-9.eE+-]+)s\s+max=([0-9.eE+-]+)s"
)
DONE_RE = re.compile(r"Done in\s*([0-9.eE+-]+)s\s*\|\s*Best val acc:\s*([0-9.eE+-]+)%")
WORK_RE = re.compile(
    r"\[HOOK_WORK_SUMMARY\].*?work_total_ms=([0-9.eE+-]+).*?work_mean_ms=([0-9.eE+-]+)"
)
TAIL_RE = re.compile(
    r"\[HOOK_TAIL_SUMMARY\].*?tail_total_ms=([0-9.eE+-]+).*?tail_mean_ms=([0-9.eE+-]+)"
)


def mean(values):
    return sum(values) / len(values) if values else None


def parse_log(path):
    text = Path(path).read_text(errors="replace")

    epoch_times = [float(m.group(2)) for m in EPOCH_RE.finditer(text)]

    summary = SUMMARY_RE.search(text)
    done = DONE_RE.search(text)

    work_total, work_mean = [], []
    tail_total, tail_mean = [], []

    for m in WORK_RE.finditer(text):
        work_total.append(float(m.group(1)))
        work_mean.append(float(m.group(2)))

    for m in TAIL_RE.finditer(text):
        tail_total.append(float(m.group(1)))
        tail_mean.append(float(m.group(2)))

    return {
        "epoch_times": epoch_times,
        "epoch_mean": float(summary.group(1)) if summary else mean(epoch_times),
        "epoch_min": float(summary.group(2)) if summary else min(epoch_times),
        "epoch_max": float(summary.group(3)) if summary else max(epoch_times),
        "total_time": float(done.group(1)) if done else None,
        "best_val_acc": float(done.group(2)) if done else None,
        "hook_work_mean_ms": mean(work_mean),
        "hook_tail_mean_ms": mean(tail_mean),
        "hook_work_total_ms": mean(work_total),
        "hook_tail_total_ms": mean(tail_total),
    }


def metric_value(stats, metric, skip_first_epoch):
    if metric == "epoch_mean":
        return mean(stats["epoch_times"][skip_first_epoch:])
    return stats[metric]


def parse_scaling_item(item):
    if ":" not in item or "=" not in item:
        raise ValueError(f"Expected GPUS:HOOK=PATH, got: {item}")

    gpu_text, rest = item.split(":", 1)
    hook, path = rest.split("=", 1)

    return int(gpu_text), hook.strip(), path.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="Items like '8:Baseline=path/to/log.out'")
    parser.add_argument("--model", default="Wide_ResNet50_2")
    parser.add_argument("--out", default="hook_scaling.png")
    parser.add_argument(
        "--metric",
        default="epoch_mean",
        choices=[
            "epoch_mean",
            "epoch_min",
            "epoch_max",
            "total_time",
            "best_val_acc",
            "hook_work_mean_ms",
            "hook_tail_mean_ms",
            "hook_work_total_ms",
            "hook_tail_total_ms",
        ],
    )
    parser.add_argument("--skip-first-epoch", type=int, default=1)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    data = {}
    gpu_counts = set()
    hooks = []

    for item in args.logs:
        gpus, hook, path = parse_scaling_item(item)
        stats = parse_log(path)
        value = metric_value(stats, args.metric, args.skip_first_epoch)

        if value is None:
            raise ValueError(f"Metric {args.metric} is missing in {path}")

        data[(gpus, hook)] = value
        gpu_counts.add(gpus)

        if hook not in hooks:
            hooks.append(hook)

    gpu_counts = sorted(gpu_counts)
    x = np.arange(len(gpu_counts))

    width = min(0.8 / max(len(hooks), 1), 0.18)

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.Set1(np.linspace(0, 1, len(hooks)))

    for i, hook in enumerate(hooks):
        offset = (i - (len(hooks) - 1) / 2) * width
        values = [data.get((gpus, hook), np.nan) for gpus in gpu_counts]
        ax.bar(
            x + offset,
            values,
            width,
            label=hook,
            color=colors[i],
            edgecolor="black",
            linewidth=0.6,
        )

    ylabel = {
        "epoch_mean": "Training time / epoch (s)",
        "epoch_min": "Min epoch time (s)",
        "epoch_max": "Max epoch time (s)",
        "total_time": "Total training time (s)",
        "best_val_acc": "Best validation accuracy (%)",
        "hook_work_mean_ms": "Hook work mean (ms)",
        "hook_tail_mean_ms": "Hook tail mean (ms)",
        "hook_work_total_ms": "Hook work total / epoch (ms)",
        "hook_tail_total_ms": "Hook tail total / epoch (ms)",
    }[args.metric]

    ax.set_title(args.title or args.model, fontsize=24, fontweight="bold")
    ax.set_xlabel("GPUs", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)

    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in gpu_counts], fontsize=15)
    ax.tick_params(axis="y", labelsize=14)

    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(fontsize=12, frameon=False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()