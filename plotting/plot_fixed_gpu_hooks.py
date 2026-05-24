#!/usr/bin/env python3
"""
Compare many communication hooks for one model at one fixed GPU count.

Example:
python plot_fixed_gpu_hooks.py \
  --model Wide_ResNet50_2 \
  --gpus 8 \
  --metric epoch_mean \
  --out wide_resnet50_2_8gpu_hooks.png \
  "Baseline=logs/builtin_8gpu.out" \
  "Ring+ZFP(rate:10)=logs/ring_zfp_online_rate10_8gpu.out"
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


EPOCH_RE = re.compile(
    r"\[Epoch\s+(\d+)\].*?Train Loss:\s*([0-9.eE+-]+), Acc:\s*([0-9.eE+-]+)%"
    r"\s*\|\s*Val Loss:\s*([0-9.eE+-]+), Acc:\s*([0-9.eE+-]+)%"
    r"\s*\|\s*Time:\s*([0-9.eE+-]+)s"
)
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

    epochs = []
    for m in EPOCH_RE.finditer(text):
        epochs.append(
            {
                "epoch": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "train_acc": float(m.group(3)),
                "val_loss": float(m.group(4)),
                "val_acc": float(m.group(5)),
                "epoch_time": float(m.group(6)),
            }
        )

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
        "epoch_times": [e["epoch_time"] for e in epochs],
        "val_acc": [e["val_acc"] for e in epochs],
        "epoch_mean": float(summary.group(1)) if summary else mean([e["epoch_time"] for e in epochs]),
        "epoch_min": float(summary.group(2)) if summary else min([e["epoch_time"] for e in epochs]),
        "epoch_max": float(summary.group(3)) if summary else max([e["epoch_time"] for e in epochs]),
        "total_time": float(done.group(1)) if done else None,
        "best_val_acc": float(done.group(2)) if done else (max([e["val_acc"] for e in epochs]) if epochs else None),
        "hook_work_mean_ms": mean(work_mean),
        "hook_tail_mean_ms": mean(tail_mean),
        "hook_work_total_ms": mean(work_total),
        "hook_tail_total_ms": mean(tail_total),
    }


def metric_value(stats, metric, skip_first_epoch):
    if metric == "epoch_mean":
        values = stats["epoch_times"][skip_first_epoch:]
        return mean(values)
    return stats[metric]


def parse_named_log(item):
    if "=" not in item:
        raise ValueError(f"Expected NAME=PATH, got: {item}")
    name, path = item.split("=", 1)
    return name.strip(), path.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="Pairs like 'Baseline=path/to/log.out'")
    parser.add_argument("--model", default="Wide_ResNet50_2")
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--out", default="fixed_gpu_hooks.png")
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

    names, values, accs = [], [], []

    for item in args.logs:
        name, path = parse_named_log(item)
        stats = parse_log(path)
        value = metric_value(stats, args.metric, args.skip_first_epoch)
        if value is None:
            raise ValueError(f"Metric {args.metric} is missing in {path}")
        names.append(name)
        values.append(value)
        accs.append(stats["best_val_acc"])

    fig, ax = plt.subplots(figsize=(11, 6))

    colors = plt.cm.Set2(range(len(names)))
    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.7)

    ax.set_title(args.title or f"{args.model} - {args.gpus} GPUs", fontsize=22, fontweight="bold")
    ax.set_xlabel("Communication hook", fontsize=16)

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
    ax.set_ylabel(ylabel, fontsize=16)

    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=25, labelsize=12)
    ax.tick_params(axis="y", labelsize=13)

    for bar, value, acc in zip(bars, values, accs):
        label = f"{value:.2f}"
        if acc is not None and args.metric != "best_val_acc":
            label += f"\nacc {acc:.1f}%"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()