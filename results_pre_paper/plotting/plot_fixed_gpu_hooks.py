#!/usr/bin/env python3
"""
Plot one model at one fixed GPU count, with one bar per communication hook.

Run from:
    ddp-allreduce-eval-framework/plotting/

Example for 8 GPUs:
    python plot_fixed_gpu_hooks.py \
        --results-root experiments_results_lr001_GlobalBS128 \
        --gpu-dir 8GPUs_2nodes \
        --model-dir wideresnet50 \
        --run-dir run1 \
        --model-title Wide_ResNet50_2 \
        --gpus-label "8 GPUs" \
        --metric epoch_mean \
        --out wide_resnet50_2_8gpu_epoch_mean.pdf
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

from ddp_plot_common import apply_paper_style, add_grouped_hook_legend


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


METRIC_LABELS = {
    "epoch_mean": "Training Time / Epoch (s)",
    "total_time": "Total Training Time (s)",
    "best_val_acc": "Best Validation Accuracy (%)",
    "hook_work_mean_ms": "Mean Hook Work Time (ms)",
    "hook_tail_mean_ms": "Mean Exposed Tail Time (ms)",
    "hook_work_total_ms": "Mean Hook Work / Epoch (ms)",
    "hook_tail_total_ms": "Mean Exposed Tail / Epoch (ms)",
}


HOOK_ORDER = [
    "Baseline",
    "Ring",
    "Ring+ZFP naive (rate:16)",
    "Ring+ZFP online (rate:16)",
    "Ring+ZFP online (rate:10)",
    "Recursive doubling",
    "RD+ZFP naive (rate:16)",
    "RD+ZFP online (rate:16)",
    "RD+ZFP online (rate:8)",
]

HOOK_X = {
    "Baseline": 0.0,
    "Ring": 1.7,
    "Ring+ZFP naive (rate:16)": 2.25,
    "Ring+ZFP online (rate:16)": 2.80,
    "Ring+ZFP online (rate:10)": 3.35,
    "Recursive doubling": 5.05,
    "RD+ZFP naive (rate:16)": 5.60,
    "RD+ZFP online (rate:16)": 6.15,
    "RD+ZFP online (rate:8)": 6.70,
}

HOOK_COLORS = {
    "Baseline": "#4472C4",
    "Ring": "#70AD47",
    "Ring+ZFP naive (rate:16)": "#ED7D31",
    "Ring+ZFP online (rate:16)": "#00B050",
    "Ring+ZFP online (rate:10)": "#92D050",
    "Recursive doubling": "#A64D79",
    "RD+ZFP naive (rate:16)": "#FF0000",
    "RD+ZFP online (rate:16)": "#FFC000",
    "RD+ZFP online (rate:8)": "#F4B183",
}


def mean(values):
    return sum(values) / len(values) if values else None


def repo_root():
    return Path(__file__).resolve().parents[1]


def result_dir(results_root, gpu_dir, model_dir, run_dir=None):
    root = Path(results_root)
    if not root.is_absolute():
        root = repo_root() / root

    path = root / gpu_dir / model_dir
    if run_dir:
        path = path / run_dir

    return path


def hook_label_from_filename(path):
    name = Path(path).stem.lower()

    if "builtin" in name or "buildin" in name:
        return "Baseline"
    if "recursive_doubling_zfp_online_coll" in name:
        label = "RD+ZFP online"
    elif "recursive_doubling_zfp_naive" in name:
        label = "RD+ZFP naive"
    elif "recursive_doubling" in name:
        label = "Recursive doubling"
    elif "ring_zfp_online_coll" in name:
        label = "Ring+ZFP online"
    elif "ring_zfp_naive" in name:
        label = "Ring+ZFP naive"
    elif re.search(r"_ring$", name):
        label = "Ring"
    else:
        label = Path(path).stem

    rate = re.search(r"rate([0-9.]+)", name)
    if rate:
        label += f" (rate:{rate.group(1)})"

    return label


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
        "epoch_mean_from_summary": float(summary.group(1)) if summary else None,
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


def sort_key(row):
    label = row[0]
    if label in HOOK_ORDER:
        return HOOK_ORDER.index(label)
    return len(HOOK_ORDER)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--gpu-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--model-title", default="Wide_ResNet50_2")
    parser.add_argument("--gpus-label", required=True)
    parser.add_argument("--metric", choices=METRIC_LABELS.keys(), default="epoch_mean")
    parser.add_argument("--skip-first-epoch", type=int, default=1)
    parser.add_argument("--out", default="fixed_gpu_hooks.pdf")
    args = parser.parse_args()

    apply_paper_style(plt)

    directory = result_dir(
        args.results_root, args.gpu_dir, args.model_dir, args.run_dir
    )
    if not directory.exists():
        raise SystemExit(f"Directory does not exist: {directory}")

    rows = []
    for log in sorted(directory.glob("*.log")):
        label = hook_label_from_filename(log)
        stats = parse_log(log)
        value = metric_value(stats, args.metric, args.skip_first_epoch)

        if value is not None:
            rows.append((label, value, log))

    if not rows:
        raise SystemExit(f"No usable values found for metric: {args.metric}")

    rows = sorted(rows, key=sort_key)

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    x_positions = [HOOK_X[label] for label in labels]
    colors = [HOOK_COLORS.get(label, "#999999") for label in labels]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    bars = ax.bar(
        x_positions,
        values,
        width=0.38,
        color=colors,
        edgecolor="black",
        linewidth=0.35,
    )

    ax.set_title(
        f"{args.model_title} - {args.gpus_label}",
        fontsize=21,
        fontweight="normal",
        pad=7,
    )

    ax.set_ylabel(METRIC_LABELS[args.metric], fontsize=20)
    ax.set_xlabel("Communication Hook", fontsize=19)

    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=15)

    ax.grid(axis="y", alpha=0.25, linewidth=0.55)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.012,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    add_grouped_hook_legend(ax, fontsize=10.5, y=-0.08)

    ax.set_xlim(-0.45, 7.15)
    ax.set_ylim(0, max(values) * 1.18)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Read logs from: {directory}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
