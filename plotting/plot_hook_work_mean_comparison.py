#!/usr/bin/env python3
"""
Compare communication hook work time across logs.

This plots, for each hook/log:
    mean(all epoch hook work_mean_ms values)

Built-in/no-hook logs are skipped by default because they do not contain
HOOK_WORK_SUMMARY lines.

Example:
python plot_hook_work_mean_comparison.py \
  --model Wide_ResNet50_2 \
  --gpus 8 \
  --out hook_work_mean_8gpu.png \
  "Ring+ZFP(rate:10)=logs/ring_zfp_online_rate10_8gpu.out" \
  "RD+ZFP(rate:8)=logs/rd_zfp_rate8_8gpu.out" \
  "Ring=logs/ring_8gpu.out"
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


WORK_RE = re.compile(
    r"\[HOOK_WORK_SUMMARY\].*?"
    r"label=(\S+)\s+"
    r"calls=([0-9]+)\s+"
    r"work_total_ms=([0-9.eE+-]+)\s+"
    r"work_mean_ms=([0-9.eE+-]+)\s+"
    r"work_min_ms=([0-9.eE+-]+)\s+"
    r"work_max_ms=([0-9.eE+-]+)"
)


def mean(values):
    return sum(values) / len(values) if values else None


def parse_named_log(item):
    if "=" not in item:
        raise ValueError(f"Expected NAME=PATH, got: {item}")
    name, path = item.split("=", 1)
    return name.strip(), path.strip()


def parse_hook_work_means(path):
    text = Path(path).read_text(errors="replace")

    means = []
    totals = []
    mins = []
    maxs = []
    calls = []
    labels = []

    for match in WORK_RE.finditer(text):
        labels.append(match.group(1))
        calls.append(int(match.group(2)))
        totals.append(float(match.group(3)))
        means.append(float(match.group(4)))
        mins.append(float(match.group(5)))
        maxs.append(float(match.group(6)))

    return {
        "labels": labels,
        "calls": calls,
        "work_total_ms": totals,
        "work_mean_ms": means,
        "work_min_ms": mins,
        "work_max_ms": maxs,
        "mean_of_work_means_ms": mean(means),
        "mean_of_work_totals_ms": mean(totals),
        "min_work_mean_ms": min(means) if means else None,
        "max_work_mean_ms": max(means) if means else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="Pairs like 'Hook name=path/to/log.out'")
    parser.add_argument("--model", default="Wide_ResNet50_2")
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--out", default="hook_work_mean_comparison.png")
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--show-error-bars",
        action="store_true",
        help="Show min/max range of per-epoch work_mean_ms values.",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Fail if a log has no HOOK_WORK_SUMMARY lines instead of skipping it.",
    )
    args = parser.parse_args()

    names = []
    values = []
    lower_errors = []
    upper_errors = []

    skipped = []

    for item in args.logs:
        name, path = parse_named_log(item)
        stats = parse_hook_work_means(path)

        value = stats["mean_of_work_means_ms"]
        if value is None:
            message = f"No HOOK_WORK_SUMMARY found in {path}"
            if args.fail_on_missing:
                raise ValueError(message)
            skipped.append((name, path))
            continue

        names.append(name)
        values.append(value)

        min_value = stats["min_work_mean_ms"]
        max_value = stats["max_work_mean_ms"]
        lower_errors.append(value - min_value)
        upper_errors.append(max_value - value)

    if not names:
        raise ValueError("No hook work timing data found in any input log.")

    fig, ax = plt.subplots(figsize=(11, 6))

    colors = plt.cm.Set2(range(len(names)))

    yerr = None
    if args.show_error_bars:
        yerr = [lower_errors, upper_errors]

    bars = ax.bar(
        names,
        values,
        yerr=yerr,
        capsize=5 if args.show_error_bars else 0,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
    )

    gpu_text = f" - {args.gpus} GPUs" if args.gpus is not None else ""
    ax.set_title(
        args.title or f"{args.model}{gpu_text}: Mean Hook Work Time",
        fontsize=21,
        fontweight="bold",
    )
    ax.set_xlabel("Communication hook", fontsize=15)
    ax.set_ylabel("Mean of epoch hook work means (ms)", fontsize=15)

    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=25, labelsize=11)
    ax.tick_params(axis="y", labelsize=13)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(args.out, dpi=300)

    print(f"Saved {args.out}")

    if skipped:
        print("Skipped logs without hook work timing:")
        for name, path in skipped:
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()