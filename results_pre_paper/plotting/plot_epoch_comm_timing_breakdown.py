#!/usr/bin/env python3
"""
Create an epoch-level communication accounting figure.

The script selects the worst combined communication case by default.
If no --results-root is provided, it searches both GlobalBS128 and LocalBS128.

It plots:
    1. Whole epoch wall-clock time as the full horizontal extent.
    2. Accumulated hook work right-aligned to the epoch end.
    3. Accumulated exposed tail right-aligned to the epoch end.

This is not a literal timeline. The bars are an accounting view:
    epoch time     = real elapsed wall-clock epoch time
    hook work time = sum of hook work over bucket hook calls
    tail time      = sum of exposed communication tail over training steps

The right-alignment is intentionally visual: it helps explain that both measured
communication work and exposed tail are end-of-epoch accounting quantities, with
the tail being the final non-overlapped part.

Run from:
    ddp-allreduce-eval-framework/plotting/

Example:
    python plot_epoch_comm_timing_breakdown.py \
        --results-root experiments_results_lr001_GlobalBS128 \
        --model-dir wideresnet \
        --model-title Wide_ResNet50_2 \
        --gpus 4 \
        --hook-contains ring_zfp_online_coll_rate10 \
        --out wide_resnet50_2_4gpu_comm_accounting.pdf

Search all current GlobalBS128 and LocalBS128 results and plot the worst combined case:
    python plot_epoch_comm_timing_breakdown.py \
        --model-dir wideresnet \
        --model-title Wide_ResNet50_2 \
        --out wide_resnet50_2_worst_comm_accounting.pdf

The default ignores extremely pathological cases where the exposed tail is more
than 50% of the epoch wall time. To include every row, pass:
    --max-tail-fraction 0

Search all current results and force selection by only accumulated hook work:
    python plot_epoch_comm_timing_breakdown.py \
        --model-dir wideresnet \
        --model-title Wide_ResNet50_2 \
        --select-by work \
        --out wide_resnet50_2_worst_work_accounting.pdf

Choose a specific epoch instead of worst-tail epoch:
    python plot_epoch_comm_timing_breakdown.py \
        --results-root experiments_results_lr001_GlobalBS128 \
        --model-dir wideresnet \
        --model-title Wide_ResNet50_2 \
        --gpus 4 \
        --hook-contains ring_zfp_online_coll_rate10 \
        --epoch 12
"""

import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ddp_plot_common import (
    apply_paper_style,
    find_logs,
    parse_gpu_count,
    pretty_hook_from_filename,
)

EPOCH_RE = re.compile(r"\[Epoch\s+(\d+)\].*?\|\s*Time:\s*([0-9.eE+-]+)s")
BLOCK_RE = re.compile(
    r"\[HOOK_TIMING_SUMMARY\]\s+epoch=([0-9]+).*?"
    r"\[HOOK_WORK_SUMMARY\].*?work_total_ms=([0-9.eE+-]+).*?work_mean_ms=([0-9.eE+-]+).*?"
    r"\[HOOK_TAIL_SUMMARY\].*?tail_total_ms=([0-9.eE+-]+).*?tail_mean_ms=([0-9.eE+-]+)",
    re.DOTALL,
)


def parse_rows(path):
    text = Path(path).read_text(errors="replace")
    epoch_times = {
        int(m.group(1)): float(m.group(2)) * 1000 for m in EPOCH_RE.finditer(text)
    }

    rows = []
    for m in BLOCK_RE.finditer(text):
        epoch = int(m.group(1))
        rows.append(
            {
                "epoch": epoch,
                "epoch_ms": epoch_times.get(epoch),
                "work_total_ms": float(m.group(2)),
                "work_mean_ms": float(m.group(3)),
                "tail_total_ms": float(m.group(4)),
                "tail_mean_ms": float(m.group(5)),
            }
        )
    return rows


def collect_candidate_rows(results_roots, model_dir, gpus, run, hook_contains):
    candidates = []
    for root in results_roots:
        for log in find_logs(root, model_dir, gpus=gpus, run=run):
            if hook_contains and hook_contains.lower() not in str(log).lower():
                continue
            for row in parse_rows(log):
                if row["epoch_ms"] is None:
                    continue
                candidates.append(
                    {
                        **row,
                        "log": log,
                        "results_root": root,
                        "gpus": parse_gpu_count(log),
                        "hook_name": pretty_hook_from_filename(log),
                    }
                )
    return candidates


def add_combined_scores(candidates):
    max_tail = max(row["tail_total_ms"] for row in candidates)
    max_work = max(row["work_total_ms"] for row in candidates)

    for row in candidates:
        tail_score = row["tail_total_ms"] / max_tail if max_tail else 0.0
        work_score = row["work_total_ms"] / max_work if max_work else 0.0
        row["tail_score"] = tail_score
        row["work_score"] = work_score
        row["combined_score"] = 0.5 * tail_score + 0.5 * work_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-root",
        action="append",
        default=None,
        help=(
            "Result root to search. May be passed multiple times. "
            "Default: GlobalBS128 and LocalBS128."
        ),
    )
    p.add_argument("--model-dir", default="wideresnet")
    p.add_argument("--model-title", default="Wide_ResNet50_2")
    p.add_argument("--gpus", type=int, default=None)
    p.add_argument(
        "--hook-contains",
        default=None,
        help="Optional filename filter, for example: ring_zfp_online_coll_rate10",
    )
    p.add_argument(
        "--select-by",
        choices=["combined", "tail", "work"],
        default="combined",
        help=(
            "Choose the worst epoch by normalized combined communication score, "
            "accumulated exposed tail, or accumulated hook work."
        ),
    )
    p.add_argument(
        "--max-tail-fraction",
        type=float,
        default=0.50,
        help=(
            "Ignore rows whose tail_total_ms / epoch_ms is above this value. "
            "Default 0.50 chooses a bad but more plausible explanatory case. "
            "Use 0 to disable this filter."
        ),
    )
    p.add_argument("--run", default=None)
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    apply_paper_style(plt)

    results_roots = args.results_root or [
        "experiments_results_lr001_GlobalBS128",
        "experiments_results_lr001_LocalBS128",
    ]

    candidates = collect_candidate_rows(
        results_roots=results_roots,
        model_dir=args.model_dir,
        gpus=args.gpus,
        run=args.run,
        hook_contains=args.hook_contains,
    )
    if not candidates:
        raise SystemExit("No hook timing rows found in the selected results.")

    if args.epoch is not None:
        candidates = [row for row in candidates if row["epoch"] == args.epoch]
        if not candidates:
            raise SystemExit(f"No hook timing rows found for epoch {args.epoch}.")

    unfiltered_count = len(candidates)
    if args.max_tail_fraction > 0:
        filtered = [
            row
            for row in candidates
            if row["tail_total_ms"] / row["epoch_ms"] <= args.max_tail_fraction
        ]
        if filtered:
            candidates = filtered
        else:
            print(
                "Warning: max-tail-fraction removed all rows; "
                "falling back to unfiltered candidates."
            )

    add_combined_scores(candidates)
    worst_tail = max(candidates, key=lambda r: r["tail_total_ms"])
    worst_work = max(candidates, key=lambda r: r["work_total_ms"])
    worst_combined = max(candidates, key=lambda r: r["combined_score"])
    if args.select_by == "work":
        row = worst_work
    elif args.select_by == "tail":
        row = worst_tail
    else:
        row = worst_combined

    log = row["log"]

    epoch_ms = row["epoch_ms"]
    work_ms = row["work_total_ms"]
    tail_ms = row["tail_total_ms"]

    # -------------------------------------------------------------------------
    # How to read this figure
    # -------------------------------------------------------------------------
    #
    # This drawing is a timing-accounting diagram, not a literal CUDA/MPI trace.
    #
    # epoch_ms:
    #     The real wall-clock duration of the epoch. This is the horizontal
    #     reference frame of the figure, from 0 to epoch_ms.
    #
    # work_ms:
    #     The accumulated sum of all hook work measured during the epoch. In DDP,
    #     communication hooks are called many times, usually once per bucket.
    #     The log's work_total_ms is the sum of those hook-work durations.
    #
    # tail_ms:
    #     The accumulated exposed communication tail. This is the part that was
    #     not hidden by overlap with backward computation. Because it is called
    #     "tail", the figure places it at the end of the epoch.
    #
    # Right-aligning communication:
    #     We draw both accumulated communication quantities so that they end at
    #     epoch_ms. The hook-work line is the sum of all hook work in the epoch,
    #     right-aligned to suggest that communication must be completed by the
    #     epoch boundary. The tail line is below it and also right-aligned,
    #     because it represents the final exposed, non-overlapped portion.
    #
    #     This is intentionally visual and explanatory. The real hook calls are
    #     spread throughout backward passes over many batches, not one single
    #     continuous communication block. Right-alignment answers the conceptual
    #     question: "How much accumulated communication work is there, and how
    #     much of it is still exposed at the tail end?"
    #
    # Important:
    #     epoch_ms, work_ms, and tail_ms do not add up. They are different
    #     measurements of the same epoch. Large hook work is not necessarily bad
    #     if most of it overlaps; large tail is the part that is visibly costly.
    # -------------------------------------------------------------------------

    comm_start = epoch_ms - work_ms
    tail_start = epoch_ms - tail_ms
    x_min = min(0.0, comm_start)
    x_max = epoch_ms

    fig, ax = plt.subplots(figsize=(7.4, 3.25))

    epoch_y = 1.16
    work_y = 0.78
    tail_y = 0.40
    height = 0.22

    ax.add_patch(
        Rectangle(
            (0, epoch_y),
            epoch_ms,
            height,
            facecolor="#4C72B0",
            edgecolor="black",
            linewidth=0.35,
        )
    )

    ax.add_patch(
        Rectangle(
            (comm_start, work_y),
            work_ms,
            height,
            facecolor="#55A868",
            edgecolor="black",
            linewidth=0.35,
        )
    )

    ax.add_patch(
        Rectangle(
            (tail_start, tail_y),
            tail_ms,
            height,
            facecolor="#C44E52",
            edgecolor="black",
            linewidth=0.35,
        )
    )

    ax.vlines(epoch_ms, tail_y, 1.50, color="black", linewidth=0.7)
    ax.text(epoch_ms, 1.53, "epoch end", ha="right", va="bottom", fontsize=9.5)

    ax.set_yticks([epoch_y + height / 2, work_y + height / 2, tail_y + height / 2])
    ax.set_yticklabels(
        ["Whole epoch", "Accumulated hook work", "Exposed tail"], fontsize=13
    )
    ax.set_title(
        f"{args.model_title} - {row['gpus']} GPUs - Epoch {row['epoch']}",
        fontsize=17,
        fontweight="normal",
        pad=7,
    )

    ax.text(0, 1.58, row["hook_name"], fontsize=10.5, va="bottom")

    ax.text(
        epoch_ms / 2,
        epoch_y + height / 2,
        f"epoch wall time - {epoch_ms:.1f} ms",
        ha="center",
        va="center",
        fontsize=10,
        color="white",
    )
    ax.text(
        comm_start + work_ms / 2,
        work_y + height / 2,
        f"hook work - {work_ms:.1f} ms",
        ha="center",
        va="center",
        fontsize=9.5,
        color="white",
    )
    ax.text(
        tail_start + tail_ms / 2,
        tail_y + height / 2,
        f"tail - {tail_ms:.1f} ms",
        ha="center",
        va="center",
        fontsize=9.5,
        color="white",
    )

    ax.set_xlim(x_min - (x_max - x_min) * 0.03, x_max * 1.03)
    ax.set_ylim(0.28, 1.77)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    gpu_name = f"{row['gpus']}gpu" if row["gpus"] is not None else "allgpus"
    out = args.out or f"{args.model_dir}_{gpu_name}_epoch_comm_breakdown.pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")

    print(f"Selected by: {args.select_by}")
    if args.max_tail_fraction > 0:
        print(
            "Tail plausibility filter: "
            f"tail/epoch <= {args.max_tail_fraction:.2f} "
            f"({len(candidates)}/{unfiltered_count} rows kept)"
        )
    else:
        print("Tail plausibility filter: disabled")
    print(f"Using log: {log}")
    print(f"Selected epoch: {row['epoch']}")
    print(
        "Selected combined score: "
        f"{row['combined_score']:.4f} "
        f"(tail_norm={row['tail_score']:.4f}, work_norm={row['work_score']:.4f})"
    )
    print(
        "Worst combined: "
        f"score={worst_combined['combined_score']:.4f}, "
        f"tail={worst_combined['tail_total_ms']:.3f} ms, "
        f"work={worst_combined['work_total_ms']:.3f} ms, "
        f"epoch {worst_combined['epoch']}, "
        f"{worst_combined['gpus']} GPUs, "
        f"{worst_combined['hook_name']}, "
        f"{worst_combined['log']}"
    )
    print(
        "Worst tail: "
        f"{worst_tail['tail_total_ms']:.3f} ms, "
        f"epoch {worst_tail['epoch']}, "
        f"{worst_tail['gpus']} GPUs, "
        f"{worst_tail['hook_name']}, "
        f"{worst_tail['log']}"
    )
    print(
        "Worst hook work: "
        f"{worst_work['work_total_ms']:.3f} ms, "
        f"epoch {worst_work['epoch']}, "
        f"{worst_work['gpus']} GPUs, "
        f"{worst_work['hook_name']}, "
        f"{worst_work['log']}"
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
