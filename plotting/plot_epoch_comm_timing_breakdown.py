#!/usr/bin/env python3
"""
Plot one epoch timing breakdown to explain communication measurement.

This figure is intentionally an ACCOUNTING VIEW, not a literal time-line.

For the selected epoch, it plots:
1. The real epoch wall-clock time.
2. The accumulated hook work time.
3. The accumulated exposed communication tail time.

Example:
python plot_epoch_comm_timing_breakdown.py \
  --log logs/ring_zfp_online_rate10_8gpu.out \
  --model Wide_ResNet50_2 \
  --hook "Ring+ZFP(rate:10)" \
  --gpus 8 \
  --out epoch_comm_breakdown.png
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


EPOCH_RE = re.compile(
    r"\[Epoch\s+(\d+)\].*?Train Loss:.*?\|\s*Val Loss:.*?\|\s*Time:\s*([0-9.eE+-]+)s"
)

HOOK_BLOCK_RE = re.compile(
    r"\[HOOK_TIMING_SUMMARY\]\s+epoch=([0-9]+).*?"
    r"\[HOOK_WORK_SUMMARY\].*?work_total_ms=([0-9.eE+-]+).*?work_mean_ms=([0-9.eE+-]+).*?"
    r"\[HOOK_TAIL_SUMMARY\].*?tail_total_ms=([0-9.eE+-]+).*?tail_mean_ms=([0-9.eE+-]+)",
    re.DOTALL,
)


def parse_log(path):
    text = Path(path).read_text(errors="replace")

    epoch_times = {}
    for match in EPOCH_RE.finditer(text):
        epoch = int(match.group(1))
        epoch_times[epoch] = float(match.group(2)) * 1000.0

    hook_rows = []
    for match in HOOK_BLOCK_RE.finditer(text):
        epoch = int(match.group(1))
        hook_rows.append(
            {
                "epoch": epoch,
                "epoch_time_ms": epoch_times.get(epoch),
                "work_total_ms": float(match.group(2)),
                "work_mean_ms": float(match.group(3)),
                "tail_total_ms": float(match.group(4)),
                "tail_mean_ms": float(match.group(5)),
            }
        )

    if not hook_rows:
        raise ValueError("No hook timing summaries found in this log.")

    return hook_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--model", default="Wide_ResNet50_2")
    parser.add_argument("--hook", default="Communication hook")
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--out", default="epoch_comm_timing_breakdown.png")
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Manually choose epoch. Default: epoch with largest tail_total_ms.",
    )
    args = parser.parse_args()

    rows = parse_log(args.log)

    if args.epoch is None:
        row = max(rows, key=lambda r: r["tail_total_ms"])
    else:
        matches = [r for r in rows if r["epoch"] == args.epoch]
        if not matches:
            raise ValueError(f"Epoch {args.epoch} not found in hook timing summaries.")
        row = matches[0]

    if row["epoch_time_ms"] is None:
        raise ValueError("Could not match epoch training time for selected epoch.")

    epoch_ms = row["epoch_time_ms"]
    work_ms = row["work_total_ms"]
    tail_ms = row["tail_total_ms"]

    # -------------------------------------------------------------------------
    # What these three quantities mean
    # -------------------------------------------------------------------------
    #
    # epoch_ms:
    #     This is the real wall-clock training time for the whole epoch.
    #     It comes from the line:
    #
    #         [Epoch N] ... | Time: 15.9s
    #
    #     This is the actual elapsed time observed by the training loop.
    #
    #
    # work_ms:
    #     This is NOT a separate block of time that happened after computation.
    #     It is the accumulated sum of all communication-hook work measured
    #     during the epoch.
    #
    #     In DDP, the model gradients are split into buckets. As backward
    #     computation produces each bucket, PyTorch invokes the registered
    #     communication hook for that bucket. So during one epoch there are many
    #     hook invocations:
    #
    #         bucket 0 hook work
    #         bucket 1 hook work
    #         bucket 2 hook work
    #         ...
    #
    #     The log line:
    #
    #         [HOOK_WORK_SUMMARY] ... calls=3128 work_total_ms=10268.977
    #
    #     means:
    #
    #         work_total_ms = sum(duration of all measured hook work calls)
    #
    #     across the epoch.
    #
    #     Therefore, work_total_ms is an epoch-level accumulated total over many
    #     small communication events. It is not necessarily equal to visible
    #     training slowdown, because much of this hook work can be overlapped
    #     with backward computation. For example, while earlier gradient buckets
    #     are being communicated, the GPU may still be computing later layers'
    #     gradients.
    #
    #
    # tail_ms:
    #     This is also an epoch-level accumulated total, but it measures a
    #     different idea.
    #
    #     The communication tail is the part of communication that remains
    #     exposed after computation overlap has been used up. In other words,
    #     this is the part that the training loop effectively has to wait for.
    #
    #     Your log line:
    #
    #         [HOOK_TAIL_SUMMARY] ... steps=391 tail_total_ms=6509.535
    #
    #     means:
    #
    #         tail_total_ms = sum(exposed communication tail over all steps)
    #
    #     across the epoch.
    #
    #     Notice the different counting:
    #
    #         work_total_ms is summed over hook calls / buckets.
    #         tail_total_ms is summed over training steps / batches.
    #
    #     That is why the example has around 3128 hook calls but around 391
    #     tail steps. In that run, each training step triggers multiple bucket
    #     communication hooks, but the exposed tail is summarized once per step.
    #
    #
    # The key interpretation:
    #
    #     epoch_ms, work_ms, and tail_ms do NOT add up.
    #
    #     epoch_ms is the real elapsed epoch time.
    #     work_ms is the total amount of communication-hook activity measured.
    #     tail_ms is the part of communication that remained exposed to the
    #     training loop after overlap with computation.
    #
    # A hook can have a large work_total_ms and still be fast if most of that
    # work overlaps with computation. Conversely, a hook with smaller work may
    # still hurt training time if its work appears late and creates a large tail.
    #
    # This figure should therefore be read as:
    #
    #     "Within this epoch, the training loop took epoch_ms total. Across that
    #      epoch, we measured work_ms of communication-hook activity. Of the
    #      communication behavior, tail_ms was exposed rather than hidden by
    #      overlap."
    #
    # It is deliberately not a literal chronological schedule.
    # -------------------------------------------------------------------------

    overlapped_work_ms = max(work_ms - tail_ms, 0.0)

    labels = [
        "Whole epoch wall time",
        "Accumulated hook work",
        "Accumulated exposed tail",
    ]
    values = [
        epoch_ms,
        work_ms,
        tail_ms,
    ]

    colors = [
        "#4C72B0",
        "#55A868",
        "#C44E52",
    ]

    fig, ax = plt.subplots(figsize=(11, 5))

    y = [2, 1, 0]
    bars = ax.barh(y, values, color=colors, edgecolor="black", height=0.42)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_xlabel("Accumulated time in selected epoch (ms)", fontsize=14)

    gpu_text = f", {args.gpus} GPUs" if args.gpus is not None else ""
    ax.set_title(
        f"{args.model}{gpu_text} - Epoch {row['epoch']} Communication Accounting",
        fontsize=18,
        fontweight="bold",
    )

    ax.text(
        0,
        2.72,
        f"{args.hook}: tail is the exposed part; hook work may overlap with backward computation",
        fontsize=11,
        va="center",
    )

    for bar, value in zip(bars, values):
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f"  {value:.1f} ms",
            va="center",
            fontsize=12,
        )

    if work_ms > 0:
        tail_pct_of_work = 100.0 * tail_ms / work_ms
        overlap_pct_of_work = 100.0 * overlapped_work_ms / work_ms

        ax.text(
            0,
            -0.72,
            f"Approx. exposed tail: {tail_pct_of_work:.1f}% of hook work | "
            f"approx. overlapped hook work: {overlap_pct_of_work:.1f}%",
            fontsize=11,
            color="#333333",
        )

    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(values) * 1.22)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300)

    print(f"Selected epoch: {row['epoch']}")
    print(f"Epoch wall time: {epoch_ms:.3f} ms")
    print(f"Accumulated hook work: {work_ms:.3f} ms")
    print(f"Accumulated exposed tail: {tail_ms:.3f} ms")
    print(f"Approx overlapped hook work: {overlapped_work_ms:.3f} ms")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()