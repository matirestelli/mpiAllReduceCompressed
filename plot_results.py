#!/usr/bin/env python3
"""
Compare training runs from results/*.csv files.

Each CSV was produced by one training run (one backend + algorithm combo).
All CSVs found in results/ are overlaid on every plot so you can directly
compare ring vs builtin, mpi vs nccl, etc.

Usage:
    python plot_results.py                        # all CSVs in results/
    python plot_results.py results/run1.csv ...   # specific files
"""

import sys
import os
import glob
import csv

import matplotlib
matplotlib.use('Agg')   # no display needed — works on cluster nodes
import matplotlib.pyplot as plt
import numpy as np


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_runs(paths: list[str]) -> dict:
    """Return {label: {metric: [values]}} for each CSV."""
    runs = {}
    label_counts: dict[str, int] = {}

    for path in sorted(paths):
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        r0     = rows[0]
        algo   = r0.get('algorithm', 'unknown')
        backend = r0.get('backend', 'unknown')
        base_label = f"{algo}/{backend}"

        # Deduplicate labels if the same algo/backend ran multiple times
        n = label_counts.get(base_label, 0)
        label_counts[base_label] = n + 1
        label = base_label if n == 0 else f"{base_label} ({n+1})"

        runs[label] = {
            'epochs':       [int(r['epoch'])          for r in rows],
            'lr':           [float(r['lr'])            for r in rows],
            'train_loss':   [float(r['train_loss'])    for r in rows],
            'train_acc':    [float(r['train_acc'])     for r in rows],
            'val_loss':     [float(r['val_loss'])      for r in rows],
            'val_acc':      [float(r['val_acc'])       for r in rows],
            'epoch_time_s': [float(r['epoch_time_s'])  for r in rows],
        }

    return runs


def _colors(n: int):
    cmap = plt.get_cmap('tab10')
    return [cmap(i % 10) for i in range(n)]


def _save(fig, name: str, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  saved → {path}")
    plt.close(fig)


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_val_accuracy(runs: dict, colors, outdir: str) -> None:
    """Validation accuracy over epochs for every run."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for (label, data), color in zip(runs.items(), colors):
        ax.plot(data['epochs'], data['val_acc'],
                label=label, color=color, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Accuracy (%)')
    ax.set_title('Validation Accuracy — all runs')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    _save(fig, 'fig1_val_accuracy.png', outdir)


def fig_losses(runs: dict, colors, outdir: str) -> None:
    """Train and val loss side-by-side for every run."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for (label, data), color in zip(runs.items(), colors):
        ax1.plot(data['epochs'], data['train_loss'],
                 label=label, color=color, linewidth=2)
        ax2.plot(data['epochs'], data['val_loss'],
                 label=label, color=color, linewidth=2)
    for ax, title in [(ax1, 'Train Loss'), (ax2, 'Val Loss')]:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{title} — all runs')
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, 'fig2_losses.png', outdir)


def fig_epoch_time(runs: dict, colors, outdir: str) -> None:
    """Epoch wall-time over training for every run.

    Epoch 1 is always inflated (cuDNN autotuning) — annotated on the plot.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for (label, data), color in zip(runs.items(), colors):
        ax.plot(data['epochs'], data['epoch_time_s'],
                label=label, color=color, linewidth=2)
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(1.2, ax.get_ylim()[0] if ax.get_ylim()[0] else 0,
            'epoch 1\n(cuDNN\nwarmup)', fontsize=8, color='gray', va='bottom')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time (s)')
    ax.set_title('Epoch Wall-Time — all runs\n(epoch 1 includes cuDNN autotuning overhead)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, 'fig3_epoch_time.png', outdir)


def fig_summary(runs: dict, colors, outdir: str) -> None:
    """Bar chart: final val accuracy and mean epoch time (excl. epoch 1)."""
    labels      = list(runs.keys())
    final_accs  = [data['val_acc'][-1]                           for data in runs.values()]
    mean_times  = [np.mean(data['epoch_time_s'][1:] or data['epoch_time_s'])
                   for data in runs.values()]

    x     = np.arange(len(labels))
    width = 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, len(labels) * 3), 5))

    bars1 = ax1.bar(x, final_accs, width, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax1.set_ylabel('Val Accuracy (%)')
    ax1.set_title('Final Validation Accuracy')
    ax1.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars1, final_accs):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.2,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    bars2 = ax2.bar(x, mean_times, width, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax2.set_ylabel('Mean Epoch Time (s)')
    ax2.set_title('Mean Epoch Time\n(epochs 2+, excl. cuDNN warmup)')
    ax2.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars2, mean_times):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.2,
                 f'{val:.1f}s', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Run Comparison Summary', fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, 'fig4_summary.png', outdir)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    paths = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob('results/*.csv'))
    if not paths:
        print("No CSV files found. Pass paths as arguments or run training first.")
        return

    runs = load_runs(paths)
    if not runs:
        print("No valid run data loaded.")
        return

    print(f"Loaded {len(runs)} run(s): {', '.join(runs.keys())}")

    outdir = 'results/plots'
    colors = _colors(len(runs))

    fig_val_accuracy(runs, colors, outdir)
    fig_losses(runs, colors, outdir)
    fig_epoch_time(runs, colors, outdir)
    fig_summary(runs, colors, outdir)

    print(f"\nAll plots saved to {outdir}/")


if __name__ == '__main__':
    main()
