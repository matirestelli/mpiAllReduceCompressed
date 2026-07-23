#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from exp_parser import load_experiment_folder, ensure_plot_dir

METHOD_ORDER = [
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

METHOD_COLORS = {
    "Baseline": "#4d4d4d",
    "Ring": "#007336",
    "Ring+ZFP naive (rate:16)": "#82ba36",
    "Ring+ZFP online (rate:16)": "#6eaa2f",
    "Ring+ZFP online (rate:10)": "#9bcb59",
    "Recursive doubling": "#005a2a",
    "RD+ZFP naive (rate:16)": "#a8d46f",
    "RD+ZFP online (rate:16)": "#5f9624",
    "RD+ZFP online (rate:8)": "#c0df8f",
}

def apply_style():
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.dpi": 300,
    })

def ordered_methods(values):
    present = list(values)
    ordered = [m for m in METHOD_ORDER if m in present]
    leftovers = sorted([m for m in present if m not in METHOD_ORDER])
    return ordered + leftovers

def metric_spec(mode: str, metric: str):
    if metric == "time":
        if mode == "strong":
            return "train_only_epoch_s", "Training time / epoch (s)"
        return "t_iter_median_ms", "Median iteration time (ms)"
    if metric == "hook":
        return "hook_work_mean_ms", "Mean hook work time (ms)"
    if metric == "tail":
        return "tail_mean_ms", "Mean exposed tail time (ms)"
    raise ValueError(f"Unsupported metric: {metric}")

def filter_df(df, mode, global_batch=None, batch_per_rank=None, gpus=None):
    if mode == "strong":
        if global_batch is not None:
            df = df[df["global_batch"] == global_batch]
    elif mode == "weak":
        if batch_per_rank is not None:
            df = df[df["batch_per_rank"] == batch_per_rank]

    if gpus is not None:
        df = df[df["ranks"] == gpus]

    return df

def annotate_bars(ax, fontsize=9):
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h) and h > 0:
            ax.annotate(
                f"{h:.2f}",
                (p.get_x() + p.get_width() / 2, h),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                xytext=(0, 3),
                textcoords="offset points",
            )

def plot_fixed(df, value_col, ylabel, title, output):
    agg = df.groupby("method", as_index=False)[value_col].median()
    methods = ordered_methods(agg["method"].unique())
    agg = agg.set_index("method").loc[methods].reset_index()

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    sns.barplot(
        data=agg,
        x="method",
        y=value_col,
        order=methods,
        palette=[METHOD_COLORS.get(m, "#888888") for m in methods],
        ax=ax,
    )

    ax.set_title(title, fontsize=20, pad=8)
    ax.set_xlabel("Communication method", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=17)
    ax.tick_params(axis="x", rotation=25, labelsize=11)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="y", alpha=0.25)
    annotate_bars(ax, fontsize=9)

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {output}")

def plot_all(df, value_col, ylabel, title, output):
    agg = df.groupby(["ranks", "method"], as_index=False)[value_col].median()
    gpu_list = sorted(agg["ranks"].unique())
    methods = ordered_methods(agg["method"].unique())

    x = np.arange(len(gpu_list))
    width = min(0.80 / max(len(methods), 1), 0.14)

    fig, ax = plt.subplots(figsize=(9.4, 5.8))

    for i, method in enumerate(methods):
        vals = []
        for g in gpu_list:
            sub = agg[(agg["ranks"] == g) & (agg["method"] == method)]
            vals.append(sub[value_col].iloc[0] if not sub.empty else np.nan)

        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            label=method,
            color=METHOD_COLORS.get(method, "#999999"),
            edgecolor="black",
            linewidth=0.35,
        )

    ax.set_title(title, fontsize=20, pad=8)
    ax.set_xlabel("GPUs", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=17)
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in gpu_list], fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {output}")

def make_default_title(root: Path, mode: str, scope: str, metric: str, gpus=None):
    model_name = root.parts[-3] if len(root.parts) >= 3 else root.name
    scaling_name = "Strong scaling" if mode == "strong" else "Weak scaling"

    if metric == "time":
        metric_name = "epoch time" if mode == "strong" else "iteration time"
    elif metric == "hook":
        metric_name = "hook work"
    else:
        metric_name = "tail"

    if scope == "fixed" and gpus is not None:
        return f"{model_name} - {gpus} GPUs - {scaling_name} - {metric_name}"
    return f"{model_name} - {scaling_name} - {metric_name}"

def main():
    ap = argparse.ArgumentParser(description="Plot experiment metrics with --mode and --scope.")
    ap.add_argument("root", help="e.g. experiments_frontier/wideresnet/cifar10/strongScaling")
    ap.add_argument("--mode", choices=["strong", "weak"], required=True)
    ap.add_argument("--scope", choices=["fixed", "all"], required=True)
    ap.add_argument("--metric", choices=["time", "hook", "tail"], required=True)

    ap.add_argument("--gpus", type=int, default=None, help="Required for --scope fixed")
    ap.add_argument("--global-batch", type=int, default=None, help="Used typically for strong scaling")
    ap.add_argument("--batch-per-rank", type=int, default=None, help="Used typically for weak scaling")
    ap.add_argument("--title", default=None)
    ap.add_argument("--png", action="store_true", help="Also save PNG besides PDF")
    ap.add_argument("--csv", action="store_true", help="Save filtered CSV used for plotting")
    ap.add_argument("--out", default=None)

    args = ap.parse_args()

    if args.scope == "fixed" and args.gpus is None:
        raise SystemExit("--gpus is required when --scope fixed")

    apply_style()

    root = Path(args.root)
    outdir = ensure_plot_dir(root)

    df = load_experiment_folder(root)
    if df.empty:
        raise SystemExit(f"No data found in {root}")

    value_col, ylabel = metric_spec(args.mode, args.metric)
    df = filter_df(
        df,
        mode=args.mode,
        global_batch=args.global_batch,
        batch_per_rank=args.batch_per_rank,
        gpus=args.gpus if args.scope == "fixed" else None,
    )
    df = df.dropna(subset=["method", "ranks", value_col])

    if df.empty:
        raise SystemExit("No rows left after filtering.")

    title = args.title or make_default_title(root, args.mode, args.scope, args.metric, args.gpus)

    if args.out:
        out_pdf = Path(args.out)
    else:
        if args.scope == "fixed":
            out_pdf = outdir / f"{args.mode}_fixed_{args.gpus}gpu_{args.metric}.pdf"
        else:
            out_pdf = outdir / f"{args.mode}_allgpus_{args.metric}.pdf"

    if args.scope == "fixed":
        plot_fixed(df, value_col, ylabel, title, out_pdf)
    else:
        plot_all(df, value_col, ylabel, title, out_pdf)

    if args.png:
        out_png = out_pdf.with_suffix(".png")
        if args.scope == "fixed":
            plot_fixed(df, value_col, ylabel, title, out_png)
        else:
            plot_all(df, value_col, ylabel, title, out_png)

    if args.csv:
        out_csv = out_pdf.with_suffix(".csv")
        df.to_csv(out_csv, index=False)
        print(f"[INFO] wrote {out_csv}")

if __name__ == "__main__":
    main()
