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
    "Baseline": "#6b6b6b",

    "Ring": "#238b45",
    "Ring+ZFP naive (rate:16)": "#41ae76",
    "Ring+ZFP online (rate:16)": "#74c476",
    "Ring+ZFP online (rate:10)": "#c7e9c0",

    "Recursive doubling": "#08519c",
    "RD+ZFP naive (rate:16)": "#3182bd",
    "RD+ZFP online (rate:16)": "#6baed6",
    "RD+ZFP online (rate:8)": "#bdd7e7",
}

# oranges block
# METHOD_COLORS = {
#     "Baseline": "#6b6b6b",
#
#     "Ring": "#006d2c",
#     "Ring+ZFP naive (rate:16)": "#31a354",
#     "Ring+ZFP online (rate:16)": "#74c476",
#     "Ring+ZFP online (rate:10)": "#bae4b3",
#
#     "Recursive doubling": "#d94801",
#     "RD+ZFP naive (rate:16)": "#f16913",
#     "RD+ZFP online (rate:16)": "#fdae6b",
#     "RD+ZFP online (rate:8)": "#fdd0a2",
# }

METHOD_GROUP = {
    "Baseline": "baseline",

    "Ring": "ring",
    "Ring+ZFP naive (rate:16)": "ring",
    "Ring+ZFP online (rate:16)": "ring",
    "Ring+ZFP online (rate:10)": "ring",

    "Recursive doubling": "rd",
    "RD+ZFP naive (rate:16)": "rd",
    "RD+ZFP online (rate:16)": "rd",
    "RD+ZFP online (rate:8)": "rd",
}

def apply_style():
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.grid": False,
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

def prepare_batch_columns(df):
    df = df.copy()

    if "batch_per_rank" not in df.columns:
        df["batch_per_rank"] = np.nan

    if "global_batch" in df.columns and "ranks" in df.columns:
        can_infer = df["global_batch"].notna() & df["ranks"].notna() & (df["ranks"] != 0)
        inferred = df["global_batch"] / df["ranks"]
        inferred = inferred.where(can_infer, np.nan)

        if df["batch_per_rank"].isna().any():
            df.loc[df["batch_per_rank"].isna(), "batch_per_rank"] = inferred[df["batch_per_rank"].isna()]

    # If values are numerically integral, store them as ints for cleaner tick labels/titles
    for col in ["global_batch", "batch_per_rank"]:
        if col in df.columns:
            vals = df[col].dropna()
            if not vals.empty and np.all(np.isclose(vals, np.round(vals))):
                df[col] = df[col].round().astype("Int64")

    return df

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

def unique_non_null(values):
    vals = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
    out = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out

def single_unique_or_none(series):
    vals = unique_non_null(series.tolist())
    if len(vals) == 1:
        return vals[0]
    return None

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

def family_offsets():
    width = 0.06
    offset_map = {
        "Baseline": -0.30,

        # Ring family: bars touch each other
        "Ring": -0.15,
        "Ring+ZFP naive (rate:16)": -0.09,
        "Ring+ZFP online (rate:16)": -0.03,
        "Ring+ZFP online (rate:10)": 0.03,

        # RD family: bars touch each other
        "Recursive doubling": 0.15,
        "RD+ZFP naive (rate:16)": 0.21,
        "RD+ZFP online (rate:16)": 0.27,
        "RD+ZFP online (rate:8)": 0.33,
    }
    return width, offset_map

def plot_fixed(df, mode, value_col, ylabel, title, output, ymax=None):
    x_col = "global_batch" if mode == "strong" else "batch_per_rank"
    x_label = "Global batch" if mode == "strong" else "Batch per rank"

    if x_col not in df.columns:
        raise SystemExit(f"Column {x_col} not available for fixed plot.")

    df = df.dropna(subset=[x_col])
    if df.empty:
        raise SystemExit(f"No rows with valid {x_col} for fixed plot.")

    agg = df.groupby([x_col, "method"], as_index=False)[value_col].median()
    x_values = sorted(agg[x_col].dropna().unique())
    methods = ordered_methods(agg["method"].unique())

    x = np.arange(len(x_values))
    width, offset_map = family_offsets()

    fig, ax = plt.subplots(figsize=(9.4, 5.8))

    for method in methods:
        vals = []
        for xv in x_values:
            sub = agg[(agg[x_col] == xv) & (agg["method"] == method)]
            vals.append(sub[value_col].iloc[0] if not sub.empty else np.nan)

        ax.bar(
            x + offset_map.get(method, 0.0),
            vals,
            width=width,
            label=method,
            color=METHOD_COLORS.get(method, "#999999"),
            edgecolor="black",
            linewidth=0.35,
        )

    ax.set_title(title, fontsize=20, pad=12, loc="center")
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=17)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x_values], fontsize=13)
    ax.tick_params(axis="y", labelsize=13)

    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    if ymax is not None:
        ax.set_ylim(0, ymax)

    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    legend_order = [m for m in METHOD_ORDER if m in label_to_handle]
    leftovers = [l for l in labels if l not in legend_order]
    legend_order.extend(leftovers)

    ax.legend(
        [label_to_handle[m] for m in legend_order],
        legend_order,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        frameon=False,
        fontsize=10,
        columnspacing=1.4,
        handletextpad=0.5,
    )

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {output}")

def plot_all(df, value_col, ylabel, title, output, ymax=None):
    agg = df.groupby(["ranks", "method"], as_index=False)[value_col].median()
    gpu_list = sorted(agg["ranks"].unique())
    methods = ordered_methods(agg["method"].unique())

    x = np.arange(len(gpu_list))
    width, offset_map = family_offsets()

    fig, ax = plt.subplots(figsize=(9.4, 5.8))

    for method in methods:
        vals = []
        for g in gpu_list:
            sub = agg[(agg["ranks"] == g) & (agg["method"] == method)]
            vals.append(sub[value_col].iloc[0] if not sub.empty else np.nan)

        ax.bar(
            x + offset_map.get(method, 0.0),
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

    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    if ymax is not None:
        ax.set_ylim(0, ymax)

    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    legend_order = [m for m in METHOD_ORDER if m in label_to_handle]
    leftovers = [l for l in labels if l not in legend_order]
    legend_order.extend(leftovers)

    ax.legend(
        [label_to_handle[m] for m in legend_order],
        legend_order,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        frameon=False,
        fontsize=10,
        columnspacing=1.4,
        handletextpad=0.5,
    )

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {output}")

def make_default_title(
    root: Path,
    mode: str,
    scope: str,
    gpus=None,
    global_batch=None,
    batch_per_rank=None,
    varying_batches=False,
):
    model_name = root.parts[-3] if len(root.parts) >= 3 else root.name
    scaling_name = "Strong Scaling" if mode == "strong" else "Weak Scaling"

    model_map = {
        "wideresnet": "WideResNet",
        "resnet50": "ResNet-50",
        "vit": "ViT",
    }
    model_name = model_map.get(model_name.lower(), model_name)

    first_line_parts = [f"{model_name} on Frontier (AMD)"]
    if scope == "fixed" and gpus is not None:
        first_line_parts.append(f"{gpus} GPUs")
    first_line_parts.append(scaling_name)

    first_line = " - ".join(first_line_parts)

    second_line = None
    if mode == "strong":
        if varying_batches:
            second_line = "Global batch: varying"
        elif global_batch is not None:
            second_line = f"Global batch {global_batch}"
    else:
        if varying_batches:
            second_line = "Local batch: varying"
        elif batch_per_rank is not None:
            second_line = f"Local batch {batch_per_rank}"

    if second_line is not None:
        return f"{first_line}\n{second_line}"
    return first_line


def validate_for_scope(df, mode, scope):
    if scope == "all":
        if mode == "strong":
            gb = single_unique_or_none(df["global_batch"]) if "global_batch" in df.columns else None
            if gb is None:
                vals = unique_non_null(df["global_batch"].tolist()) if "global_batch" in df.columns else []
                raise SystemExit(
                    "For --scope all --mode strong, data must contain exactly one global_batch "
                    f"after filtering. Found: {vals}"
                )
            return {"global_batch": gb, "batch_per_rank": None, "varying_batches": False}

        bpr = single_unique_or_none(df["batch_per_rank"]) if "batch_per_rank" in df.columns else None
        if bpr is None:
            vals = unique_non_null(df["batch_per_rank"].tolist()) if "batch_per_rank" in df.columns else []
            raise SystemExit(
                "For --scope all --mode weak, data must contain exactly one batch_per_rank "
                f"after filtering/inference. Found: {vals}"
            )
        return {"global_batch": None, "batch_per_rank": bpr, "varying_batches": False}

    # scope == fixed
    if mode == "strong":
        vals = unique_non_null(df["global_batch"].tolist()) if "global_batch" in df.columns else []
        if len(vals) == 0:
            raise SystemExit("For --scope fixed --mode strong, no global_batch values were found.")
        return {
            "global_batch": vals[0] if len(vals) == 1 else None,
            "batch_per_rank": None,
            "varying_batches": len(vals) > 1,
        }

    vals = unique_non_null(df["batch_per_rank"].tolist()) if "batch_per_rank" in df.columns else []
    if len(vals) == 0:
        raise SystemExit("For --scope fixed --mode weak, no batch_per_rank values were found.")
    return {
        "global_batch": None,
        "batch_per_rank": vals[0] if len(vals) == 1 else None,
        "varying_batches": len(vals) > 1,
    }

def main():
    ap = argparse.ArgumentParser(description="Plot experiment metrics with --mode and --scope.")
    ap.add_argument("root", help="e.g. experiments_frontier/wideresnet/cifar10/strongScaling")
    ap.add_argument("--mode", choices=["strong", "weak"], required=True)
    ap.add_argument("--scope", choices=["fixed", "all"], required=True)
    ap.add_argument("--metric", choices=["time", "hook", "tail"], required=True)

    ap.add_argument("--gpus", type=int, default=None, help="Required for --scope fixed")
    ap.add_argument("--global-batch", type=int, default=None, help="Optional strong-scaling filter")
    ap.add_argument("--batch-per-rank", type=int, default=None, help="Optional weak-scaling filter")
    ap.add_argument("--title", default=None)
    ap.add_argument("--png", action="store_true", help="Also save PNG besides PDF")
    ap.add_argument("--csv", action="store_true", help="Save filtered CSV used for plotting")
    ap.add_argument("--ymax", type=float, default=None, help="Fixed upper y-axis limit")
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

    df = prepare_batch_columns(df)

    value_col, ylabel = metric_spec(args.mode, args.metric)
    df = filter_df(
        df,
        mode=args.mode,
        global_batch=args.global_batch,
        batch_per_rank=args.batch_per_rank,
        gpus=args.gpus if args.scope == "fixed" else None,
    )

    needed = ["method", "ranks", value_col]
    if args.mode == "strong":
        needed.append("global_batch")
    else:
        needed.append("batch_per_rank")

    df = df.dropna(subset=needed)

    if df.empty:
        raise SystemExit("No rows left after filtering.")

    meta = validate_for_scope(df, args.mode, args.scope)

    title = args.title or make_default_title(
        root,
        args.mode,
        args.scope,
        gpus=args.gpus,
        global_batch=meta["global_batch"],
        batch_per_rank=meta["batch_per_rank"],
        varying_batches=meta["varying_batches"],
    )

    if args.out:
        out_pdf = Path(args.out)
    else:
        if args.scope == "fixed":
            out_pdf = outdir / f"{args.mode}_fixed_{args.gpus}gpu_{args.metric}.pdf"
        else:
            out_pdf = outdir / f"{args.mode}_allgpus_{args.metric}.pdf"

    if args.scope == "fixed":
        plot_fixed(df, args.mode, value_col, ylabel, title, out_pdf, ymax=args.ymax)
    else:
        plot_all(df, value_col, ylabel, title, out_pdf, ymax=args.ymax)

    if args.png:
        out_png = out_pdf.with_suffix(".png")
        if args.scope == "fixed":
            plot_fixed(df, args.mode, value_col, ylabel, title, out_png, ymax=args.ymax)
        else:
            plot_all(df, value_col, ylabel, title, out_png, ymax=args.ymax)

    if args.csv:
        out_csv = out_pdf.with_suffix(".csv")
        df.to_csv(out_csv, index=False)
        print(f"[INFO] wrote {out_csv}")

if __name__ == "__main__":
    main()
