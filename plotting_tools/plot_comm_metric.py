#!/usr/bin/env python3
"""
plot_comm_metric.py

Standalone plotting script for communication metrics.

Supports:
- bar plots like your current time plots
- horizontal accounting plots like the paper figure

Input:
    parsed_comm.csv   (from exp_parser_comm.py)

Examples:
    python plot_comm_metric.py parsed_comm.csv --mode strong --scope all --metric hook_mean --kind bar
    python plot_comm_metric.py parsed_comm.csv --mode weak   --scope all --metric tail_mean --kind bar
    python plot_comm_metric.py parsed_comm.csv --mode strong --scope fixed --gpus 8 --metric tail_pct --kind bar
    python plot_comm_metric.py parsed_comm.csv --mode strong --scope fixed --gpus 8 --kind accounting
    python plot_comm_metric.py parsed_comm.csv --mode weak --scope all --kind accounting --method "RD+ZFP online (rate:8)"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- method order ------------------------------------------------------------

METHOD_ORDER = [
    "Baseline",
    "Ring",
    "Recursive doubling",
    "Ring+ZFP naive (rate:16)",
    "RD+ZFP naive (rate:16)",
    "Ring+ZFP online (rate:16)",
    "RD+ZFP online (rate:16)",
    "Ring+ZFP online (rate:10)",
    "Ring+ZFP online (rate:8)",
    "RD+ZFP online (rate:8)",
]

# --- colors: same families, greens/blues/oranges -----------------------------

COLORS = {
    "Baseline": "#9e9e9e",

    # greens
    "Ring": "#2e7d32",
    "Ring+ZFP naive (rate:16)": "#66bb6a",
    "Ring+ZFP online (rate:16)": "#388e3c",
    "Ring+ZFP online (rate:10)": "#1b5e20",
    "Ring+ZFP online (rate:8)": "#1b5e20",

    # blues
    "Recursive doubling": "#1565c0",
    "RD+ZFP naive (rate:16)": "#64b5f6",
    "RD+ZFP online (rate:16)": "#1976d2",
    "RD+ZFP online (rate:8)": "#0d47a1",

    # oranges, kept available if you want to switch family emphasis
    # "Some orange family 1": "#ef6c00",
    # "Some orange family 2": "#ff9800",
    # "Some orange family 3": "#e65100",
}

def metric_spec(metric: str):
    if metric == "hook_mean":
        return "hook_work_mean_ms", "Mean hook work time (ms)"
    if metric == "hook_total":
        return "hook_work_total_ms", "Accumulated hook work per epoch (ms)"
    if metric == "tail_mean":
        return "tail_mean_ms", "Mean exposed tail time (ms)"
    if metric == "tail_total":
        return "tail_total_ms", "Accumulated exposed tail per epoch (ms)"
    if metric == "tail_pct":
        return "tail_pct_of_train_epoch", "Exposed tail / train epoch time (%)"
    if metric == "tail_hook_pct":
        return "tail_pct_of_hook", "Exposed tail / hook work (%)"
    raise ValueError(metric)

def backend_label(df: pd.DataFrame) -> Optional[str]:
    """
    Resolve a single clean backend label (e.g. "NCCL", "RCCL") from the parsed CSV.
    Returns None if the column is missing, empty, or mixes multiple backends.
    Call this on the raw CSV before aggregation, since aggregation drops the
    non-numeric 'backend' column.
    """
    if "backend" not in df.columns:
        return None
    labels = []
    for v in df["backend"].dropna().tolist():
        s = str(v).strip().lower()
        if not s:
            continue
        if "rccl" in s:
            lab = "RCCL"
        elif "nccl" in s:
            lab = "NCCL"
        elif "gloo" in s:
            lab = "Gloo"
        elif "mpi" in s:
            lab = "MPI"
        else:
            lab = s.upper()
        if lab not in labels:
            labels.append(lab)
    return labels[0] if len(labels) == 1 else None

def title_with_backend(title: str, backend: Optional[str]) -> str:
    return f"{title} — {backend}" if backend else title

def infer_scaling(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    strong: fixed global batch -> for each method, choose the most common global_batch
    weak:   fixed local batch  -> for each method, choose the most common batch_per_rank
    """
    if mode == "strong":
        if "global_batch" not in df.columns:
            return df
        gb = df["global_batch"].dropna().mode()
        if len(gb):
            return df[df["global_batch"] == gb.iloc[0]].copy()
        return df

    if mode == "weak":
        if "batch_per_rank" not in df.columns:
            return df
        bs = df["batch_per_rank"].dropna().mode()
        if len(bs):
            return df[df["batch_per_rank"] == bs.iloc[0]].copy()
        return df

    raise ValueError(mode)

def aggregate_epochs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use epochs 2..N if available, else all.
    Aggregate by median over epochs for each (method, ranks).
    """
    if df.empty:
        return df

    if (df["epoch"] > 1).any():
        df = df[df["epoch"] > 1].copy()

    num_cols = df.select_dtypes(include=[np.number]).columns
    group_cols = ["method", "ranks"]
    agg = df.groupby(group_cols, as_index=False)[num_cols].median(numeric_only=True)
    return agg

def filter_scope(df: pd.DataFrame, scope: str, gpus: Optional[int]) -> pd.DataFrame:
    if scope == "all":
        return df
    if scope == "fixed":
        if gpus is None:
            raise SystemExit("--gpus is required when --scope fixed")
        return df[df["ranks"] == gpus].copy()
    raise ValueError(scope)

def ordered_methods_present(df: pd.DataFrame) -> List[str]:
    present = set(df["method"].dropna().tolist())
    ordered = [m for m in METHOD_ORDER if m in present]
    leftovers = sorted(present - set(ordered))
    return ordered + leftovers

def _config_signature(df: pd.DataFrame):
    """
    Robust per-config identity: (ranks, effective_global_batch), where
    effective_global_batch = global_batch if recorded else ranks * batch_per_rank,
    normalized to plain ints. Model name is excluded so .log/.out capitalization
    differences don't split a config.
    """
    def rk(v):
        try:
            return int(round(float(v))) if pd.notna(v) else None
        except Exception:
            return None

    n = len(df)
    ranks = df["ranks"] if "ranks" in df.columns else pd.Series([None] * n, index=df.index)
    gb = df["global_batch"] if "global_batch" in df.columns else pd.Series([np.nan] * n, index=df.index)
    bpr = df["batch_per_rank"] if "batch_per_rank" in df.columns else pd.Series([np.nan] * n, index=df.index)

    rk_sig, gb_sig = [], []
    for r, g, b in zip(ranks, gb, bpr):
        ri = rk(r)
        eff = rk(g)
        if eff is None and ri is not None:
            bi = rk(b)
            eff = ri * bi if bi is not None else None
        rk_sig.append(ri)
        gb_sig.append(eff)
    return pd.Series(rk_sig, index=df.index), pd.Series(gb_sig, index=df.index)

def prefer_online_rate8(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ring+ZFP online was re-run from rate 10 to rate 8. Per configuration, if a
    rate-8 run exists, use it and drop the stale rate-10 run for that config.
    Configs with only rate 10 are left untouched; if nothing was re-run at
    rate 8, the data is unchanged.
    """
    if "method" not in df.columns:
        return df

    old, new = "Ring+ZFP online (rate:10)", "Ring+ZFP online (rate:8)"
    if new not in set(df["method"].dropna()):
        return df

    df = df.copy()
    df["_rk_sig"], df["_gb_sig"] = _config_signature(df)

    drop_idx = []
    for _, g in df.groupby(["_rk_sig", "_gb_sig"], dropna=False):
        methods_here = set(g["method"].dropna())
        if new in methods_here and old in methods_here:
            drop_idx.extend(g.index[g["method"] == old].tolist())

    df = df.drop(index=drop_idx) if drop_idx else df
    return df.drop(columns=["_rk_sig", "_gb_sig"])

def plot_bar_all(df: pd.DataFrame, value_col: str, ylabel: str, mode: str, out: Path,
                 backend: Optional[str] = None):
    methods = ordered_methods_present(df)
    gpu_counts = sorted(df["ranks"].dropna().unique())

    x = np.arange(len(gpu_counts))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("ranks")
        vals = [sub.loc[g, value_col] if g in sub.index else np.nan for g in gpu_counts]
        ax.bar(
            x + (i - (len(methods) - 1) / 2) * width,
            vals,
            width,
            label=method,
            color=COLORS.get(method, None),
            edgecolor="black",
            linewidth=0.7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(g)) for g in gpu_counts])
    ax.set_xlabel("GPUs")
    ax.set_ylabel(ylabel)
    ax.set_title(title_with_backend(f"{ylabel} — {mode.capitalize()} scaling", backend))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

def plot_bar_fixed(df: pd.DataFrame, value_col: str, ylabel: str, mode: str, gpus: int, out: Path,
                   backend: Optional[str] = None):
    methods = ordered_methods_present(df)
    sub = df.copy()
    sub["method"] = pd.Categorical(sub["method"], categories=methods, ordered=True)
    sub = sub.sort_values("method")

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        sub["method"],
        sub[value_col],
        color=[COLORS.get(m, "#888888") for m in sub["method"]],
        edgecolor="black",
        linewidth=0.8,
    )

    for rect, val in zip(bars, sub[value_col]):
        if pd.notna(val):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Communication Hook")
    ax.set_title(title_with_backend(f"{ylabel} — {mode.capitalize()} scaling — {gpus} GPUs", backend))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

def plot_accounting_by_method(df: pd.DataFrame, method: str, mode: str, out: Path,
                              backend: Optional[str] = None):
    """
    Same method, compare GPU counts.
    Horizontal bars:
    - full train epoch
    - accumulated hook work
    - exposed tail
    """
    sub = df[df["method"] == method].copy()
    if sub.empty:
        raise SystemExit(f"No rows found for method: {method}")

    sub = sub.sort_values("ranks")
    labels = [f"{int(r)} GPUs" for r in sub["ranks"]]

    epoch_ms = sub["train_only_epoch_s"] * 1000.0
    hook_ms = sub["hook_work_total_ms"]
    tail_ms = sub["tail_total_ms"]

    y = np.arange(len(sub))
    h = 0.22

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.barh(y + h, epoch_ms, height=h, color="#d9d9d9", edgecolor="black", label="Train epoch")
    ax.barh(y, hook_ms, height=h, color="#90caf9", edgecolor="black", label="Accumulated hook work")
    ax.barh(y - h, tail_ms, height=h, color="#ef9a9a", edgecolor="black", label="Exposed tail")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (ms)")
    ax.set_title(title_with_backend(f"Communication accounting — {method} — {mode.capitalize()} scaling", backend))
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

def plot_accounting_fixed_gpus(df: pd.DataFrame, gpus: int, mode: str, out: Path,
                               backend: Optional[str] = None):
    """
    Same GPU count, compare methods.
    Horizontal bars:
    - full train epoch
    - accumulated hook work
    - exposed tail
    """
    sub = df.copy()
    methods = ordered_methods_present(sub)
    sub["method"] = pd.Categorical(sub["method"], categories=methods, ordered=True)
    sub = sub.sort_values("method")

    labels = sub["method"].tolist()
    epoch_ms = sub["train_only_epoch_s"] * 1000.0
    hook_ms = sub["hook_work_total_ms"]
    tail_ms = sub["tail_total_ms"]

    y = np.arange(len(sub))
    h = 0.22

    fig_h = max(5, 0.45 * len(sub) + 2)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    ax.barh(y + h, epoch_ms, height=h, color="#d9d9d9", edgecolor="black", label="Train epoch")
    ax.barh(y, hook_ms, height=h, color="#a5d6a7", edgecolor="black", label="Accumulated hook work")
    ax.barh(y - h, tail_ms, height=h, color="#ffcc80", edgecolor="black", label="Exposed tail")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (ms)")
    ax.set_title(title_with_backend(f"Communication accounting — {gpus} GPUs — {mode.capitalize()} scaling", backend))
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, help="parsed_comm.csv from exp_parser_comm.py")
    ap.add_argument("--mode", choices=["strong", "weak"], required=True)
    ap.add_argument("--scope", choices=["all", "fixed"], required=True)
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--metric", choices=[
        "hook_mean", "hook_total", "tail_mean", "tail_total", "tail_pct", "tail_hook_pct"
    ], default="hook_mean")
    ap.add_argument("--kind", choices=["bar", "accounting"], required=True)
    ap.add_argument("--method", type=str, default=None,
                    help="For accounting/all: compare GPU counts for one method")
    ap.add_argument("--backend", type=str, default=None,
                    help="Override backend label in title (e.g. NCCL, RCCL). "
                         "Auto-detected from the CSV's backend column if omitted.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Resolve the backend before aggregation, which drops the non-numeric column.
    backend = args.backend or backend_label(df)
    df = prefer_online_rate8(df)
    df = infer_scaling(df, args.mode)
    df = aggregate_epochs(df)
    df = filter_scope(df, args.scope, args.gpus)

    if df.empty:
        raise SystemExit("No data left after filtering")

    if args.out is None:
        tag = f"{args.mode}_{args.scope}"
        if args.scope == "fixed":
            tag += f"_g{args.gpus}"
        if args.kind == "bar":
            args.out = Path(f"comm_{args.metric}_{tag}.png")
        else:
            if args.method:
                safe_m = args.method.replace(" ", "_").replace("/", "_")
                args.out = Path(f"comm_accounting_{tag}_{safe_m}.png")
            else:
                args.out = Path(f"comm_accounting_{tag}.png")

    if args.kind == "bar":
        value_col, ylabel = metric_spec(args.metric)
        if args.scope == "all":
            plot_bar_all(df, value_col, ylabel, args.mode, args.out, backend=backend)
        else:
            plot_bar_fixed(df, value_col, ylabel, args.mode, args.gpus, args.out, backend=backend)
        return

    # accounting
    if args.scope == "fixed":
        plot_accounting_fixed_gpus(df, args.gpus, args.mode, args.out, backend=backend)
    else:
        if not args.method:
            raise SystemExit("--method is required for --kind accounting when --scope all")
        plot_accounting_by_method(df, args.method, args.mode, args.out, backend=backend)

if __name__ == "__main__":
    main()