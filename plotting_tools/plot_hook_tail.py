#!/usr/bin/env python3
"""
plot_hook_tail.py

Hook-work and exposed-tail plots straight from the experiment .out/.log files
(no results/*.csv needed). Bar layout, colors, family offsets, value labels and
title style match plot_metric.py. Baseline is intentionally excluded (it has no
meaningful communication-hook work).

Metric (per configuration = one run at a given GPU count / batch):
  per epoch, HOOK_WORK_SUMMARY.work_total_ms = total hook time in that epoch.
  The config value is the MEAN over epochs of that per-epoch total.
    --unit epoch : mean per-epoch hook total (ms)          [default for strong]
    --unit iter  : same, divided by steps/epoch -> per iter [default for weak]
    --unit call  : mean of work_mean_ms (per hook call)     [matches plot_metric --metric hook]
  Tail is treated per epoch and, in the tail plot, compared against the
  train-only epoch time so tail can never exceed it.

Hook plot mirrors plot_metric.py:
    --scope all   -> x-axis = GPU counts, grouped family bars (all GPUs, one figure)
    --scope fixed -> x-axis = global batch (strong) / batch per rank (weak), --gpus required
Tail plot: horizontal, exposed tail vs train-only epoch, one row per method.

Examples:
    python plot_hook_tail.py .../weakScaling   --mode weak   --scope all --kind hook --batch-per-rank 16
    python plot_hook_tail.py .../strongScaling --mode strong --scope all --kind hook --global-batch 128
    python plot_hook_tail.py .../strongScaling --mode strong --scope fixed --gpus 8 --kind hook --global-batch 128
    python plot_hook_tail.py .../strongScaling --mode strong --scope all --kind tail --global-batch 128
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import exp_parser as ep

# --------------------------------------------------------------------------- #
# Style / colors — copied from plot_metric.py (+ Ring online rate:8)           #
# --------------------------------------------------------------------------- #

METHOD_ORDER = [
    "Baseline",
    "Ring",
    "Ring+ZFP naive (rate:16)",
    "Ring+ZFP online (rate:16)",
    "Ring+ZFP online (rate:10)",
    "Ring+ZFP online (rate:8)",
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
    "Ring+ZFP online (rate:8)": "#c7e9c0",

    "Recursive doubling": "#08519c",
    "RD+ZFP naive (rate:16)": "#3182bd",
    "RD+ZFP online (rate:16)": "#6baed6",
    "RD+ZFP online (rate:8)": "#bdd7e7",
}

def family_offsets():
    width = 0.06
    offset_map = {
        "Baseline": -0.30,
        "Ring": -0.15,
        "Ring+ZFP naive (rate:16)": -0.09,
        "Ring+ZFP online (rate:16)": -0.03,
        "Ring+ZFP online (rate:10)": 0.03,
        "Ring+ZFP online (rate:8)": 0.03,
        "Recursive doubling": 0.15,
        "RD+ZFP naive (rate:16)": 0.21,
        "RD+ZFP online (rate:16)": 0.27,
        "RD+ZFP online (rate:8)": 0.33,
    }
    return width, offset_map

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

def annotate_bars(ax, fmt="{:.3f}", fontsize=9):
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h) and h > 0:
            ax.annotate(
                fmt.format(h),
                (p.get_x() + p.get_width() / 2, h),
                ha="center", va="bottom", fontsize=fontsize,
                xytext=(0, 3), textcoords="offset points",
            )

# --------------------------------------------------------------------------- #
# Backend label + rate-8 preference (self-contained)                          #
# --------------------------------------------------------------------------- #

def backend_label(df: pd.DataFrame) -> Optional[str]:
    if "backend" not in df.columns:
        return None
    labels = []
    for v in df["backend"].dropna().tolist():
        s = str(v).strip().lower()
        if not s:
            continue
        lab = {"rccl": "RCCL", "nccl": "NCCL", "gloo": "Gloo", "mpi": "MPI"}.get(s, s.upper())
        if lab not in labels:
            labels.append(lab)
    return labels[0] if len(labels) == 1 else None

def prefer_online_rate8(df: pd.DataFrame) -> pd.DataFrame:
    """Per config (ranks, effective global batch), keep Ring online rate 8 and
    drop stale rate 10 when both exist."""
    if "method" not in df.columns:
        return df
    old, new = "Ring+ZFP online (rate:10)", "Ring+ZFP online (rate:8)"
    if new not in set(df["method"].dropna()):
        return df

    def rk(v):
        try:
            return int(round(float(v))) if pd.notna(v) else None
        except Exception:
            return None

    df = df.copy()
    rks, gbs = [], []
    for _, r in df.iterrows():
        ri = rk(r.get("ranks"))
        eff = rk(r.get("global_batch"))
        if eff is None and ri is not None:
            bi = rk(r.get("batch_per_rank"))
            eff = ri * bi if bi is not None else None
        rks.append(ri); gbs.append(eff)
    df["_rk"], df["_gb"] = rks, gbs
    drop = []
    for _, g in df.groupby(["_rk", "_gb"], dropna=False):
        ms = set(g["method"].dropna())
        if new in ms and old in ms:
            drop += g.index[g["method"] == old].tolist()
    df = df.drop(index=drop) if drop else df
    return df.drop(columns=["_rk", "_gb"])

# --------------------------------------------------------------------------- #
# Parsing: one record per run, keeping per-epoch arrays                        #
# --------------------------------------------------------------------------- #

def _iter_run_blocks(text: str):
    """
    Yield (meta_dict_or_None, block_text) for each run in a file, trying several
    per-run delimiters so concatenated multi-run files split correctly even when
    the '=== Starting training:' banner is absent.
    """
    # 1) preferred: '=== Starting training: ... ===' banners (carry all fields)
    starts = list(ep.RE_STARTING_META.finditer(text))
    if starts:
        idxs = [m.start() for m in starts] + [len(text)]
        for i, m in enumerate(starts):
            yield m.groupdict(), text[idxs[i]:idxs[i + 1]]
        return

    # 2) fallback: '=== Finished training: ... ===' lines (also carry all fields);
    #    each block ends at its finished line.
    fins = list(ep.RE_FINISHED_META.finditer(text))
    if fins:
        prev = 0
        for m in fins:
            yield m.groupdict(), text[prev:m.end()]
            prev = m.end()
        tail = text[prev:]
        if ep.RE_WORK.search(tail) or ep.RE_TAIL.search(tail):
            yield None, tail
        return

    # 3) fallback: split on each 'Distributed Training:' header (one per run)
    heads = list(ep.RE_MODEL.finditer(text))
    if len(heads) > 1:
        idxs = [m.start() for m in heads] + [len(text)]
        for i in range(len(heads)):
            yield None, text[idxs[i]:idxs[i + 1]]
        return

    # 4) single run / single block
    yield None, text


_ZFP_IN_BLOCK = re.compile(r"zfp_rate\s*=\s*(\d+)|zfp\s*=\s*(\d+)")

def _zfp_from_block(block: str) -> Optional[str]:
    m = _ZFP_IN_BLOCK.search(block)
    if m:
        return m.group(1) or m.group(2)
    return None

def _method_from(meta, block):
    backend = hook = zfp = None
    if meta:
        backend = (meta.get("backend") or "").strip().lower() or None
        hook = meta.get("hook")
        z = (meta.get("zfp") or "").strip()
        zfp = None if z.lower() in ("none", "") else (ep.infer_zfp_rate(z) or z)
    mb = ep.RE_BACKEND_ALGO.search(block)
    if mb:
        backend = backend or mb.group(1).strip().lower()
        if hook in (None, ""):
            hook = mb.group(2).strip()
    algo = ep.normalize_algorithm(hook)
    if zfp is None:
        zfp = _zfp_from_block(block)
    if zfp is None:
        w = ep.RE_WORK.findall(block)
        if w:
            zfp = ep.infer_zfp_rate(w[-1][0])
    return ep.pretty_method_name(algo, zfp), backend

def parse_run_records(root: Path, warmup_epochs: int = 1) -> pd.DataFrame:
    files: List[Path] = []
    logs = root / "logs"
    if logs.exists():
        files += sorted(logs.glob("*.log"))
    files += sorted(root.glob("*.out")) + sorted(root.glob("*.txt")) + sorted(root.glob("*.log"))
    seen = set()
    files = [f for f in files if not (f in seen or seen.add(f))]

    rows = []
    for f in files:
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        for meta, block in _iter_run_blocks(text):
            work = ep.RE_WORK.findall(block)          # (label, calls, total, mean)
            tailm = ep.RE_TAIL.findall(block)          # (label, steps, total, mean)
            work_totals = [float(m[2]) for m in work]
            work_means = [float(m[3]) for m in work]
            steps = [int(m[1]) for m in tailm]
            tail_totals = [float(m[2]) for m in tailm]
            if not work_totals and not tail_totals:
                continue

            ranks = gbatch = bpr = math.nan
            mr = ep.RE_RANKS.search(block)
            if mr:
                ranks = int(mr.group(1)); bpr = int(mr.group(3)); gbatch = int(mr.group(4))
            if (isinstance(bpr, float) and math.isnan(bpr)) and meta and meta.get("batch"):
                try:
                    bpr = int(meta["batch"])
                except Exception:
                    pass

            train_only = [float(x) for x in ep.RE_TRAIN_ONLY.findall(block)]
            method, backend = _method_from(meta, block)

            def _skip(seq):
                return seq[warmup_epochs:] if len(seq) > warmup_epochs else seq
            w, wm, tt, st, tr = map(_skip, (work_totals, work_means, tail_totals, steps, train_only))

            n = min(len(w), len(st))
            hook_iter = [w[i] / st[i] for i in range(n) if st[i] > 0]
            m = min(len(tt), len(st))
            tail_iter = [tt[i] / st[i] for i in range(m) if st[i] > 0]

            mean = lambda x: float(np.mean(x)) if len(x) else math.nan
            rows.append({
                "source_file": str(f), "method": method, "backend": backend,
                "ranks": ranks, "global_batch": gbatch, "batch_per_rank": bpr,
                "hook_per_epoch": mean(w), "hook_per_iter": mean(hook_iter),
                "hook_per_call": mean(wm),
                "tail_per_epoch": mean(tt), "tail_per_iter": mean(tail_iter),
                "epoch_ms": (mean(tr) * 1000.0) if len(tr) else math.nan,
            })
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------- #
# Title                                                                       #
# --------------------------------------------------------------------------- #

def make_title(root, mode, scope, gpus=None, global_batch=None, batch_per_rank=None,
               backend=None):
    model_name = root.parts[-3] if len(root.parts) >= 3 else root.name
    model_name = {"wideresnet": "WideResNet", "resnet50": "ResNet-50", "vit": "ViT"}.get(
        model_name.lower(), model_name)
    scaling = "Strong Scaling" if mode == "strong" else "Weak Scaling"
    rs = str(root).lower()
    plat = "Frontier" if "frontier" in rs else "Polaris"
    vendor = "AMD" if plat == "Frontier" else "NVIDIA"
    quals = [vendor] + ([backend] if backend else [])
    platform = f"{plat} ({', '.join(quals)})"

    parts = [f"{model_name} on {platform}"]
    if gpus is not None:
        parts.append(f"{gpus} GPUs")
    parts.append(scaling)
    line1 = " — ".join(parts)

    line2 = None
    if mode == "strong" and global_batch is not None:
        line2 = f"Global batch {global_batch}"
    elif mode == "weak" and batch_per_rank is not None:
        line2 = f"Local batch {batch_per_rank}"
    return f"{line1} — {line2}" if line2 else line1

# --------------------------------------------------------------------------- #
# Hook plots (grouped family bars, plot_metric.py style)                       #
# --------------------------------------------------------------------------- #

def _legend_below(ax):
    handles, labels = ax.get_legend_handles_labels()
    l2h = dict(zip(labels, handles))
    order = [m for m in METHOD_ORDER if m in l2h] + [l for l in labels if l not in METHOD_ORDER]
    ax.legend([l2h[m] for m in order], order, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False,
              fontsize=10, columnspacing=1.4, handletextpad=0.5)

def plot_hook_grouped(df, value_col, ylabel, title, out, x_col, x_label, ymax=None, fmt="{:.3f}"):
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
        ax.bar(x + offset_map.get(method, 0.0), vals, width=width, label=method,
               color=METHOD_COLORS.get(method, "#999999"), edgecolor="black", linewidth=0.35)

    fig.suptitle(title, fontsize=18, y=0.99)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x_values], fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    _legend_below(ax)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {out}")

# --------------------------------------------------------------------------- #
# Tail plot (horizontal, tail vs epoch)                                        #
# --------------------------------------------------------------------------- #

def plot_tail(df, mode, backend, title, out, xmax=None):
    agg = (df.groupby("method", as_index=False)
             .agg(tail=("tail_per_epoch", "mean"), epoch=("epoch_ms", "mean")))
    order = ordered_methods(agg["method"].unique())
    agg["method"] = pd.Categorical(agg["method"], categories=order, ordered=True)
    agg = agg.sort_values("method").reset_index(drop=True)

    for _, r in agg.iterrows():
        if pd.notna(r["epoch"]) and pd.notna(r["tail"]) and r["tail"] > r["epoch"]:
            print(f"[WARN] tail > train-only epoch for {r['method']}: "
                  f"tail={r['tail']:.1f} ms  epoch={r['epoch']:.1f} ms")

    # Per-epoch times are large in ms -> show in seconds.
    agg["tail_s"] = agg["tail"] / 1000.0
    agg["epoch_s"] = agg["epoch"] / 1000.0

    y = np.arange(len(agg))
    h = 0.38
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.55 * len(agg) + 2)))
    ax.barh(y + h / 2, agg["epoch_s"], height=h, color="#d9d9d9",
            edgecolor="black", label="Train-only epoch")
    ax.barh(y - h / 2, agg["tail_s"], height=h,
            color=[METHOD_COLORS.get(m, "#ef9a9a") for m in agg["method"]],
            edgecolor="black", label="Exposed tail")
    ax.set_yticks(y)
    ax.set_yticklabels(agg["method"].astype(str), fontsize=12)
    ax.set_xlabel("Time per epoch (s)", fontsize=15)
    if xmax is not None:
        ax.set_xlim(0, xmax)
    fig.suptitle(title, fontsize=15, y=0.99)
    ax.grid(axis="x", alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {out}")

# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Experiment folder (contains .out/.log files)")
    ap.add_argument("--mode", choices=["strong", "weak"], required=True)
    ap.add_argument("--kind", choices=["hook", "tail"], required=True)
    ap.add_argument("--scope", choices=["all", "fixed"], default="all")
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--global-batch", type=int, default=None,
                    help="Filter to one global batch (strong scaling)")
    ap.add_argument("--batch-per-rank", type=int, default=None,
                    help="Filter to one local batch/rank (weak scaling)")
    ap.add_argument("--unit", choices=["auto", "iter", "epoch", "call"], default="auto",
                    help="hook unit; auto = per-iter for weak, per-epoch for strong")
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="Drop this many leading (warmup) epochs per run before "
                         "averaging. Default 0 = include all epochs.")
    ap.add_argument("--backend", type=str, default=None)
    ap.add_argument("--ymax", type=float, default=None, help="Max y (hook plot)")
    ap.add_argument("--xmax", type=float, default=None,
                    help="Max x for the tail plot in SECONDS — pin the same scale across GPU counts")
    ap.add_argument("--debug", action="store_true",
                    help="Print what each run block parsed into (method/ranks/epochs)")
    ap.add_argument("--png", action="store_true", help="Also write a .png next to the .pdf")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.root.exists():
        raise SystemExit(f"Root does not exist: {args.root}")
    if args.scope == "fixed" and args.gpus is None:
        raise SystemExit("--gpus is required when --scope fixed")
    if args.kind == "tail" and args.gpus is None:
        raise SystemExit(
            "Tail must be computed at a fixed GPU count — pass --gpus N.\n"
            "  Exposed tail depends on compute/comm at a specific GPU count, so "
            "pooling across GPU counts is not meaningful. (Hook may use --scope all.)"
        )

    apply_style()

    df = parse_run_records(args.root, warmup_epochs=max(0, args.warmup_epochs))
    if df.empty:
        raise SystemExit(f"No hook/tail data found under {args.root}")

    if args.debug:
        cols = ["method", "backend", "ranks", "global_batch", "batch_per_rank",
                "hook_per_epoch", "tail_per_epoch"]
        print("[DEBUG] parsed runs (before dropping Baseline):")
        print(df[cols].to_string(index=False))
        print(f"[DEBUG] method counts: {df['method'].value_counts().to_dict()}")

    backend = args.backend or backend_label(df)
    # On Frontier, PyTorch reports the backend string as "nccl" even though the
    # actual library is RCCL. Prefer RCCL when the path signals it.
    if args.backend is None:
        rs = str(args.root).lower()
        if "rccl" in rs or "frontier" in rs:
            backend = "RCCL"
    df = prefer_online_rate8(df)

    # Baseline has no meaningful hook -> drop it everywhere.
    non_baseline = df[df["method"] != "Baseline"].copy()
    if non_baseline.empty:
        found = df["method"].value_counts().to_dict()
        raise SystemExit(
            "No non-baseline methods with hook/tail data.\n"
            f"  Parsed {len(df)} run(s), but every one was labelled: {found}\n"
            "  This means the hook/algorithm name wasn't recovered from the run "
            "banner or the 'Backend: X, Algorithm: Y' line.\n"
            "  Re-run with --debug to see per-run parsing, and check that each "
            "run block contains either '=== Starting training: ... hook=... ===' "
            "or a 'Backend: ..., Algorithm: ...' line."
        )
    df = non_baseline

    if args.global_batch is not None:
        df = df[df["global_batch"] == args.global_batch].copy()
    if args.batch_per_rank is not None:
        df = df[df["batch_per_rank"] == args.batch_per_rank].copy()
    if df.empty:
        raise SystemExit("No runs left after batch filtering")

    # Tail is always per-GPU-count; hook uses the GPU filter only for --scope fixed.
    if args.gpus is not None and (args.scope == "fixed" or args.kind == "tail"):
        df = df[df["ranks"] == args.gpus].copy()
        if df.empty:
            raise SystemExit(f"No data at {args.gpus} GPUs")

    unit = args.unit
    if unit == "auto":
        # Default to the physically-bounded per-call mean (matches the reference
        # "Mean hook work time" figure). The per-epoch/per-iter *sums* over-count
        # overlapping async bucket allreduces and can exceed wall time, so they're
        # opt-in via --unit epoch / --unit iter.
        unit = "call"

    gb = args.global_batch
    bpr = args.batch_per_rank
    if gb is None and args.mode == "strong":
        u = df["global_batch"].dropna().unique()
        gb = int(u[0]) if len(u) == 1 else None
    if bpr is None and args.mode == "weak":
        u = df["batch_per_rank"].dropna().unique()
        bpr = int(u[0]) if len(u) == 1 else None

    title = args.title if getattr(args, "title", None) else make_title(
        args.root, args.mode, args.scope, gpus=args.gpus,
        global_batch=gb, batch_per_rank=bpr, backend=backend)

    # Output: default into <root>/plots/ as PDF (mirrors plot_metric.py); --png adds PNG.
    if args.out is None:
        outdir = ep.ensure_plot_dir(args.root)
        scope_tag = f"{args.gpus}gpu" if args.gpus is not None else "allgpus"
        batch_tag = (f"_gb{gb}" if gb is not None else "") + (f"_bpr{bpr}" if bpr is not None else "")
        if args.kind == "hook":
            name = f"{args.mode}_{scope_tag}_hook_{unit}{batch_tag}"
        else:
            name = f"{args.mode}_{scope_tag}_tail{batch_tag}"
        args.out = outdir / f"{name}.pdf"
    outputs = [args.out]
    if args.png and args.out.suffix.lower() != ".png":
        outputs.append(args.out.with_suffix(".png"))

    if args.kind == "hook":
        value_col = {"iter": "hook_per_iter", "epoch": "hook_per_epoch",
                     "call": "hook_per_call"}[unit]
        ylabel = {"iter": "Mean hook work per iter (ms)",
                  "epoch": "Avg total hook work per epoch (ms)",
                  "call": "Mean hook work time (ms)"}[unit]
        fmt = {"iter": "{:.1f}", "epoch": "{:.0f}", "call": "{:.3f}"}[unit]

        # Warn when a summed unit exceeds wall time (overlap over-count).
        if unit in ("epoch", "iter"):
            chk = df.groupby("method").agg(v=(value_col, "median"),
                                           ep=("epoch_ms", "median"))
            for m, r in chk.iterrows():
                bound = r["ep"] if unit == "epoch" else (r["ep"] / 1.0)
                if unit == "epoch" and pd.notna(r["ep"]) and r["v"] > r["ep"]:
                    print(f"[WARN] {m}: {value_col}={r['v']:.0f} ms exceeds epoch "
                          f"{r['ep']:.0f} ms — this is summed async hook work "
                          f"(overlapping calls), not wall time.")

        for out in outputs:
            if args.scope == "all":
                plot_hook_grouped(df, value_col, ylabel, title, out,
                                  x_col="ranks", x_label="GPUs", ymax=args.ymax, fmt=fmt)
            else:
                x_col = "global_batch" if args.mode == "strong" else "batch_per_rank"
                x_label = "Global batch" if args.mode == "strong" else "Batch per rank"
                plot_hook_grouped(df, value_col, ylabel, title, out,
                                  x_col=x_col, x_label=x_label, ymax=args.ymax, fmt=fmt)
    else:
        for out in outputs:
            plot_tail(df, args.mode, backend, title.replace("\n", " — "), out, xmax=args.xmax)


if __name__ == "__main__":
    main()