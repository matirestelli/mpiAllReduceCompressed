#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional
import pandas as pd

RE_MODEL = re.compile(r"Distributed Training:\s*(.+?)\s+on\s+(.+)")
RE_BACKEND_ALGO = re.compile(r"Backend:\s*(\w+),\s*Algorithm:\s*(.+)")
RE_RANKS = re.compile(
    r"Ranks:\s*(\d+),\s*Epochs:\s*(\d+),\s*Batch/rank:\s*(\d+),\s*Effective batch:\s*(\d+)"
)
RE_LR = re.compile(r"LR:\s*([0-9.eE+-]+),\s*Device:")
RE_TITER = re.compile(
    r"t_iter median:\s*([0-9.]+)\s*ms\s*\(fwd\s*([0-9.]+)\s*\|\s*bwd\+comm\s*([0-9.]+)\s*\|\s*opt\s*([0-9.]+)\s*\|\s*data\s*([0-9.]+)\)"
)
RE_TRAIN_ONLY = re.compile(r"train-only epoch:\s*([0-9.]+)s")
RE_EPOCH_TOTAL = re.compile(r"\[Epoch\s+(\d+)\].*?\|\s*Time:\s*([0-9.eE+-]+)s")
RE_DONE = re.compile(r"Done in\s*([0-9.]+)s\s*\|\s*Best val acc:\s*([0-9.]+)%")

RE_WORK = re.compile(
    r"\[HOOK_WORK_SUMMARY\]\s*label=([^\s]+)\s*calls=(\d+)\s*work_total_ms=([0-9.]+)\s*work_mean_ms=([0-9.]+)"
)
RE_TAIL = re.compile(
    r"\[HOOK_TAIL_SUMMARY\]\s*label=([^\s]+)\s*steps=(\d+)\s*tail_total_ms=([0-9.]+)\s*tail_mean_ms=([0-9.]+)"
)

RE_START_BLOCK = re.compile(
    r"=== Starting:\s*model=([^,]+),\s*ws=(\d+),\s*bs=(\d+),\s*gb=(\d+),\s*lr=([0-9.eE+-]+),.*?backend=([^,]+),\s*hook=([^,]+),\s*zfp=([^\s]+)"
)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan

def normalize_algorithm(algo: Optional[str]) -> str:
    if algo is None:
        return "unknown"
    algo = algo.strip().lower()
    algo = algo.replace("built-in (no hook)", "none")
    algo = algo.replace("built-in", "none")
    algo = algo.replace("no hook", "none")
    return algo

def infer_zfp_rate(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"rate[:]?(\d+)", text)
    if m:
        return m.group(1)
    m = re.search(r"rate(\d+)", text)
    if m:
        return m.group(1)
    m = re.search(r"zfp[=:]?(\d+)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    return None

def pretty_method_name(algo: str, zfp_rate: Optional[str]) -> str:
    algo = normalize_algorithm(algo)

    if algo == "none":
        return "Baseline"
    if algo == "ring":
        return "Ring"
    if algo == "recursive_doubling":
        return "Recursive doubling"
    if algo == "ring_zfp_naive":
        return f"Ring+ZFP naive (rate:{zfp_rate})" if zfp_rate else "Ring+ZFP naive"
    if algo == "ring_zfp_online_coll":
        return f"Ring+ZFP online (rate:{zfp_rate})" if zfp_rate else "Ring+ZFP online"
    if algo == "recursive_doubling_zfp_naive":
        return f"RD+ZFP naive (rate:{zfp_rate})" if zfp_rate else "RD+ZFP naive"
    if algo == "recursive_doubling_zfp_online_coll":
        return f"RD+ZFP online (rate:{zfp_rate})" if zfp_rate else "RD+ZFP online"

    return algo

def infer_algorithm_from_filename(name: str) -> Optional[str]:
    name = name.lower()
    if "builtin" in name or "_none_" in name:
        return "none"
    if "ring_zfp_naive" in name:
        return "ring_zfp_naive"
    if "ring_zfp_online_coll" in name:
        return "ring_zfp_online_coll"
    if "recursive_doubling_zfp_naive" in name:
        return "recursive_doubling_zfp_naive"
    if "recursive_doubling_zfp_online_coll" in name:
        return "recursive_doubling_zfp_online_coll"
    if "recursive_doubling" in name:
        return "recursive_doubling"
    if "ring" in name:
        return "ring"
    return None

def parse_log_file(path: Path) -> dict:
    text = path.read_text(errors="ignore")

    rec = {
        "source_file": str(path),
        "source_kind": "log",
        "model": None,
        "dataset": None,
        "backend": None,
        "algorithm": None,
        "zfp_rate": None,
        "method": None,
        "ranks": math.nan,
        "epochs": math.nan,
        "batch_per_rank": math.nan,
        "global_batch": math.nan,
        "lr": math.nan,
        "best_val_acc": math.nan,
        "train_only_epoch_s": math.nan,
        "epoch_wall_mean_s": math.nan,
        "t_iter_median_ms": math.nan,
        "hook_work_mean_ms": math.nan,
        "tail_mean_ms": math.nan,
    }

    m = RE_MODEL.search(text)
    if m:
        rec["model"] = m.group(1).strip()
        rec["dataset"] = m.group(2).strip()

    m = RE_BACKEND_ALGO.search(text)
    if m:
        rec["backend"] = m.group(1).strip().lower()
        rec["algorithm"] = normalize_algorithm(m.group(2).strip())

    m = RE_RANKS.search(text)
    if m:
        rec["ranks"] = int(m.group(1))
        rec["epochs"] = int(m.group(2))
        rec["batch_per_rank"] = int(m.group(3))
        rec["global_batch"] = int(m.group(4))

    m = RE_LR.search(text)
    if m:
        rec["lr"] = safe_float(m.group(1))

    titer = RE_TITER.findall(text)
    if titer:
        rec["t_iter_median_ms"] = safe_float(titer[-1][0])

    train_only = RE_TRAIN_ONLY.findall(text)
    if train_only:
        rec["train_only_epoch_s"] = safe_float(train_only[-1])

    epoch_times = [safe_float(m.group(2)) for m in RE_EPOCH_TOTAL.finditer(text)]
    if epoch_times:
        rec["epoch_wall_mean_s"] = sum(epoch_times) / len(epoch_times)

    done = RE_DONE.search(text)
    if done:
        rec["best_val_acc"] = safe_float(done.group(2))

    work = RE_WORK.findall(text)
    if work:
        rec["hook_work_mean_ms"] = safe_float(work[-1][3])
        rec["zfp_rate"] = infer_zfp_rate(work[-1][0])

    tail = RE_TAIL.findall(text)
    if tail:
        rec["tail_mean_ms"] = safe_float(tail[-1][3])
        rec["zfp_rate"] = rec["zfp_rate"] or infer_zfp_rate(tail[-1][0])

    if rec["algorithm"] is None:
        rec["algorithm"] = infer_algorithm_from_filename(path.name)

    if rec["zfp_rate"] is None:
        rec["zfp_rate"] = infer_zfp_rate(path.name)

    rec["method"] = pretty_method_name(rec["algorithm"], rec["zfp_rate"])
    return rec

def parse_out_file(path: Path) -> pd.DataFrame:
    text = path.read_text(errors="ignore")
    lines = text.splitlines()

    records = []
    current = None
    block = []

    def flush():
        nonlocal current, block
        if current is None:
            return

        block_text = "\n".join(block)
        rec = {
            "source_file": str(path),
            "source_kind": "out",
            "model": current["model"],
            "dataset": "cifar10",
            "backend": current["backend"],
            "algorithm": normalize_algorithm(current["hook"]),
            "zfp_rate": None if current["zfp"] == "none" else current["zfp"],
            "method": None,
            "ranks": int(current["ws"]),
            "epochs": math.nan,
            "batch_per_rank": int(current["bs"]),
            "global_batch": int(current["gb"]),
            "lr": safe_float(current["lr"]),
            "best_val_acc": math.nan,
            "train_only_epoch_s": math.nan,
            "epoch_wall_mean_s": math.nan,
            "t_iter_median_ms": math.nan,
            "hook_work_mean_ms": math.nan,
            "tail_mean_ms": math.nan,
        }

        m = RE_RANKS.search(block_text)
        if m:
            rec["epochs"] = int(m.group(2))

        titer = RE_TITER.findall(block_text)
        if titer:
            rec["t_iter_median_ms"] = safe_float(titer[-1][0])

        train_only = RE_TRAIN_ONLY.findall(block_text)
        if train_only:
            rec["train_only_epoch_s"] = safe_float(train_only[-1])

        epoch_times = [safe_float(m.group(2)) for m in RE_EPOCH_TOTAL.finditer(block_text)]
        if epoch_times:
            rec["epoch_wall_mean_s"] = sum(epoch_times) / len(epoch_times)

        done = RE_DONE.search(block_text)
        if done:
            rec["best_val_acc"] = safe_float(done.group(2))

        work = RE_WORK.findall(block_text)
        if work:
            rec["hook_work_mean_ms"] = safe_float(work[-1][3])
            rec["zfp_rate"] = infer_zfp_rate(work[-1][0]) or rec["zfp_rate"]

        tail = RE_TAIL.findall(block_text)
        if tail:
            rec["tail_mean_ms"] = safe_float(tail[-1][3])
            rec["zfp_rate"] = infer_zfp_rate(tail[-1][0]) or rec["zfp_rate"]

        rec["method"] = pretty_method_name(rec["algorithm"], rec["zfp_rate"])
        records.append(rec)
        current = None
        block = []

    for line in lines:
        m = RE_START_BLOCK.search(line)
        if m:
            flush()
            current = {
                "model": m.group(1),
                "ws": m.group(2),
                "bs": m.group(3),
                "gb": m.group(4),
                "lr": m.group(5),
                "backend": m.group(6),
                "hook": m.group(7),
                "zfp": m.group(8),
            }
            block = [line]
            continue

        if current is not None:
            block.append(line)
            if line.startswith("=== Finished training:"):
                flush()

    flush()
    return pd.DataFrame(records)

def load_experiment_folder(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    logs_dir = root / "logs"

    log_records = []
    if logs_dir.exists():
        for p in sorted(logs_dir.glob("*.log")):
            try:
                log_records.append(parse_log_file(p))
            except Exception as e:
                print(f"[WARN] failed to parse {p}: {e}")

    out_frames = []
    for p in sorted(list(root.glob("*.out")) + list(root.glob("*.txt"))):
        try:
            df = parse_out_file(p)
            if not df.empty:
                out_frames.append(df)
        except Exception as e:
            print(f"[WARN] failed to parse {p}: {e}")

    df_log = pd.DataFrame(log_records)
    df_out = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()

    if df_log.empty and df_out.empty:
        return pd.DataFrame()
    if df_log.empty:
        df = df_out.copy()
    elif df_out.empty:
        df = df_log.copy()
    else:
        key_cols = ["model", "ranks", "batch_per_rank", "global_batch", "algorithm", "zfp_rate"]
        df = df_log.copy()

        for idx, row in df.iterrows():
            mask = pd.Series(True, index=df_out.index)
            for c in key_cols:
                mask &= (df_out[c].astype(str) == str(row[c]))
            matches = df_out[mask]
            if matches.empty:
                continue
            ref = matches.iloc[0]
            for c in df.columns:
                if (pd.isna(df.at[idx, c]) or df.at[idx, c] in [None, ""]) and c in ref.index:
                    df.at[idx, c] = ref[c]

        existing = set(
            tuple(str(v) for v in row)
            for row in df[key_cols].fillna("NA").itertuples(index=False, name=None)
        )

        add_rows = []
        for _, row in df_out.iterrows():
            key = tuple(str(row[c]) for c in key_cols)
            if key not in existing:
                add_rows.append(row)

        if add_rows:
            df = pd.concat([df, pd.DataFrame(add_rows)], ignore_index=True)

    df["algorithm"] = df["algorithm"].apply(normalize_algorithm)
    df["method"] = df.apply(lambda r: pretty_method_name(r["algorithm"], r["zfp_rate"]), axis=1)
    return df

def ensure_plot_dir(root: str | Path) -> Path:
    root = Path(root)
    out = root / "plots"
    out.mkdir(exist_ok=True)
    return out
