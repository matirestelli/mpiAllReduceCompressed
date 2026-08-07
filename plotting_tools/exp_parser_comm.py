#!/usr/bin/env python3
"""
exp_parser_comm.py

Separate parser for communication-oriented metrics:
- hook work mean/total
- tail mean/total
- tail as % of train epoch time
- tail as % of hook work

Parses experiment directories recursively and emits one normalized CSV:
    parsed_comm.csv

Expected inputs inside each run directory (or descendants):
- results/*.csv                      training CSV written by your script
- *.log                             stdout log containing HOOK_*_SUMMARY lines

Usage:
    python exp_parser_comm.py /path/to/experiments --out parsed_comm.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RE_WORK = re.compile(
    r"\[HOOK_WORK_SUMMARY\].*?epoch=(?P<epoch>\d+).*?"
    r"work_total_ms=(?P<total>[0-9.]+).*?"
    r"work_mean_ms=(?P<mean>[0-9.]+).*?"
    r"calls=(?P<calls>\d+)"
)

RE_TAIL = re.compile(
    r"\[HOOK_TAIL_SUMMARY\].*?epoch=(?P<epoch>\d+).*?"
    r"tail_total_ms=(?P<total>[0-9.]+).*?"
    r"tail_mean_ms=(?P<mean>[0-9.]+).*?"
    r"steps=(?P<steps>\d+)"
)

def find_training_csvs(root: Path) -> List[Path]:
    return sorted(root.rglob("results/*.csv"))

def find_logs_near(csv_path: Path) -> List[Path]:
    """
    Look for logs near the CSV first, then higher-level fallback.
    """
    candidates = set()

    # same dir as csv
    for p in csv_path.parent.glob("*.log"):
        candidates.add(p)

    # parent of results/
    if csv_path.parent.name == "results":
        run_dir = csv_path.parent.parent
        for p in run_dir.glob("*.log"):
            candidates.add(p)

    # one more fallback: anywhere below run_dir
    if csv_path.parent.name == "results":
        run_dir = csv_path.parent.parent
        for p in run_dir.rglob("*.log"):
            candidates.add(p)

    return sorted(candidates)

def parse_log_summaries(log_path: Path) -> Tuple[Dict[int, dict], Dict[int, dict]]:
    work_by_epoch: Dict[int, dict] = {}
    tail_by_epoch: Dict[int, dict] = {}

    try:
        text = log_path.read_text(errors="ignore")
    except Exception:
        return work_by_epoch, tail_by_epoch

    for m in RE_WORK.finditer(text):
        epoch = int(m.group("epoch"))
        work_by_epoch[epoch] = {
            "hook_work_total_ms": float(m.group("total")),
            "hook_work_mean_ms": float(m.group("mean")),
            "hook_calls": int(m.group("calls")),
        }

    for m in RE_TAIL.finditer(text):
        epoch = int(m.group("epoch"))
        tail_by_epoch[epoch] = {
            "tail_total_ms": float(m.group("total")),
            "tail_mean_ms": float(m.group("mean")),
            "tail_steps": int(m.group("steps")),
        }

    return work_by_epoch, tail_by_epoch

def choose_best_log(csv_path: Path) -> Optional[Path]:
    """
    Pick the log with the largest number of hook summary matches.
    """
    best = None
    best_score = -1
    for logp in find_logs_near(csv_path):
        work, tail = parse_log_summaries(logp)
        score = len(work) + len(tail)
        if score > best_score:
            best = logp
            best_score = score
    return best

def parse_method(row: dict) -> str:
    algo = (row.get("algorithm") or "").strip()
    zfp_rate = (row.get("zfp_rate") or "").strip()

    if algo in ("", "builtin"):
        return "Baseline"
    if algo == "ring":
        return "Ring"
    if algo == "recursive_doubling":
        return "Recursive doubling"
    if algo == "ring_zfp":
        return f"Ring+ZFP naive (rate:{zfp_rate})"
    if algo == "recursive_doubling_zfp":
        return f"RD+ZFP naive (rate:{zfp_rate})"
    if algo == "ring_zfp_overlap":
        return f"Ring+ZFP online (rate:{zfp_rate})"
    if algo == "recursive_doubling_zfp_overlap":
        return f"RD+ZFP online (rate:{zfp_rate})"

    # fallback
    if "zfp" in algo and zfp_rate:
        return f"{algo} (rate:{zfp_rate})"
    return algo or "Unknown"

def safe_float(x, default=math.nan):
    try:
        return float(x)
    except Exception:
        return default

def safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

def parse_training_csv(csv_path: Path) -> List[dict]:
    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def normalize_records(csv_path: Path) -> List[dict]:
    raw_rows = parse_training_csv(csv_path)
    if not raw_rows:
        return []

    log_path = choose_best_log(csv_path)
    work_by_epoch, tail_by_epoch = ({}, {})
    if log_path is not None:
        work_by_epoch, tail_by_epoch = parse_log_summaries(log_path)

    out = []
    for row in raw_rows:
        epoch = safe_int(row.get("epoch"))
        ranks = safe_int(row.get("world_size"))
        batch_per_rank = safe_int(row.get("batch_size"))
        global_batch = safe_int(row.get("global_batch_size"))
        model = (row.get("model") or "").strip()
        backend = (row.get("backend") or "").strip()
        method = parse_method(row)

        train_only_epoch_s = safe_float(row.get("epoch_train_time_s"))
        t_iter_median_ms = safe_float(row.get("t_iter_median_ms"))
        epoch_wall_s = safe_float(row.get("epoch_time_s"))

        work = work_by_epoch.get(epoch, {})
        tail = tail_by_epoch.get(epoch, {})

        hook_work_total_ms = work.get("hook_work_total_ms", math.nan)
        hook_work_mean_ms = work.get("hook_work_mean_ms", math.nan)
        hook_calls = work.get("hook_calls", 0)

        tail_total_ms = tail.get("tail_total_ms", math.nan)
        tail_mean_ms = tail.get("tail_mean_ms", math.nan)
        tail_steps = tail.get("tail_steps", 0)

        if math.isfinite(train_only_epoch_s) and math.isfinite(tail_total_ms) and train_only_epoch_s > 0:
            tail_pct_of_train_epoch = 100.0 * tail_total_ms / (train_only_epoch_s * 1000.0)
        else:
            tail_pct_of_train_epoch = math.nan

        if math.isfinite(hook_work_total_ms) and math.isfinite(tail_total_ms) and hook_work_total_ms > 0:
            tail_pct_of_hook = 100.0 * tail_total_ms / hook_work_total_ms
        else:
            tail_pct_of_hook = math.nan

        if math.isfinite(train_only_epoch_s) and math.isfinite(hook_work_total_ms) and train_only_epoch_s > 0:
            hook_pct_of_train_epoch = 100.0 * hook_work_total_ms / (train_only_epoch_s * 1000.0)
        else:
            hook_pct_of_train_epoch = math.nan

        out.append({
            "source_csv": str(csv_path),
            "source_log": str(log_path) if log_path else "",
            "epoch": epoch,
            "model": model,
            "backend": backend,
            "method": method,
            "ranks": ranks,
            "batch_per_rank": batch_per_rank,
            "global_batch": global_batch,
            "epoch_wall_s": epoch_wall_s,
            "train_only_epoch_s": train_only_epoch_s,
            "t_iter_median_ms": t_iter_median_ms,
            "hook_work_mean_ms": hook_work_mean_ms,
            "hook_work_total_ms": hook_work_total_ms,
            "hook_calls": hook_calls,
            "tail_mean_ms": tail_mean_ms,
            "tail_total_ms": tail_total_ms,
            "tail_steps": tail_steps,
            "tail_pct_of_train_epoch": tail_pct_of_train_epoch,
            "tail_pct_of_hook": tail_pct_of_hook,
            "hook_pct_of_train_epoch": hook_pct_of_train_epoch,
        })

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="Experiment root")
    ap.add_argument("--out", type=Path, default=Path("parsed_comm.csv"))
    args = ap.parse_args()

    csvs = find_training_csvs(args.root)
    if not csvs:
        raise SystemExit(f"No results/*.csv found under {args.root}")

    records: List[dict] = []
    for csvp in csvs:
        records.extend(normalize_records(csvp))

    if not records:
        raise SystemExit("No records parsed")

    fieldnames = list(records[0].keys())
    with args.out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(records)

    print(f"Wrote {len(records)} rows to {args.out}")

if __name__ == "__main__":
    main()
