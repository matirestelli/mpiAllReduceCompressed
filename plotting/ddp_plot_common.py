from pathlib import Path
import re

from matplotlib.patches import Patch

EPOCH_RE = re.compile(r"\[Epoch\s+(\d+)\].*?\|\s*Time:\s*([0-9.eE+-]+)s")
SUMMARY_RE = re.compile(
    r"Epoch times:\s*mean=([0-9.eE+-]+)s\s+min=([0-9.eE+-]+)s\s+max=([0-9.eE+-]+)s"
)
DONE_RE = re.compile(r"Done in\s*([0-9.eE+-]+)s\s*\|\s*Best val acc:\s*([0-9.eE+-]+)%")
WORK_RE = re.compile(
    r"\[HOOK_WORK_SUMMARY\].*?work_total_ms=([0-9.eE+-]+).*?work_mean_ms=([0-9.eE+-]+)"
)
TAIL_RE = re.compile(
    r"\[HOOK_TAIL_SUMMARY\].*?tail_total_ms=([0-9.eE+-]+).*?tail_mean_ms=([0-9.eE+-]+)"
)

HOOK_ORDER = [
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

HOOK_X = {
    "Baseline": 0.0,
    "Ring": 1.7,
    "Ring+ZFP naive (rate:16)": 2.25,
    "Ring+ZFP online (rate:16)": 2.80,
    "Ring+ZFP online (rate:10)": 3.35,
    "Recursive doubling": 5.05,
    "RD+ZFP naive (rate:16)": 5.60,
    "RD+ZFP online (rate:16)": 6.15,
    "RD+ZFP online (rate:8)": 6.70,
}

HOOK_COLORS = {
    "Baseline": "#4472C4",
    "Ring": "#70AD47",
    "Ring+ZFP naive (rate:16)": "#ED7D31",
    "Ring+ZFP online (rate:16)": "#00B050",
    "Ring+ZFP online (rate:10)": "#92D050",
    "Recursive doubling": "#A64D79",
    "RD+ZFP naive (rate:16)": "#FF0000",
    "RD+ZFP online (rate:16)": "#FFC000",
    "RD+ZFP online (rate:8)": "#F4B183",
}


def apply_paper_style(plt):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )


def hook_sort_key(label):
    if label in HOOK_ORDER:
        return HOOK_ORDER.index(label)
    return len(HOOK_ORDER)


def add_grouped_hook_legend(ax, fontsize=10.5, y=-0.08):
    baseline_handles = [
        Patch(
            facecolor=HOOK_COLORS["Baseline"],
            edgecolor="black",
            linewidth=0.3,
            label="Baseline",
        )
    ]

    ring_handles = [
        Patch(
            facecolor=HOOK_COLORS["Ring"],
            edgecolor="black",
            linewidth=0.3,
            label="Ring",
        ),
        Patch(
            facecolor=HOOK_COLORS["Ring+ZFP naive (rate:16)"],
            edgecolor="black",
            linewidth=0.3,
            label="Ring+ZFP naive (rate:16)",
        ),
        Patch(
            facecolor=HOOK_COLORS["Ring+ZFP online (rate:16)"],
            edgecolor="black",
            linewidth=0.3,
            label="Ring+ZFP online (rate:16)",
        ),
        Patch(
            facecolor=HOOK_COLORS["Ring+ZFP online (rate:10)"],
            edgecolor="black",
            linewidth=0.3,
            label="Ring+ZFP online (rate:10)",
        ),
    ]

    rd_handles = [
        Patch(
            facecolor=HOOK_COLORS["Recursive doubling"],
            edgecolor="black",
            linewidth=0.3,
            label="Recursive doubling",
        ),
        Patch(
            facecolor=HOOK_COLORS["RD+ZFP naive (rate:16)"],
            edgecolor="black",
            linewidth=0.3,
            label="RD+ZFP naive (rate:16)",
        ),
        Patch(
            facecolor=HOOK_COLORS["RD+ZFP online (rate:16)"],
            edgecolor="black",
            linewidth=0.3,
            label="RD+ZFP online (rate:16)",
        ),
        Patch(
            facecolor=HOOK_COLORS["RD+ZFP online (rate:8)"],
            edgecolor="black",
            linewidth=0.3,
            label="RD+ZFP online (rate:8)",
        ),
    ]

    baseline_legend = ax.legend(
        handles=baseline_handles,
        loc="upper left",
        bbox_to_anchor=(-0.02, y),
        frameon=False,
        fontsize=fontsize,
        handlelength=0.9,
        handletextpad=0.45,
    )
    ax.add_artist(baseline_legend)

    ring_legend = ax.legend(
        handles=ring_handles,
        loc="upper left",
        bbox_to_anchor=(0.25, y),
        frameon=False,
        fontsize=fontsize,
        handlelength=0.9,
        handletextpad=0.45,
    )
    ax.add_artist(ring_legend)

    ax.legend(
        handles=rd_handles,
        loc="upper left",
        bbox_to_anchor=(0.66, y),
        frameon=False,
        fontsize=fontsize,
        handlelength=0.9,
        handletextpad=0.45,
    )


def repo_root():
    return Path(__file__).resolve().parents[1]


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def parse_gpu_count(path):
    for part in Path(path).parts:
        m = re.fullmatch(r"(\d+)GPU[s]?(?:_.*)?", part)
        if m:
            return int(m.group(1))
    return None


def pretty_hook_from_filename(path):
    name = Path(path).stem.lower()

    if "builtin" in name or "buildin" in name:
        return "Baseline"
    if "recursive_doubling_zfp_online_coll" in name:
        base = "RD+ZFP online"
    elif "ring_zfp_online_coll" in name:
        base = "Ring+ZFP online"
    elif "recursive_doubling_zfp_naive" in name:
        base = "RD+ZFP naive"
    elif "ring_zfp_naive" in name:
        base = "Ring+ZFP naive"
    elif "recursive_doubling" in name:
        base = "Recursive doubling"
    elif "ring" in name:
        base = "Ring"
    elif "default" in name:
        base = "Default hook"
    else:
        base = Path(path).stem

    rate = re.search(r"rate([0-9.]+)", name)
    if rate:
        base += f" (rate:{rate.group(1)})"
    return base


def find_logs(results_root, model, gpus=None, run=None):
    root = Path(results_root)
    if not root.is_absolute():
        root = repo_root() / root

    logs = []
    for path in root.rglob("*.log"):
        lowered = str(path).lower()
        if model.lower() not in lowered:
            continue
        if gpus is not None and parse_gpu_count(path) != gpus:
            continue
        if run is not None and f"/{run.lower()}/" not in lowered:
            continue
        logs.append(path)

    return sorted(logs)


def parse_log(path):
    text = Path(path).read_text(errors="replace")

    epoch_times = [float(m.group(2)) for m in EPOCH_RE.finditer(text)]

    summary = SUMMARY_RE.search(text)
    done = DONE_RE.search(text)

    work_total, work_mean = [], []
    tail_total, tail_mean = [], []

    for m in WORK_RE.finditer(text):
        work_total.append(float(m.group(1)))
        work_mean.append(float(m.group(2)))

    for m in TAIL_RE.finditer(text):
        tail_total.append(float(m.group(1)))
        tail_mean.append(float(m.group(2)))

    return {
        "epoch_times": epoch_times,
        "epoch_mean": float(summary.group(1)) if summary else mean(epoch_times),
        "epoch_min": float(summary.group(2))
        if summary
        else min(epoch_times)
        if epoch_times
        else None,
        "epoch_max": float(summary.group(3))
        if summary
        else max(epoch_times)
        if epoch_times
        else None,
        "total_time": float(done.group(1)) if done else None,
        "best_val_acc": float(done.group(2)) if done else None,
        "hook_work_mean_ms": mean(work_mean),
        "hook_tail_mean_ms": mean(tail_mean),
        "hook_work_total_ms": mean(work_total),
        "hook_tail_total_ms": mean(tail_total),
    }


def metric_value(stats, metric, skip_first_epoch):
    if metric == "epoch_mean":
        return mean(stats["epoch_times"][skip_first_epoch:])
    return stats[metric]
