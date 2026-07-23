# Plotting Tools

Lightweight plotting utilities for distributed training experiment logs.

These scripts parse experiment outputs and generate plots for:
- **strong scaling**
- **weak scaling**
- **single GPU-count comparisons**
- **all GPU-count comparisons**

Supported metrics:
- **time**
- **hook**
- **tail**

---

## Folder Contents

```bash
plotting_tools/
├── README.md
├── exp_parser.py
├── plot_metric.py
└── requirements.txt
```

### File Descriptions

#### `exp_parser.py`
Shared parser for experiment logs. It scans an experiment folder and extracts metrics from:
- `logs/*.log`
- Top-level `*.out`
- Top-level `*.txt`

It collects information such as:
- Model name and dataset
- Backend and communication algorithm / hook
- ZFP compression rate
- GPU count and batch size
- Epoch time and iteration time
- Hook work time and tail time

*Note: This file is not meant to be run directly.*

#### `plot_metric.py`
Main plotting script. This script produces all plots using three key arguments:
- `--mode` (Controls the scaling mode: `strong` or `weak`)
- `--scope` (Controls the plot scope: `fixed` for one GPU count or `all` for all GPU counts combined)
- `--metric` (Controls what metric to plot: `time`, `hook`, or `tail`)

#### `requirements.txt`
Python dependencies including `pandas`, `matplotlib`, `seaborn`, and `numpy`. Install them using:
```bash
pip install -r requirements.txt
```

---

## Installation & Setup

From inside the `plotting_tools/` directory, run:
```bash
pip install -r requirements.txt
```

### Recommended Project Layout
```text
my_project/
├── plotting_tools/
│   ├── README.md
│   ├── exp_parser.py
│   ├── plot_metric.py
│   └── requirements.txt
└── experiments_frontier/
    └── wideresnet/
        └── cifar10/
            ├── strongScaling/
            └── weakScaling/
```

---

## Running the Script

You can run the plotting script either from the project root or from inside `plotting_tools/`.

### Option A: Run from project root
```bash
python plotting_tools/plot_metric.py experiments_frontier/wideresnet/cifar10/strongScaling --mode strong --scope all --metric time --global-batch 512
```

### Option B: Run from inside plotting_tools/
```bash
cd plotting_tools
python plot_metric.py ../experiments_frontier/wideresnet/cifar10/strongScaling --mode strong --scope all --metric time --global-batch 512
```

---

## Command Line Interface

### Main Command Interface
```bash
python plotting_tools/plot_metric.py ROOT --mode {strong,weak} --scope {fixed,all} --metric {time,hook,tail}
```

### Argument Details

#### Positional Argument
- **`root`**: Experiment directory. (e.g., `experiments_frontier/wideresnet/cifar10/strongScaling`)

#### Required Arguments
- **`--mode {strong,weak}`**: Select experiment type.
  - `strong`: Time metric means train-only epoch time.
  - `weak`: Time metric means median iteration time.
- **`--scope {fixed,all}`**: Select plotting scope.
  - `fixed`: One GPU count only (x-axis = methods).
  - `all`: All GPU counts combined (x-axis = GPU counts, grouped by method).
- **`--metric {time,hook,tail}`**: Select metric to plot.
  - `time` / `hook` / `tail`

#### Optional Arguments
- **`--gpus N`**: Required when using `--scope fixed`. (e.g., `--gpus 8`)
- **`--global-batch N`**: Usually used for strong scaling plots. (e.g., `--global-batch 512`)
- **`--batch-per-rank N`**: Usually used for weak scaling plots. (e.g., `--batch-per-rank 64`)
- **`--title "Custom title"`**: Override the default plot title.
- **`--png`**: Also save a PNG version in addition to the PDF.
- **`--csv`**: Save the filtered dataframe used to generate the plot.
- **`--out FILE`**: Override the default output filename. (e.g., `--out my_plot.pdf`)

---

## Behavior Profiles

### Plot Scope Behavior
- **`Scope = fixed`**: Produces a single plot for one GPU count.
  - **x-axis**: Methods (e.g., Baseline, Ring, RD, Ring+ZFP at 8 GPUs)
  - **y-axis**: Selected metric
- **`Scope = all`**: Produces a single plot across all available GPU counts.
  - **x-axis**: GPU counts (e.g., 4, 8, 16 GPUs)
  - **grouped bars**: Methods
  - **y-axis**: Selected metric

### Metric Behavior
- **`--metric time`**: Plots train-only epoch time if `--mode strong`, or median iteration time if `--mode weak`.
- **`--metric hook`**: Plots mean hook work time.
- **`--metric tail`**: Plots mean exposed tail time.

---

## Example Commands

### Strong Scaling Examples
**Fixed GPU count, epoch time:**
```bash
python plotting_tools/plot_metric.py \
  experiments_frontier/wideresnet/cifar10/strongScaling \
  --mode strong \
  --scope fixed \
  --metric time \
  --gpus 8 \
  --global-batch 512
```
*Result: One figure, 8 GPUs only, x-axis = methods, y-axis = training time per epoch.*

**All GPU counts, epoch time:**
```bash
python plotting_tools/plot_metric.py \
  experiments_frontier/wideresnet/cifar10/strongScaling \
  --mode strong \
  --scope all \
  --metric time \
  --global-batch 512
```
*Result: One figure, x-axis = GPU counts, grouped bars = methods, y-axis = training time per epoch.*

### Weak Scaling Examples
**Fixed GPU count, iteration time:**
```bash
python plotting_tools/plot_metric.py \
  experiments_frontier/wideresnet/cifar10/weakScaling \
  --mode weak \
  --scope fixed \
  --metric time \
  --gpus 8 \
  --batch-per-rank 64
```
*Result: One figure, 8 GPUs only, x-axis = methods, y-axis = median iteration time.*

**All GPU counts, iteration time:**
```bash
python plotting_tools/plot_metric.py \
  experiments_frontier/wideresnet/cifar10/weakScaling \
  --mode weak \
  --scope all \
  --metric time \
  --batch-per-rank 64
```
*Result: One figure, x-axis = GPU counts, grouped bars = methods, y-axis = median iteration time.*

### Hook & Tail Profiles
```bash
# Hook work, fixed GPU count
python plotting_tools/plot_metric.py experiments_frontier/wideresnet/cifar10/strongScaling --mode strong --scope fixed --metric hook --gpus 8 --global-batch 512

# Hook work, all GPU counts
python plotting_tools/plot_metric.py experiments_frontier/wideresnet/cifar10/strongScaling --mode strong --scope all --metric hook --global-batch 512

# Tail, fixed GPU count
python plotting_tools/plot_metric.py experiments_frontier/wideresnet/cifar10/strongScaling --mode strong --scope fixed --metric tail --gpus 8 --global-batch 512

# Tail, all GPU counts
python plotting_tools/plot_metric.py experiments_frontier/wideresnet/cifar10/strongScaling --mode strong --scope all --metric tail --global-batch 512
```

---

## Output Files

If `--out` is not provided, plots are written directly to `<root>/plots/`.

### Default Filenames

| Scenario | Fixed Scope Output (`--scope fixed --gpus 8`) | All Scope Output (`--scope all`) |
| :--- | :--- | :--- |
| **Strong Scaling** | `strong_fixed_8gpu_[metric].pdf` | `strong_allgpus_[metric].pdf` |
| **Weak Scaling** | `weak_fixed_8gpu_[metric].pdf` | `weak_allgpus_[metric].pdf` |

*Note: If `--png` or `--csv` flags are checked, a PNG image or CSV datasheet with the same base file stem will also be generated.*

---

## Important Notes

- **Required Flags**: `--gpus` is strictly mandatory for `fixed` scope plots to isolate data.
- **Filtering Practice**: Use `--global-batch` to clean up strong scaling profiles, and `--batch-per-rank` to clean up weak scaling runs. 
- **Missing Baselines**: Baseline runs occasionally lack structured hook/tail summaries. Time plots will show the baseline, but hook or tail plots may automatically omit it.
- **Cluster Environments**: The plotting script relies on a non-interactive backend (`matplotlib.use("Agg")`). It is safe to use across headless node clusters, batch configurations, and remote login nodes.
