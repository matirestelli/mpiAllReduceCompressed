# DDP Scaling Study — Experiment Tracker

**Model:** `wide_resnet50_2` (CIFAR stem, from scratch) · **Dataset:** CIFAR-10 · **System:** Frontier (1 node = 8 GCDs)
**Backend:** GPU-aware MPI (non-negotiable) · **Epochs:** 20 everywhere · **Seed:** 42

---

## Key facts to keep in your head

| Quantity | Value |
|---|---|
| Params (10-class head) | 66.85 M |
| **Gradient buffer, fp32** | **~267 MB per allreduce** |
| Allreduces per epoch @ gb128 | 390 |
| 1 Frontier node | 4× MI250X = **8 GCDs** = 8 ranks |
| P=4 | half a node (**document which GCDs**) |
| P=16 | **first inter-node hop** — sample this point carefully |

**Steps per epoch** (50 000 imgs, `drop_last=true`):

| B_global | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| steps/epoch | 390 | 195 | 97 | 48 | **24** | **12** |
| total steps (20 ep) | 7800 | 3900 | 1940 | 960 | 480 | 240 |

> ⚠️ **Accuracy claims are only valid for `B_global ≤ 1024`.** Above that there are too few optimizer steps in 20 epochs. Larger batches are **throughput-only** rows — say so in every caption.

---

## The LR rule (everything depends on this)

```
lr(B_global) = LR_REF × B_global / 128
```

`LR_REF` is determined in **Phase 1a** and then **never changes**. Expect `LR_REF = 0.1`.

| B_global | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|
| **LR** (if LR_REF=0.1) | 0.1 | 0.2 | 0.4 | 0.8 | 1.6 | 3.2 | 6.4 |
| **`WARMUP_EPOCHS`** | 0 | 0 | 2 | 2 | 3 | 3 | 3 |

---

## Constants — set once, never touch again

```bash
export MODEL_NAME="wide_resnet50_2"
export DATASET="cifar10"
export NUM_CLASSES="10"
export IMAGE_SIZE="32"
export NUM_EPOCHS="20"
export CIFAR_STEM="true"
export PRETRAINED="false"
export INIT_CIFAR_STEM_FROM_PRETRAINED_CENTER="false"
export DROP_LAST="true"          # MANDATORY — equal steps/rank, no hang
export SCHEDULER="cosine"        # per-ITERATION stepping
export GRAD_CLIP="none"
export SEED="42"
export NUM_WORKERS="7"           # 56 cores / 8 ranks
export PIN_MEMORY="true"
export BACKEND="mpi"

# Frontier env — MANDATORY
export MIOPEN_USER_DB_PATH="/tmp/miopen-$USER-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export OMP_NUM_THREADS=7

# Profiling — keep BOTH at 0 for all timing runs
export DDP_ITER_LOG=0
export DDP_PROFILE_BARRIER=0
```

---

# PHASE 0 — Correctness (do not skip)

Short runs. No science, just proof the framework is sound.

| # | Test | Pass criterion | ☐ |
|---|---|---|---|
| 0.1 | Gradient check: P=1 vs P=4, same fixed batch, same seed | averaged grads match single-GPU grad on concatenated batch to ~1e-5 (fp32) | ☐ |
| 0.2 | Loss curve P=1 vs P=4, same `B_global`, 100 steps | overlap within noise | ☐ |
| 0.3 | `len(train_loader)` identical on every rank | ✅ (guaranteed by `drop_last=true` on **both** DataLoader and DistributedSampler) | ☐ |
| 0.4 | `total` inside `validate()` | **exactly 10000**, at every P | ☐ |
| 0.5 | `[Optim] group 1` param count | **≈130,000** (not 0 — if 0, the WD split failed) | ☐ |
| 0.6 | `[Sched]` line prints `steps/epoch`, `total_steps`, `warmup_steps` | matches the table above | ☐ |
| 0.7 | Instrumentation overhead A/B (`DDP_ITER_TIMING=1` vs `0`, 3 epochs) | **<1%** delta on `epoch_time_s` — record the number for your methods section | ☐ |
| 0.8 | `osu_allreduce` @ 267 MB on the same allocation, P ∈ {8,16,64} | your hook within ~80% of it | ☐ |

---

# PHASE 1 — Calibration · **P = 8 (one node)** · 17 runs

> **Rule: `EXPERIMENTS=("none:")` only.** Lossless hooks compute identical gradients → identical accuracy. Sweeping hooks here just measures the same number N times.

## 1a — Optimizer / LR sweep → **picks `LR_REF`** · 7 runs

Fixed: `BATCH_SIZE=16`, `NUM_PROCS=8` → **B_global = 128**, `WARMUP_EPOCHS=0`, `MOMENTUM=0.9`, `NESTEROV=true`, `WD_ON_BN_BIAS=false`

| # | `OPTIMIZER` | `LEARNING_RATE` | `WEIGHT_DECAY` | Purpose | val_acc | ☐ |
|---|---|---|---|---|---|---|
| **A1** | sgd | **0.001** | 5e-4 | literal paper reproduction | | ☐ |
| **A2** | sgd | 0.01 | 5e-4 | | | ☐ |
| **A3** | sgd | 0.05 | 5e-4 | | | ☐ |
| **A4** | sgd | **0.1** | 5e-4 | ← expected winner | | ☐ |
| **A5** | sgd | 0.2 | 5e-4 | | | ☐ |
| **A6** | adamw | 0.001 | **0.05** | the recipe the paper *probably* used | | ☐ |
| **A7** | adamw | 0.003 | **0.05** | | | ☐ |

> ⚠️ AdamW needs `WEIGHT_DECAY=0.05`, **not** 5e-4. Run A6/A7 as a separate job with `WEIGHT_DECAYS=("0.05")`.

**Output:** `LR_REF = ______` (winner among A1–A5). A1-vs-A4 is your answer to *"was the paper's 0.001 an SGD or an Adam LR?"*

## 1b — Momentum / WD / Nesterov ablation · 5 runs

Fixed: `OPTIMIZER=sgd`, `LEARNING_RATE=LR_REF`, `BATCH_SIZE=16` (B_global=128), `WARMUP_EPOCHS=0`

| # | `MOMENTUM` | `NESTEROV` | `WEIGHT_DECAY` | `WD_ON_BN_BIAS` | val_acc | ☐ |
|---|---|---|---|---|---|---|
| **B1** | 0.9 | `true` | 5e-4 | `false` | ← **baseline** | ☐ |
| **B2** | 0.9 | `false` | 5e-4 | `false` | | ☐ |
| **B3** | 0.9 | `true` | 1e-4 | `false` | | ☐ |
| **B4** | 0.9 | `true` | 5e-4 | **`true`** | ← old behaviour | ☐ |
| **B5** | 0.95 | `true` | 5e-4 | `false` | | ☐ |

**B1 − B4 = the "free 0.5–1%".** Now citable, not asserted. Runnable in **one job** via loops:
```bash
NESTEROV_VALUES=("true" "false"); WD_ON_BN_BIAS_VALUES=("false" "true"); WEIGHT_DECAYS=("5e-4" "1e-4")
```

## 1c — LR-rule validation → **justifies the B_global ≤ 1024 cap** · 5 runs

Fixed: **P=8**, winning recipe from 1a/1b. Vary `BATCH_SIZE` to move `B_global`.

| # | `BATCH_SIZE` | B_global | `LEARNING_RATE` | `WARMUP_EPOCHS` | steps/ep | val_acc | ☐ |
|---|---|---|---|---|---|---|---|
| **C1** | 16 | 128 | LR_REF × 1 | 0 | 390 | | ☐ |
| **C2** | 32 | 256 | LR_REF × 2 | 0 | 195 | | ☐ |
| **C3** | 64 | 512 | LR_REF × 4 | 2 | 97 | | ☐ |
| **C4** | 128 | 1024 | LR_REF × 8 | 2 | 48 | | ☐ |
| **C5** | 256 | 2048 | LR_REF × 16 | 3 | 24 | | ☐ |
| **C5′** | 256 | 2048 | LR_REF × **4** (√ rule) | 3 | 24 | | ☐ |

**Output:** accuracy-vs-global-batch curve. Expect flat → 512, mild drop at 1024, **cliff at 2048**. That cliff is the justification for the cap.

---

## ✅ FREEZE POINT

After Phase 1, write these down and **never change them again**:

```
LR_REF        = __________
OPTIMIZER     = __________
MOMENTUM      = __________
NESTEROV      = __________
WEIGHT_DECAY  = __________
WD_ON_BN_BIAS = __________
```

> From here on: **optimizer lists pinned to one value, `EXPERIMENTS` sweeps all hooks.** Never both.

---

# PHASE 2 — Strong scaling · **read `epoch_train_time_s`**

**`B_global` fixed. `B_local = B_global / P` → shrinks.** Hyperparameters **identical down each row** — same B_global ⇒ same LR.

> 🔬 **Free correctness check:** accuracy must be **P-invariant** within seed noise. If it drifts with P, your allreduce is buggy (summing instead of averaging, or dividing by world_size twice).

## Row S1 — B_global = 512 · `LEARNING_RATE = LR_REF × 4` · `WARMUP_EPOCHS = 2`

| P | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| **`BATCH_SIZE`** | 512 | 128 | 64 | 32 | 16 | 8 |
| `NUM_PROCS` / `PPN` | 1/1 | 4/4 | 8/8 | 16/8 | 32/8 | 64/8 |
| nodes | ⅛ | ½ | 1 | 2 | 4 | 8 |
| steps/epoch | 97 | 97 | 97 | 97 | 97 | 97 |
| ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

## Row S2 — B_global = 2048 · `LEARNING_RATE = LR_REF × 16` · `WARMUP_EPOCHS = 3` · **throughput-only**

| P | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| **`BATCH_SIZE`** | 2048 | 512 | 256 | 128 | 64 | 32 |
| ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

**Metrics:** `speedup = t(P=1)/t(P)`, `efficiency = speedup/P`, both from `epoch_train_time_s` (median epochs 2–20).

**Expected story:** S1 efficiency **collapses past P=8** (B_local → 16 → 8; compute vanishes, the 267 MB allreduce doesn't). S2 **holds up** (B_local stays ≥32). Same model, same allreduce, opposite conclusions. **That contrast is the systems result.**

---

# PHASE 3 — Weak scaling · **read `t_iter_median_ms`**

**`B_local` fixed. `B_global = B_local × P` → grows. LR scales with it.**

> ⚠️ **Never plot epoch time here.** The 50k dataset is fixed, so iterations/epoch shrink as P grows and epoch time falls even at **zero** parallel efficiency → fake superlinear speedup. `t_iter` is the only honest metric.

## Row W1 — `BATCH_SIZE = 8` (comms-bound corner case)

| P | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| B_global | 8 | 32 | 64 | 128 | 256 | 512 |
| **`LEARNING_RATE`** | LR_REF/16 | LR_REF/4 | LR_REF/2 | LR_REF | LR_REF×2 | LR_REF×4 |
| **`WARMUP_EPOCHS`** | 0 | 0 | 0 | 0 | 0 | 2 |
| ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

## Row W2 — `BATCH_SIZE = 32` (the paper's number · **main operating point**)

| P | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| B_global | 32 | 128 | 256 | 512 | 1024 | 2048 |
| **`LEARNING_RATE`** | LR_REF/4 | LR_REF | LR_REF×2 | LR_REF×4 | LR_REF×8 | LR_REF×16 |
| **`WARMUP_EPOCHS`** | 0 | 0 | 0 | 2 | 2 | 3 |
| ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

## Row W3 — `BATCH_SIZE = 128` (compute-saturated reference)

| P | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| B_global | 128 | 512 | 1024 | 2048 | 4096 | 8192 |
| **`LEARNING_RATE`** | LR_REF | LR_REF×4 | LR_REF×8 | LR_REF×16 | LR_REF×32 | LR_REF×64 |
| **`WARMUP_EPOCHS`** | 0 | 2 | 2 | 3 | 3 | 3 |
| ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

> ⚠️ **W3 at P ≥ 32:** B_global ≥ 4096 → 12 steps/epoch, LR > 3. **Throughput row. Do not report accuracy.** Say so in the caption.

**Expected:** W1 `t_iter` **flat and comm-dominated** (comm ≈ compute). W3 **good weak-scaling efficiency**. Together they bracket the regime.

---

# PHASE 4 — Communication algorithm comparison ⭐ *the paper*

**Run every hook across Row W2 (all P) and Row S1 (all P).** Same hyperparameters as those rows — only `COMM_ALGORITHM` changes.

```bash
EXPERIMENTS=(
    "none:"                          # DDP built-in MPI allreduce (baseline)
    "ring:"
    "recursive_doubling:"
    "ring_zfp_naive:16"
    "ring_zfp_online_coll:16"
    "ring_zfp_online_coll:10"
    "recursive_doubling_zfp_naive:16"
    "recursive_doubling_zfp_online_coll:16"
    "recursive_doubling_zfp_online_coll:8"
)
```

| | Coverage | ☐ |
|---|---|---|
| **Timing** — all hooks × S1 (P=1..64) | `epoch_train_time_s` | ☐ |
| **Timing** — all hooks × W2 (P=1..64) | `t_iter_median_ms` | ☐ |
| **Accuracy** — **lossless** hooks (`none`, `ring`, `recursive_doubling`) | identical up to fp32 non-associativity → **free correctness check** | ☐ |
| **Accuracy** — **lossy** hooks (all ZFP rates) | ⚠️ **changes the math** — needs its own 20-epoch runs at P=16, `BATCH_SIZE=32` | ☐ |
| **Exposed comm** — `t_iter(hook) − t_iter(no-allreduce)` | requires a `--no-allreduce` mode | ☐ |
| **Load imbalance** — one run with `DDP_PROFILE_BARRIER=1` | read `t_barrier_median_ms`; report separately | ☐ |
| **Non-power-of-2 P** (12, 20, 24 — the "4-in-4") | MPI falls back from recursive-halving to slower rings; asymmetric NIC load at 1.5 nodes | ☐ |
| **Bus bandwidth** — `busbw = (M/t_comm) × 2(P−1)/P`, M=267 MB | compare to `osu_allreduce` | ☐ |

> **Primary P axis = {4, 8, 16, 32, 64}.** Do the 4-in-4 sweep only as a deliberate *finding*, not as the main axis.

---

# PHASE 5 — Seeds

**Accuracy runs only.** Timing needs 3 *repeats of one seed*, not 3 seeds.

| Config | `SEED` | ☐ |
|---|---|---|
| Best recipe @ B_global = 128 | 42, 43, 44 | ☐ |
| Best recipe @ B_global = 512 | 42, 43, 44 | ☐ |
| Best recipe @ B_global = 1024 | 42, 43, 44 | ☐ |

Report **mean ± std**.

---

# Reading the results

| Plot | CSV column | Source row |
|---|---|---|
| Accuracy vs epoch | `val_acc` | any |
| **Strong scaling** speedup / efficiency | **`epoch_train_time_s`** (median ep 2–20) | S1, S2 |
| **Weak scaling** | **`t_iter_median_ms`** | W1, W2, W3 |
| Comm cost per hook | `t_bwd_median_ms` (bwd compute + exposed allreduce) | Phase 4 |
| Dataloader bottleneck check | `t_data_median_ms` | any |
| Time breakdown (stacked bar) | `t_fwd` + `t_bwd` + `t_opt` + `t_data` | any |
| Accuracy vs global batch | `val_acc` vs `global_batch_size` | C1–C5 |
| Load imbalance | `t_barrier_median_ms` | the one `DDP_PROFILE_BARRIER=1` run |

```python
df = pd.concat([pd.read_csv(f) for f in glob("results/*.csv")])
df[df.epoch > 1].groupby(["world_size","batch_size","algorithm"])["epoch_train_time_s"].median()
```

Filename globs:
- `results/*_gb512_*.csv` → an entire **strong-scaling** row
- `results/*_bs32_*.csv` → an entire **weak-scaling** row

---

# Rules I will forget and regret

1. **Discard epoch 1** from every timing median. MIOpen autotune + allocator warmup + first-touch faults.
2. **Report medians + IQR**, never means.
3. **`DDP_PROFILE_BARRIER=0`** for every timing run — it kills backward/comm overlap.
4. Phase 1: **sweep optimizer, pin hooks.** Phase 2–4: **pin optimizer, sweep hooks.** Never both — that's a cartesian-product explosion measuring the same number N times.
5. **`DRY_RUN=1 bash run.sh | grep -c '\[DRY\]'`** before every submission.
6. **19 `for`s → 19 `done`s.** Bash's `unexpected EOF` won't tell you which one.
7. **P=4 = half a node.** Document *which* GCDs — {0,1,2,3} (2 packages) and {0,2,4,6} (4 packages) give different numbers.
8. **Strong scaling ⇒ `t_epoch`. Weak scaling ⇒ `t_iter`.** Getting this backwards produces fake superlinear speedup.
9. Accuracy claims only for **B_global ≤ 1024**. Everything above is throughput-only — caption it.
10. At `BATCH_SIZE=8`, BN stats over 8 samples are noisy and `broadcast_buffers=False` means each rank keeps its own. It's a **deliberate corner case**, not the default. Footnote it.