# How to Choose Batch Size for Distributed Training on A100

## The three constraints

Batch size must satisfy all three independently. The correct value is the one that fits within all three bounds simultaneously.

---

### 1. Memory constraint (hard limit)

GPU memory = **fixed overhead** + **batch_size × per-sample activation cost**

**Fixed overhead** (independent of batch size):
| Component | Cost |
|---|---|
| Model parameters (float32) | params × 4 bytes |
| Gradients | same as parameters |
| SGD momentum buffer | same as parameters |
| **Total fixed** | **params × 12 bytes** |

For the models in this project:

| Model | Params | Fixed overhead |
|---|---|---|
| ResNet18 | ~11M | ~132 MB |
| ResNet50 | ~25M | ~300 MB |
| ResNet101 | ~44M | ~528 MB |
| ResNeXt101-32x8d | ~88M | ~1.06 GB |
| ConvNeXt-Tiny | ~28M | ~336 MB |
| ConvNeXt-Small | ~50M | ~600 MB |

**Per-sample activation cost** depends on both the model depth and the input spatial resolution. CIFAR-10 (32×32) produces much smaller activation tensors than ImageNet (224×224) at the same batch size because spatial dimensions stay small throughout the network.

Rough estimates for CIFAR-10 with the stem fix applied (stride-1, no maxpool):

| Model | Approx. activation cost / sample |
|---|---|
| ResNet18 | ~10–20 MB |
| ResNet50 | ~30–50 MB |
| ResNet101 | ~50–80 MB |
| ResNeXt101-32x8d | ~50–80 MB |

**Total memory at a given batch size** = fixed overhead + batch_size × per-sample cost.

Target: stay below **~85% of VRAM** to leave room for allocator fragmentation and CUDA context overhead.

On A100-40GB:

| Model | batch_size=64 | batch_size=128 | batch_size=256 |
|---|---|---|---|
| ResNet50 | ~3.5 GB | ~6.7 GB | ~13 GB |
| ResNeXt101-32x8d | ~4.3 GB | ~7.5 GB | ~14 GB |

All values above are safe on A100-40GB. OOM is unlikely for any of the models in this project at batch_size=128 on CIFAR-10.

**How to measure precisely** (run once before committing to a batch size):

```python
torch.cuda.reset_peak_memory_stats()
outputs = model(inputs)      # inputs: one batch on GPU
loss = criterion(outputs, targets)
loss.backward()
print(torch.cuda.max_memory_allocated() / 1e9, "GB")
```

---

### 2. Statistical efficiency constraint (training quality)

Large batches see fewer gradient updates per epoch and tend to converge to sharper minima, which generalize worse (Keskar et al., 2017, "On Large-Batch Training for Deep Learning").

The practical rule for CIFAR-10: **effective batch ≤ 2048** without compensating with a larger learning rate.

Effective batch = `batch_size × world_size` (number of GPUs).

| batch_size | world_size=4 | Notes |
|---|---|---|
| 64 | 256 | Very conservative. More gradient noise (often good). Slower throughput. |
| 128 | 512 | **Good default.** Conservative, generalizes well, no LR adjustment needed. |
| 256 | 1024 | Still fine for CIFAR-10. Consider mild LR increase. |
| 512 | 2048 | At the boundary. Apply the linear scaling rule (see below). |
| 1024 | 4096 | Beyond the safe zone for CIFAR-10. Requires careful LR tuning + warmup. |

**Linear scaling rule** (Goyal et al., 2017, "Accurate, Large Minibatch SGD"):  
If you multiply effective batch by k, multiply the learning rate by k as well.  
Example: doubling from batch_size=128 to 256 → multiply `learning_rate` by 2.

The warmup scheduler in `ddp_training.py` already handles the regime where this is needed — it ramps LR from near-zero over the first few epochs, which is the standard technique for large-batch training.

---

### 3. Throughput constraint (GPU utilization)

Larger batches amortize CUDA kernel launch overhead and improve GPU utilization, but only up to the point where you saturate memory bandwidth or compute units.

For compute-heavy models (ResNeXt, ResNet101) on A100: batch_size=128 already gives near-peak GPU utilization on CIFAR-10. Increasing batch size further does not meaningfully improve throughput.

For very lightweight models (ResNet18): batch_size=256 or 512 may be needed to keep the GPU busy.

**How to check GPU utilization:**
```
nvidia-smi dmon -s u   # streams GPU utilization % every second
```
If utilization is consistently below 70%, your batch size may be too small and you are CPU/IO bound (check `num_workers` in the DataLoader, or use `pin_memory=True`).

---

## Decision procedure

1. Start with **batch_size=128**. This is safe for all models in this project on A100-40GB.
2. Run one epoch, monitor with `nvidia-smi`. If GPU memory usage is above 85% → go to 64.
3. If you want to experiment with larger batches to reduce epoch wall time:
   - Double `batch_size` to 256 → scale `learning_rate` × 2 in `interface.py`.
   - Keep effective batch ≤ 2048 (batch_size=512 at 4 GPUs) to stay in the safe zone.
4. If you OOM unexpectedly: check that no `.clone()` or `torch.empty()` calls appear inside the hook body — these are illegal (see CLAUDE.md bug list).

---

## This project's recommended settings

| Model | Recommended batch_size | Notes |
|---|---|---|
| ResNet18 | 128–256 | Lightweight — can push higher |
| ResNet50 | 128 | Standard baseline |
| ResNet101 | 128 | Same as ResNet50, memory similar on CIFAR-10 |
| ResNeXt101-32x8d | 128 | ~88M params but CIFAR-10 keeps activations small |
| ConvNeXt-Tiny | 128 | ConvNeXt uses more activation memory due to LayerNorm — start conservative |
| ConvNeXt-Small | 128 | Same caution as ConvNeXt-Tiny |

**Default**: `batch_size=128` in `interface.py`. Change only if you OOM or are deliberately experimenting with the scaling rule.
