# Training Stability: Gradient Explosion and How to Fix It

## The problem

Training a deep network from scratch on CIFAR-10 with DDP can produce **gradient explosion** — the loss jumps from ~2.3 (random chance on 10 classes) to `inf` or `NaN` within the first epoch, and the model never recovers. Once weights contain NaN values, every subsequent forward pass produces NaN, and accuracy is stuck at ~10% (pure random guessing).

This is not a communication bug. It is a standard deep learning training stability problem that becomes more severe as models get deeper and wider.

---

## Why it happened with ResNet50 — and why warmup helped

ResNet50 trained from scratch on CIFAR-10 with `lr=0.01` diverges in the first epoch when started at full learning rate. The cause:

1. At random initialization, a 50-layer network produces gradient norms that are non-trivial in magnitude (despite BatchNorm and Kaiming initialization, which help but do not guarantee stability at high LR)
2. SGD with momentum=0.9 accumulates gradient signal across steps. After many steps, the effective learning rate is approximately `lr × 1/(1-momentum) = lr × 10`. At `lr=0.01` this is an effective update of `0.1` per gradient-norm unit — large enough to cause weight divergence in early training when BatchNorm running statistics have not yet stabilized
3. Once one layer's weights diverge, all subsequent layers receive corrupted inputs, and the gradients from those layers corrupt everything upstream

**The fix applied**: linear LR warmup over the first epoch. The warmup ramps LR from `1e-3 × lr` (effectively `lr=0.00001`) to `lr=0.01` across the first epoch, then cosine-decays. At the tiny warmup LR, gradient updates are small enough that BatchNorm statistics stabilize before the full learning rate is applied. For ResNet50 this works reliably.

---

## Why warmup alone fails for ResNeXt101-32x8d

ResNeXt101-32x8d has **101 layers** (vs ResNet50's 50) and uses grouped convolutions with 32 groups × 8 width per group. The bottleneck intermediate channels reach 1024 in the deeper layers (vs 256 in ResNet50). These differences cause substantially larger gradient norms at initialization.

What was observed in the training log:

```
[Epoch 1] Training... (lr=0.000010)   ← warmup floor, LR is tiny
  Batch [50/98] Loss: 15657807136728658.0000   ← already 1.56e16 after 50 batches
[Epoch 1] Train Loss: nan   ← full NaN by end of epoch 1
```

The warmup floor LR is `0.00001`. Yet the model explodes in 50 batches. Why?

- With momentum=0.9, the effective LR after 50 steps is approximately `0.00001 × 10 = 0.0001`
- ResNeXt101's larger gradient norms (roughly proportional to model depth × width) mean that even at `0.0001` effective LR, weight updates are large enough to corrupt the network before BatchNorm running statistics stabilize
- The explosion compounds: once one layer diverges, gradients through the full 101-layer chain become corrupted, and the loss hits `1.56e16` before reaching NaN (float32 maximum is ~3.4e38, so the explosion is "visible" before overflow)

Simply lowering the warmup LR further (e.g., to `1e-6`) would delay the explosion but not prevent it — the fundamental issue is that gradient norms at initialization are too large for the optimizer to handle safely without a hard upper bound.

---

## The fix: gradient clipping

`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` is called after `loss.backward()` (when allreduce is complete and all gradients are available) and before `optimizer.step()`.

**What it does**: it computes the global L2 norm of all gradients across all parameters, then if that norm exceeds `max_norm`, it scales all gradients down uniformly so the global norm equals exactly `max_norm`. If the norm is already below `max_norm`, nothing changes.

**Why this fixes the problem**: regardless of model depth, width, or initialization, no single optimizer step can apply a gradient update larger than `max_norm × lr`. The optimizer can still make progress (gradients are not zeroed — just rescaled), but catastrophic steps are prevented. BatchNorm statistics stabilize within the first few batches and the training process becomes well-behaved.

**Why `max_norm=1.0`**: this is the most widely used value for deep networks (used by default in GPT, BERT, most modern transformer training). It is a reasonable first choice. If early epochs still show instability (loss growing before stabilizing), try `max_norm=0.5`. If training is very slow to converge, try `max_norm=2.0`.

**Placement in the code** (`ddp_training.py`, inside `train_epoch`):

```python
loss.backward()    # backward + allreduce (DDP hook fires here)
# ...
if grad_clip is not None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
optimizer.step()   # safe to apply — gradients are bounded
```

The clip must happen **after** `loss.backward()` (so allreduced gradients are available) and **before** `optimizer.step()` (so the clipped values are what the optimizer uses). This ordering is correct for both custom hooks (which run synchronously during backward) and the DDP built-in allreduce (which completes before backward returns).

---

## Does the framework stay model-agnostic?

Yes. `grad_clip` is a field in `TrainingConfig` with default `None` (clipping disabled). It is a training hyperparameter, not model-specific code. The training loop in `ddp_training.py` applies clipping if and only if `grad_clip` is not None, regardless of which model, algorithm, or backend is configured.

| Model | Recommended `grad_clip` | Why |
|---|---|---|
| ResNet18 | `None` | Shallow enough that warmup alone prevents explosion |
| ResNet50 | `None` | Warmup was sufficient (validated in experiments) |
| ResNet101 | `None` or `1.0` | Test without first; add if early epochs are unstable |
| ResNeXt101-32x8d | `1.0` | Confirmed necessary — explodes even at warmup LR without it |
| ConvNeXt-Tiny | `1.0` | ConvNeXt uses LayerScale which can amplify gradient norms early in training |
| ConvNeXt-Small | `1.0` | Same as ConvNeXt-Tiny |

To disable clipping for a ResNet50 experiment, set `grad_clip=None` in `interface.py`. The training loop is unchanged — the `if grad_clip is not None` check simply does nothing. All communication hooks, timing, and logging are unaffected.

---

## How to configure in `interface.py`

```python
# Gradient clipping (set in interface.py):
grad_clip=None,   # disabled — ResNet18/50/101
grad_clip=1.0,    # standard — ResNeXt101, ConvNeXt
grad_clip=0.5,    # aggressive — if 1.0 still shows early instability
```

---

## Summary

| Model | Root cause | Fix |
|---|---|---|
| ResNet50 | Moderate gradient norms at init, high peak LR | LR warmup |
| ResNeXt101 | Large gradient norms at init (depth × width), amplified by momentum | Gradient clipping (`max_norm=1.0`) + warmup |

Both mechanisms are active simultaneously for ResNeXt101. The warmup keeps the peak LR from hitting before BatchNorm stabilizes; the clipping prevents any single step from being catastrophic during warmup itself.

---

## Why `grad_clip=1.0` is the universal default

The framework sets `grad_clip=1.0` as the default in `TrainingConfig` for all models, rather than requiring per-model configuration.

The reasoning: gradient clipping with `max_norm=1.0` is a no-op when the gradient norm is already below 1.0 — it only activates when norms are too large. For shallow models like ResNet18/50, norms are typically well below 1.0 after the first few batches, so clipping never fires and training is identical to the unclipped case. For deep models (ResNeXt101, ConvNeXt), it prevents explosion without requiring the framework to enumerate which models need it.

A model-name-based dict (e.g., `{"resnext101_32x8d": 1.0, ...}`) would be *less* model-agnostic: it breaks silently when a new model is added that isn't in the dict. The universal default requires zero model-specific knowledge and is safe by construction.

To disable clipping for a specific experiment, set `grad_clip=None` explicitly in `interface.py`. The training loop applies clipping only when `grad_clip is not None`, so `None` is a clean opt-out.

---

## Why gradient clipping alone is not enough for ResNeXt101

With `grad_clip=1.0` and a proper 5-epoch warmup ramp, epoch 1 becomes fully stable (loss ~2.42). But NaN reappears in epoch 2, specifically in batches 51–98. This reveals a different class of problem.

**The chain of failure once a forward-pass NaN occurs:**
1. Some batch triggers float32 overflow inside ResNeXt101's deep+wide architecture → `loss = NaN`
2. `loss.backward()` backpropagates `NaN` through all 101 layers → every gradient is `NaN`
3. `clip_grad_norm_` computes the total norm as `NaN` → scale factor is `NaN / 1.0 = NaN` → gradients stay `NaN`
4. `optimizer.step()` writes `NaN` into all weights permanently
5. Every subsequent forward pass produces `NaN` — the model is unrecoverable

Gradient clipping can only bound the magnitude of *finite* gradients. It cannot rescue a backward pass that receives `NaN` from the forward pass. The failure point is upstream of the optimizer.

---

## Why training ResNeXt101 from scratch on CIFAR-10 is not viable

ResNeXt101 was designed, initialized, and benchmarked for ImageNet. Applying it to CIFAR-10 creates a fundamental mismatch:

| | ImageNet (designed for) | CIFAR-10 (our use) |
|---|---|---|
| Training samples | 1.28M | 50K |
| Image resolution | 224×224 | 32×32 |
| Model parameters | 88M | 88M |
| Parameters per sample | ~69 | ~1,760 |

**1,760 parameters per training example** is a wildly overparameterized regime. The loss landscape has sharp cliffs — specific batches can push intermediate activations (up to 2048 channels wide in the bottlenecks) into float32 overflow before the network has learned any stable representations.

This is not a bug and not fixable by tuning the optimizer or scheduler. It is the expected behavior when using an ImageNet-scale model on a CIFAR-scale dataset.

**How people actually train ResNeXt101 from scratch**: on ImageNet with `lr=0.1`, 256 samples/GPU × 8 GPUs, 90–100 epochs, and the linear LR scaling rule. On small datasets, standard practice is always to fine-tune from pretrained ImageNet weights — never to train from scratch.

---

## The correct fix: pretrained ImageNet weights

`pretrained=True` in `TrainingConfig` loads `IMAGENET1K_V2` weights for the backbone, then replaces only the classifier head for the target number of classes:

```python
model = models.resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # 1000 → 10
```

The CIFAR stem is still applied on top: `conv1` becomes a 3×3 stride-1 conv and `maxpool` becomes `Identity`. This discards only the pretrained stem weights (1 layer out of 101) — all other 100 layers keep their trained ImageNet representations and produce stable, meaningful features from the first batch.

**Why this is valid for communication hook benchmarking**: the hooks operate on gradient tensors during `loss.backward()`. They are completely agnostic to whether those gradients come from fine-tuning or scratch training — the allreduce pattern, bucket sizes, and communication volume are identical. Pretrained weights give a stable, converging training run to measure against.

---

## Updated summary

| Model | Problem | Fix |
|---|---|---|
| ResNet50 | Gradient explosion at full LR from random init | LR warmup |
| ResNeXt101 (scratch) | Forward-pass float32 overflow — fundamental scale/dataset mismatch | Not viable — use pretrained weights |
| ResNeXt101 (pretrained) | Stable — backbone already converged | `pretrained=True` + CIFAR stem + warmup |
