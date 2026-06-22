# Bug Report: Ring Hook — B0 Bucket Never Re-fired After First Backward

---

## What is B0?

DDP divides the model's parameters into communication *buckets*. During the
backward pass, as soon as all gradients in a bucket are computed, DDP fires the
comm hook for that bucket. The buckets are numbered by gradient-readiness order:
bucket 0 (B0) contains the parameters whose gradients are computed **first**
during backward — which in a ResNet50 is the output side of the network
(layer4, FC layer). B0 is the largest bucket: **23.5M of the model's ~46.5M
parameters (~50%)**, covering roughly the last two ResBlocks and the classifier.

Because B0 fires first and contains half the model, it is the most
safety-critical bucket. A bug that silences B0 is equivalent to training half
the model without gradient synchronization.

---

## Symptom

Training with the ring allreduce hook produces catastrophically high loss
despite the ring allreduce verifying as numerically correct against
`dist.all_reduce` for every logged call.

| Config                           | Ep1 B50 Loss | Ep1 Val Loss | Ep3 Val Acc |
|----------------------------------|-------------|-------------|------------|
| Builtin DDP (no hook), 15 ep     | 2.38        | 2.11        | 62.00%     |
| Custom default hook, 3 ep        | 2.47        | 2.07        | 32.48%     |
| Ring hook — run 1 (pre-resolved) | 16.18       | 59,650      | 9.52%      |
| Ring hook — run 2 (static_graph) | 19.93       | 189,370     | 12.12%     |

---

## Evidence — Run 1

### Ring is numerically correct (all VERIFY pass)

```
Line  61: [VERIFY | R0|B0|C1] max_abs_err=5.960e-08  ✓ MATCH
Line 108: [VERIFY | R0|B1|C1] max_abs_err=7.451e-09  ✓ MATCH
Line 155: [VERIFY | R0|B2|C1] max_abs_err=2.328e-10  ✓ MATCH
Line 202: [VERIFY | R0|B3|C1] max_abs_err=4.657e-10  ✓ MATCH
Line 249: [VERIFY | R0|B4|C1] max_abs_err=9.313e-10  ✓ MATCH
Line 296: [VERIFY | R0|B5|C1] max_abs_err=5.960e-08  ✓ MATCH
```

All errors are floating-point rounding (~1e-7 for float32). The algorithm is
mathematically correct.

### B0 vanishes from backward pass 2 onward

Line 296 is the last B0 output. Line 298, the very next line, jumps to B1C2:

```
Line 296: [VERIFY | R0|B5|C1] ✓ MATCH
Line 298: [RING | R0|B1|C2] global_tag_before=36
```

`grep "B0|C2"` returns zero matches across the entire 3-epoch log.

### Tag counter proof

Each ring invocation consumes 6 tags (3 steps × 2 phases). The tag counter
goes from 36 → 36 between B5C1 ending and B1C2 starting — zero tags consumed,
B0's ring never ran.

---

## A note on "chunk[0] after Phase 1 looks unreduced"

The Phase 1 log shows rank 0's chunk[0] keeping its original input value after
Phase 1, with a misleading `<-- fully reduced` label:

```
[R0|B0|C1] === AFTER Phase 1 ===
  chunk[0] <-- fully reduced: [0.4086, 0.3648, ...]   ← same as input!
  chunk[1]: [-0.0008, ...]                              ← tiny, gradient-scale
```

This looks suspicious but is **not a bug**. The `<-- fully reduced` label in
our logging code is wrong: it marks `chunk[i]` as owned when `i == rank`, but
rank 0 actually owns chunk[1] in this Python implementation (the index formula
shifts ownership by 1 vs the C++ reference). Rank 3 owns chunk[0] — it does
the final accumulation for chunk[0] in Phase 1 and then sends it to rank 0 in
Phase 2 step 0:

```
P2 step=0 RECVD chunk[0]←rank3: [0.6768, 0.4521, 0.3578, -0.3830]
  chunk[0] now=[0.6768, 0.4521, 0.3578, -0.3830]
```

After Phase 2, chunk[0] = [0.6768, ...] which is the correct fully-reduced
sum. The VERIFY confirms: `max_abs_err=5.96e-08 ✓ MATCH`. The ring algorithm
is correct.

---

## Root Cause

Our ring hook returns a **pre-resolved Future**:

```python
fut = torch.futures.Future()
fut.set_result(tensor)   # resolved synchronously before return
return fut
```

DDP's `_default_allreduce_hook` (which works correctly) returns an
**unresolved** Future:

```python
fut = dist.all_reduce(tensor, async_op=True).get_future()
return fut.then(lambda f: f.value()[0])
```

This future resolves asynchronously on a communication thread *after* the
current C++ call stack has unwound. DDP's C++ Reducer tracks bucket completion
through the Work object behind that future, and correctly resets B0's autograd
hook registration before the next backward.

When DDP receives a pre-resolved Future for B0 (the first bucket to complete),
it processes the result synchronously *inside the same C++ stack frame* as the
backward pass. This interferes with `prepare_for_backward()`'s re-registration
of B0's autograd hook for the next backward — the hook is silently dropped.
B1–B5 fire later and are outside the critical window, so they are unaffected.

---

## Run 2 — `static_graph=True` attempt

Adding `static_graph=True` was a failed first attempt. Results from the second
run show it made things worse (Ep1 batch 50 loss: 19.93 vs 16.18).

The new `[HOOK | ...]` unconditional print reveals what happened:

```
Line  15: [HOOK | R0|B0|C1|numel=23520842]   ← B0 fires once...
Line  63: [HOOK | R0|B0|C2|numel=23520842]   ← ...then immediately again, same batch!
Line 111: [HOOK | R0|B1|C1|numel=1073162]    ← only now does B1 fire for the first time
...
Line 590: [VERIFY | R0|B5|C2] ✓ MATCH
Line 591: [HOOK | R0|B1|C3|numel=1073162]    ← no B0C3 anywhere from here on
```

`static_graph=True` changed the symptom: B0 now fires **twice** during batch 1
(both calls verified MATCH), then stops completely. The pre-resolved Future is
still the problem. With `static_graph=True`, DDP's first-backward initialization
phase fires B0's hook twice as part of static graph construction, consuming
`call_count` slots 1 and 2. From batch 2 onward, B0's hook still doesn't fire.
Result: B0 gets a double gradient update in batch 1 and then no updates ever
again — explaining the worse loss of 19.93.

`static_graph=True` has been reverted.

---

## Fix (Run 3)

Pre-allocate a scalar zero GPU tensor (`_dummy`) in `RingBucketState`. After
the ring completes, do a scalar `dist.all_reduce` on `_dummy` with
`async_op=True` and chain our already-correct `tensor` onto its future:

```python
# In RingBucketState.initialize():
self._dummy = torch.zeros(1, dtype=tensor.dtype, device=tensor.device)

# In _ring_allreduce_hook(), replacing the pre-resolved Future:
work = dist.all_reduce(bstate._dummy, op=dist.ReduceOp.SUM, async_op=True)
return work.get_future().then(lambda _: tensor)
```

The scalar allreduce on a persistent zero tensor is a no-op on the gradients
(sums to zero across all ranks, tensor never touched). It completes almost
instantly since all ranks are already synchronized by the ring. DDP's C++ layer
receives a real async Work future it knows how to track, properly resetting B0's
autograd hook registration before each subsequent backward.

---

## Verification (expected from Run 3)

The `[HOOK | ...]` lines should now show all 6 buckets appearing once per
backward pass, in every batch across all epochs:

```
[HOOK | R0|B0|C1|numel=23520842]   ← batch 1
[HOOK | R0|B1|C1|numel=1073162]
...
[HOOK | R0|B0|C2|numel=23520842]   ← batch 2, was missing before
[HOOK | R0|B1|C2|numel=1073162]
...
[HOOK | R0|B0|C3|numel=23520842]   ← batch 3, was missing before
```

Training loss should match the default hook and builtin (~2.4 at epoch 1
batch 50, converging to ~32% val acc at epoch 3).

---

## Training Results After Fix

*(To be filled after Run 3.)*
