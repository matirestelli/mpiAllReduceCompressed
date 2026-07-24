#!/usr/bin/env python3
"""
Standalone staged test for the HIERARCHICAL compressed allreduce algorithm.

Runs the SAME algorithm as the DDP hook on tiny, analytically-known vectors and
checks each stage separately, so a failure tells you WHICH layer is broken.

Two ways to run
---------------
1. LOGIN NODE, no GPU, fake multi-node topology (fastest iteration):

     torchrun --nproc_per_node=8 test_hier_algo.py \
         --backend gloo --fake-ppn 4 --codec identity

   8 processes on one machine, but --fake-ppn 4 makes the algorithm believe
   there are 2 nodes of 4 ranks. With --codec identity the compression is a
   no-op, so ANY mismatch is a pure algorithm/indexing bug.

2. COMPUTE NODES, real NCCL/RCCL (see run_hier_test.sh):

     srun -N2 --ntasks-per-node=8 python -u test_hier_algo.py \
         --backend nccl --codec identity
     srun -N2 --ntasks-per-node=8 python -u test_hier_algo.py \
         --backend nccl --codec zfp --rate 16

Stages
------
  S0 topology resolution
  S1 process-group construction
  S2 intra-node allreduce (step 1 alone)
  S3 leader compressed allreduce, phase 1 only  (reduce-scatter)
  S4 leader compressed allreduce, both phases
  S5 full hierarchical allreduce vs native allreduce

With --codec identity, S2-S5 must match EXACTLY (0.0 error). Any nonzero error
is an algorithm bug, not a compression artifact.
"""

import argparse
import os
import sys
import traceback

import torch
import torch.distributed as dist


# ═══════════════════════════════════════════════════════════════════════════
# Codecs — pluggable so we can separate "algorithm bug" from "ZFP bug"
# ═══════════════════════════════════════════════════════════════════════════

class IdentityCodec:
    """No-op 'compression': raw bytes. Lets us test the algorithm exactly."""
    name = "identity"

    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device
        self.itemsize = torch.empty(0, dtype=dtype).element_size()

    def max_bytes(self, numel):
        return numel * self.itemsize

    def compress(self, src, dst_u8):
        """src: float tensor [n]; dst_u8: uint8 [max_bytes(n)]"""
        dst_u8.copy_(src.contiguous().view(torch.uint8))

    def decompress(self, src_u8, nbytes, dst):
        dst.copy_(src_u8[:nbytes].view(self.dtype))


class ZfpCodec:
    """Real ZFP, using the project's helpers. Only importable where they exist."""
    name = "zfp"

    def __init__(self, dtype, device, rate):
        from communication_strategy import (  # noqa
            _zfp_max_output_bytes,
            _zfp_compress_into_current_stream,
            _zfp_decompress_into_current_stream,
            _zfp_pad_tail_to_B_,
        )
        self._max = _zfp_max_output_bytes
        self._comp = _zfp_compress_into_current_stream
        self._decomp = _zfp_decompress_into_current_stream
        self._pad = _zfp_pad_tail_to_B_
        self.rate = rate
        self.dtype = dtype
        self.device = device

    def max_bytes(self, numel):
        probe = torch.empty(numel, dtype=self.dtype, device=self.device)
        return int(self._max(probe, self.rate))

    def compress(self, src, dst_u8):
        used_bits = self._comp(src, dst_u8, self.rate)
        self._pad(dst_u8, used_bits, dst_u8.numel())

    def decompress(self, src_u8, nbytes, dst):
        self._decomp(src_u8, nbytes, dst, self.rate)


# ═══════════════════════════════════════════════════════════════════════════
# Backend-compat collectives (gloo lacks all_to_all_single)
# ═══════════════════════════════════════════════════════════════════════════

def all_to_all_single_compat(out, inp, group):
    """out block j on rank r  <-  inp block r on rank j."""
    backend = dist.get_backend(group)
    if backend == "gloo":
        L = dist.get_world_size(group)
        blk = inp.numel() // L
        r = dist.get_rank(group)
        gathered = [torch.empty_like(inp) for _ in range(L)]
        dist.all_gather(gathered, inp.contiguous(), group=group)
        for j in range(L):
            out[j * blk:(j + 1) * blk].copy_(gathered[j][r * blk:(r + 1) * blk])
    else:
        dist.all_to_all_single(out, inp, group=group)


def all_gather_into_tensor_compat(out, inp, group):
    backend = dist.get_backend(group)
    if backend == "gloo":
        L = dist.get_world_size(group)
        n = inp.numel()
        gathered = [torch.empty_like(inp) for _ in range(L)]
        dist.all_gather(gathered, inp.contiguous(), group=group)
        for j in range(L):
            out[j * n:(j + 1) * n].copy_(gathered[j])
    else:
        dist.all_gather_into_tensor(out, inp, group=group)


# ═══════════════════════════════════════════════════════════════════════════
# Topology (mirrors the hook)
# ═══════════════════════════════════════════════════════════════════════════

class Topo:
    def __init__(self, ppn):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.ppn = ppn
        self.num_nodes = self.world_size // ppn
        self.node_id = self.rank // ppn
        self.local_rank = self.rank % ppn
        self.is_leader = (self.local_rank == 0)
        self.leader_ranks = [n * ppn for n in range(self.num_nodes)]
        self.local_group = None
        self.leader_group = None

    def build_groups(self):
        for n in range(self.num_nodes):
            members = list(range(n * self.ppn, (n + 1) * self.ppn))
            g = dist.new_group(ranks=members)
            if self.node_id == n:
                self.local_group = g
        self.leader_group = dist.new_group(ranks=self.leader_ranks)

    def __repr__(self):
        return (f"rank={self.rank}/{self.world_size} ppn={self.ppn} "
                f"nodes={self.num_nodes} node_id={self.node_id} "
                f"local_rank={self.local_rank} leader={self.is_leader} "
                f"leaders={self.leader_ranks}")


# ═══════════════════════════════════════════════════════════════════════════
# The algorithm under test
# ═══════════════════════════════════════════════════════════════════════════

class LeaderBuffers:
    """Preallocated buffers for the leader-group compressed allreduce."""

    def __init__(self, numel, topo, codec, num_chunks, dtype, device):
        L = topo.num_nodes
        gran = L * num_chunks
        self.padded = ((numel + gran - 1) // gran) * gran
        self.shard = self.padded // L
        self.num_chunks = num_chunks
        self.cshard = self.shard // num_chunks
        self.L = L
        self.cb = codec.max_bytes(self.cshard)

        self.buf = torch.zeros(self.padded, dtype=dtype, device=device)
        self.my_pos = topo.node_id
        self.p1_send = [torch.zeros(L * self.cb, dtype=torch.uint8, device=device)
                        for _ in range(num_chunks)]
        self.p1_recv = [torch.zeros(L * self.cb, dtype=torch.uint8, device=device)
                        for _ in range(num_chunks)]
        self.p1_dec = [torch.zeros(L * self.cshard, dtype=dtype, device=device)
                       for _ in range(num_chunks)]
        # Preallocated accumulator (avoids allocating on a side stream).
        self.p1_acc = [torch.zeros(self.cshard, dtype=dtype, device=device)
                       for _ in range(num_chunks)]
        self.p2_send = [torch.zeros(self.cb, dtype=torch.uint8, device=device)
                        for _ in range(num_chunks)]
        self.p2_recv = [torch.zeros(L * self.cb, dtype=torch.uint8, device=device)
                        for _ in range(num_chunks)]

    def my_shard(self):
        return self.buf.narrow(0, self.my_pos * self.shard, self.shard)

    def view2d(self):
        return self.buf.view(self.L, self.shard)


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class Pipe:
    """Stream/event pipeline. streams=None → everything on the current stream
    (simple, obviously-correct reference). streams=(comp,comm,dec) → mirrors the
    hook's exact side-stream + event pattern, so we can tell a stream-ordering
    bug apart from an algorithm bug."""

    def __init__(self, streams, device):
        self.on = streams is not None
        self.device = device
        if self.on:
            self.comp, self.comm, self.dec = streams

    def ctx(self, which):
        if not self.on:
            return _NullCtx()
        return torch.cuda.stream(getattr(self, which))

    def gate(self, src, dst):
        """Make stream `dst` wait for work enqueued on `src`."""
        if not self.on:
            return
        ev = torch.cuda.Event()
        ev.record(getattr(self, src))
        getattr(self, dst).wait_event(ev)

    def open(self):
        if not self.on:
            return
        cur = torch.cuda.current_stream(device=self.device)
        for s in (self.comp, self.comm, self.dec):
            s.wait_stream(cur)

    def close(self):
        if not self.on:
            return
        cur = torch.cuda.current_stream(device=self.device)
        ev = torch.cuda.Event()
        ev.record(self.dec)
        cur.wait_event(ev)


def leader_phase1_(lb, topo, codec, pipe):
    """Reduce-scatter over the leader group. After this, my_shard holds the
    sum over all leaders of that shard region."""
    lg = topo.leader_group
    L, cb, cs = lb.L, lb.cb, lb.cshard
    v = lb.view2d()

    for c in range(lb.num_chunks):
        off = c * cs
        send = lb.p1_send[c]
        with pipe.ctx("comp"):
            for j in range(L):
                codec.compress(v[j].narrow(0, off, cs),
                               send.narrow(0, j * cb, cb))
        pipe.gate("comp", "comm")

        recv = lb.p1_recv[c]
        with pipe.ctx("comm"):
            all_to_all_single_compat(recv, send, lg)
        pipe.gate("comm", "dec")

        dec = lb.p1_dec[c]
        with pipe.ctx("dec"):
            for j in range(L):
                codec.decompress(recv.narrow(0, j * cb, cb), cb,
                                 dec.narrow(0, j * cs, cs))
            acc = lb.p1_acc[c]
            torch.sum(dec.view(L, cs), dim=0, out=acc)
            lb.my_shard().narrow(0, off, cs).copy_(acc)


def leader_phase2_(lb, topo, codec, pipe):
    """All-gather over the leader group. Fills the whole buf with the reduced
    result."""
    lg = topo.leader_group
    L, cb, cs = lb.L, lb.cb, lb.cshard
    v = lb.view2d()

    # Phase 2 reads my_shard, which phase 1 wrote on the dec stream.
    pipe.gate("dec", "comp")

    for c in range(lb.num_chunks):
        off = c * cs
        send = lb.p2_send[c]
        with pipe.ctx("comp"):
            codec.compress(lb.my_shard().narrow(0, off, cs), send)
        pipe.gate("comp", "comm")

        recv = lb.p2_recv[c]
        with pipe.ctx("comm"):
            all_gather_into_tensor_compat(recv, send, lg)
        pipe.gate("comm", "dec")

        with pipe.ctx("dec"):
            for j in range(L):
                codec.decompress(recv.narrow(0, j * cb, cb), cb,
                                 v[j].narrow(0, off, cs))


def hier_allreduce_(tensor, topo, lb, codec, pipe, average=True):
    """Full three-step hierarchical allreduce, in place on `tensor`."""
    flat = tensor.flatten().clone()

    # Step 1: intra-node SUM.
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=topo.local_group)

    # Step 2: inter-node compressed allreduce (leaders only).
    if topo.is_leader:
        lb.buf.zero_()
        lb.buf.narrow(0, 0, flat.numel()).copy_(flat)
        pipe.open()
        leader_phase1_(lb, topo, codec, pipe)
        leader_phase2_(lb, topo, codec, pipe)
        pipe.close()
        flat.copy_(lb.buf.narrow(0, 0, flat.numel()))

    # Step 3: intra-node broadcast from the node leader.
    dist.broadcast(flat, src=topo.node_id * topo.ppn, group=topo.local_group)

    if average:
        flat.div_(topo.world_size)
    tensor.copy_(flat.view_as(tensor))


# ═══════════════════════════════════════════════════════════════════════════
# Test driver
# ═══════════════════════════════════════════════════════════════════════════

def make_vec(numel, rank, dtype, device):
    """Structured so any index/shard permutation shows up as a mismatch."""
    idx = torch.arange(numel, dtype=dtype, device=device)
    return idx + (rank + 1) * 1000.0


_DEVICE = torch.device("cpu")   # set in main(); NCCL cannot reduce CPU tensors


def _barrier():
    """NCCL barrier needs an explicit device or it guesses (and warns)."""
    if dist.get_backend() == "nccl":
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def report(stage, ok, err, extra=""):
    rank = dist.get_rank()
    # Reduce pass/fail across all ranks so one bad rank fails the stage.
    # NOTE: tensors MUST live on _DEVICE. The NCCL/RCCL backend has no CPU
    # implementation and raises "No backend type associated with device type
    # cpu" for host tensors.
    flag = torch.tensor([0.0 if ok else 1.0], device=_DEVICE)
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    emax = torch.tensor([float(err)], device=_DEVICE)
    dist.all_reduce(emax, op=dist.ReduceOp.MAX)
    if rank == 0:
        status = "PASS" if flag.item() == 0 else "FAIL"
        print(f"  [{status}] {stage:<44} max_err={emax.item():.6e} {extra}",
              flush=True)
    return flag.item() == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    ap.add_argument("--fake-ppn", type=int, default=0,
                    help="Override ranks-per-node (simulate multi-node on one box)")
    ap.add_argument("--codec", default="identity", choices=["identity", "zfp"])
    ap.add_argument("--rate", type=float, default=16.0)
    ap.add_argument("--numel", type=int, default=37,
                    help="Small and deliberately not divisible, to test padding")
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--tol", type=float, default=0.0)
    ap.add_argument("--streams", action="store_true",
                    help="Run phases on CUDA side streams with events, exactly "
                         "as the DDP hook does (nccl only). Compare against a "
                         "run without this flag to isolate stream-ordering bugs.")
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "127.0.0.1"))
    os.environ.setdefault("MASTER_PORT", "29577")
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    if args.backend == "nccl":
        local = int(os.environ.get("SLURM_LOCALID",
                                   os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(local % max(1, torch.cuda.device_count()))
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    global _DEVICE
    _DEVICE = device

    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    dtype = torch.float32

    ppn = args.fake_ppn or int(os.environ.get(
        "PPN", os.environ.get("SLURM_NTASKS_PER_NODE", "0")).split("(")[0] or 0)
    if ppn <= 0:
        if rank == 0:
            print("ERROR: could not determine ppn; pass --fake-ppn N", flush=True)
        dist.destroy_process_group()
        sys.exit(2)

    if world % ppn != 0:
        if rank == 0:
            print(f"ERROR: world={world} not divisible by ppn={ppn}", flush=True)
        dist.destroy_process_group()
        sys.exit(2)

    topo = Topo(ppn)
    if rank == 0:
        print("=" * 78)
        print(f"HIER ALLREDUCE ALGORITHM TEST  backend={args.backend} "
              f"codec={args.codec} numel={args.numel} chunks={args.chunks}")
        print(f"world={world} ppn={ppn} nodes={topo.num_nodes} "
              f"leaders={topo.leader_ranks}")
        print("=" * 78)
    _barrier()

    if topo.num_nodes < 2:
        if rank == 0:
            print("ERROR: need >=2 (possibly fake) nodes to exercise the "
                  "hierarchical path. Use --fake-ppn to split.", flush=True)
        dist.destroy_process_group()
        sys.exit(2)

    # ---- S0 topology ----
    print(f"    S0 {topo}", flush=True)
    _barrier()

    # ---- S1 groups ----
    topo.build_groups()
    ok = (topo.local_group is not None and topo.leader_group is not None)
    lsz = dist.get_world_size(topo.local_group)
    report("S1 groups built", ok and lsz == ppn, 0.0,
           f"(local_group size={lsz})")

    codec = (IdentityCodec(dtype, device) if args.codec == "identity"
             else ZfpCodec(dtype, device, args.rate))

    numel = args.numel
    x = make_vec(numel, rank, dtype, device)

    # Reference: what a native allreduce-average produces.
    ref = x.clone()
    dist.all_reduce(ref, op=dist.ReduceOp.SUM)
    ref.div_(world)

    # ---- S2 intra-node allreduce ----
    t = x.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=topo.local_group)
    node_lo = topo.node_id * ppn
    exp = torch.zeros_like(x)
    for r in range(node_lo, node_lo + ppn):
        exp += make_vec(numel, r, dtype, device)
    err = (t - exp).abs().max().item()
    report("S2 intra-node all_reduce (step 1)", err <= 1e-4, err)

    lb = LeaderBuffers(numel, topo, codec, args.chunks, dtype, device)
    if rank == 0:
        print(f"    buffers: padded={lb.padded} shard={lb.shard} "
              f"cshard={lb.cshard} comp_bytes/chunk={lb.cb}", flush=True)

    # ---- S2b codec round-trip, NO communication at all ----
    # If this fails, the codec/rate/size combination is broken and no amount of
    # algorithm debugging will help.
    probe = make_vec(lb.cshard, rank, dtype, device)
    tmp_u8 = torch.zeros(lb.cb, dtype=torch.uint8, device=device)
    back = torch.zeros(lb.cshard, dtype=dtype, device=device)
    codec.compress(probe, tmp_u8)
    codec.decompress(tmp_u8, lb.cb, back)
    if device.type == "cuda":
        torch.cuda.synchronize()
    rt_abs = (back - probe).abs().max().item()
    rt_rel = rt_abs / max(probe.abs().max().item(), 1e-12)
    finite = bool(torch.isfinite(back).all().item())
    report(f"S2b codec round-trip ({codec.name}, no comm)",
           finite and rt_rel <= max(args.tol, 0.5), rt_abs,
           f"rel={rt_rel:.3e} finite={finite} n={lb.cshard} bytes={lb.cb}")

    streams = None
    if args.streams and device.type == "cuda":
        streams = (torch.cuda.Stream(device=device),
                   torch.cuda.Stream(device=device),
                   torch.cuda.Stream(device=device))
    pipe = Pipe(streams, device)
    if rank == 0:
        print(f"    streams: {'ON (hook-identical)' if streams else 'off'}",
              flush=True)

    # ---- S3 leader phase 1 only ----
    # Every leader starts from its node-sum; after phase 1 its own shard must
    # equal the global sum restricted to that shard.
    global_sum = torch.zeros(numel, dtype=dtype, device=device)
    for r in range(world):
        global_sum += make_vec(numel, r, dtype, device)
    gs_pad = torch.zeros(lb.padded, dtype=dtype, device=device)
    gs_pad.narrow(0, 0, numel).copy_(global_sum)

    err3 = 0.0
    if topo.is_leader:
        lb.buf.zero_()
        lb.buf.narrow(0, 0, numel).copy_(t)   # t == node sum from S2
        pipe.open()
        leader_phase1_(lb, topo, codec, pipe)
        pipe.close()
        if device.type == "cuda":
            torch.cuda.synchronize()
        expected_shard = gs_pad.narrow(0, lb.my_pos * lb.shard, lb.shard)
        err3 = (lb.my_shard() - expected_shard).abs().max().item()
    report("S3 leader phase 1 (reduce-scatter)", err3 <= max(args.tol, 1e-3), err3)

    # ---- S4 leader phase 1 + 2 ----
    err4 = 0.0
    if topo.is_leader:
        pipe.open()
        leader_phase2_(lb, topo, codec, pipe)
        pipe.close()
        if device.type == "cuda":
            torch.cuda.synchronize()
        err4 = (lb.buf - gs_pad).abs().max().item()
    report("S4 leader phase 1+2 (full leader AR)", err4 <= max(args.tol, 1e-3), err4)

    # ---- S5 full hierarchical vs native ----
    y = x.clone()
    hier_allreduce_(y, topo, lb, codec, pipe, average=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    err5 = (y - ref).abs().max().item()
    rel = err5 / max(ref.abs().max().item(), 1e-12)
    report("S5 full hierarchical vs native allreduce",
           err5 <= max(args.tol, 1e-3), err5, f"rel={rel:.3e}")

    _barrier()
    if rank == 0:
        print("=" * 78)
        print("done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)