"""Thin Python wrapper for cuZFP.

This module builds a small PyTorch C++ extension that calls libzfp's CUDA
backend in two ways:

1. compress_into / decompress_into
   Original high-level cuZFP path. Kept for the already-tested naive hooks.

2. compress_into_current_stream / decompress_into_current_stream
   Stream-aware path using the modified ZFP CUDA entry points. This is used by
   the paper-style online compression hooks so compression/decompression kernels
   launch on PyTorch's current CUDA stream.

The extension operates directly on CUDA tensors so compressed streams can live
on device memory and be exchanged by DDP P2P ops without host staging.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Optional

import torch
from torch.utils.cpp_extension import load


@dataclass(frozen=True)
class ZfpCompressionConfig:
    # Official ZFP CUDA support is fixed-rate only.
    rate: float = float(os.getenv("DDP_ZFP_RATE", "8.0"))


_EXT = None


def _load_extension():
    global _EXT
    if _EXT is not None:
        return _EXT

    here = Path(__file__).resolve().parent
    src = here / "zfp_cuda_extension.cpp"

    zfp_home = os.getenv("ZFP_HOME")
    include_dirs = []
    extra_ldflags = []

    if zfp_home:
        lib64 = Path(zfp_home) / "lib64"
        lib = Path(zfp_home) / "lib"
        lib_dir = lib64 if lib64.exists() else lib

        include_dirs.append(str(Path(zfp_home) / "include"))
        extra_ldflags.extend([
            f"-L{lib_dir}",
            f"-Wl,-rpath,{lib_dir}",
            "-lzfp",
        ])
    else:
        extra_ldflags.append("-lzfp")

    _EXT = load(
        name="framework_allreduce_zfp_cuda",
        sources=[str(src)],
        extra_include_paths=include_dirs,
        extra_cflags=["-O3"],
        extra_ldflags=extra_ldflags,
        verbose=False,
    )
    return _EXT


def max_output_bytes(tensor: torch.Tensor, rate: Optional[float] = None) -> int:
    ext = _load_extension()
    return int(ext.max_output_bytes(
        tensor.contiguous(),
        float(rate or ZfpCompressionConfig().rate),
    ))


def compress_into(
    src: torch.Tensor,
    dst: torch.Tensor,
    rate: Optional[float] = None,
) -> int:
    """Original cuZFP path used by the naive ZFP hooks."""
    ext = _load_extension()
    return int(ext.compress_into(
        src.contiguous(),
        dst,
        float(rate or ZfpCompressionConfig().rate),
    ))


def decompress_into(
    src: torch.Tensor,
    used_bytes: int,
    dst: torch.Tensor,
    rate: Optional[float] = None,
) -> None:
    """Original cuZFP path used by the naive ZFP hooks."""
    ext = _load_extension()
    ext.decompress_into(
        src,
        int(used_bytes),
        dst,
        float(rate or ZfpCompressionConfig().rate),
    )


def compress_into_current_stream(
    src: torch.Tensor,
    dst: torch.Tensor,
    rate: Optional[float] = None,
) -> int:
    """Stream-aware path used by online compression hooks."""
    ext = _load_extension()
    return int(ext.compress_into_current_stream(
        src.contiguous(),
        dst,
        float(rate or ZfpCompressionConfig().rate),
    ))


def decompress_into_current_stream(
    src: torch.Tensor,
    used_bytes: int,
    dst: torch.Tensor,
    rate: Optional[float] = None,
) -> None:
    """Stream-aware path used by online compression hooks."""
    ext = _load_extension()
    ext.decompress_into_current_stream(
        src,
        int(used_bytes),
        dst,
        float(rate or ZfpCompressionConfig().rate),
    )