"""Thin Python wrapper for the official ZFP CUDA backend used as "cuZFP".

This module builds a small PyTorch C++ extension that calls libzfp's high-level
API with the CUDA execution policy `zfp_exec_cuda` and fixed-rate mode.
That is the official GPU path documented by ZFP and is the path the paper
refers to as cuZFP.

The extension operates directly on CUDA tensors so compressed streams can live
on device memory and be exchanged by DDP P2P ops without host staging.

Requirements at runtime:
  - PyTorch with CUDA
  - libzfp built with CUDA support (`ZFP_WITH_CUDA=ON`)
  - `ZFP_HOME` pointing at the ZFP install prefix, or include/lib discoverable
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
        include_dirs.append(str(Path(zfp_home) / "include"))
        extra_ldflags.extend([
            f"-L{Path(zfp_home) / 'lib'}",
            f"-Wl,-rpath,{Path(zfp_home) / 'lib'}",
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
    return int(ext.max_output_bytes(tensor.contiguous(), float(rate or ZfpCompressionConfig().rate)))


def compress_into(
    src: torch.Tensor,
    dst: torch.Tensor,
    rate: Optional[float] = None,
) -> int:
    ext = _load_extension()
    return int(ext.compress_into(src.contiguous(), dst, float(rate or ZfpCompressionConfig().rate)))


def decompress_into(
    src: torch.Tensor,
    used_bytes: int,
    dst: torch.Tensor,
    rate: Optional[float] = None,
) -> None:
    ext = _load_extension()
    ext.decompress_into(src, int(used_bytes), dst, float(rate or ZfpCompressionConfig().rate))
