"""Thin Python wrapper for cuZFP.

This module expects a prebuilt PyTorch C++/CUDA extension named
`framework_allreduce_zfp_cuda` to be installed in the current environment.

Do NOT JIT-compile (torch.utils.cpp_extension.load) inside multi-rank training:
it races across ranks and often fails on clusters.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

import torch

@dataclass(frozen=True)
class ZfpCompressionConfig:
    # Official ZFP CUDA support is fixed-rate only.
    rate: float = float(os.getenv("DDP_ZFP_RATE", "8.0"))

try:
    import framework_allreduce_zfp_cuda as _EXT
except ImportError as e:
    raise ImportError(
        "Could not import `framework_allreduce_zfp_cuda`.\n\n"
        "You must build/install the extension once in this conda env (outside the "
        "distributed training run), e.g.:\n\n"
        "  python -m pip install -v --no-deps /path/to/extension_dir\n\n"
        "After that, re-run training and this module will just import the .so."
    ) from e

def max_output_bytes(tensor: torch.Tensor, rate: Optional[float] = None) -> int:
    return int(_EXT.max_output_bytes(
        tensor.contiguous(),
        float(rate or ZfpCompressionConfig().rate),
    ))

def compress_into(
    src: torch.Tensor,
    dst: torch.Tensor,
    rate: Optional[float] = None,
) -> int:
    """Original cuZFP path used by the naive ZFP hooks."""
    return int(_EXT.compress_into(
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
    _EXT.decompress_into(
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
    return int(_EXT.compress_into_current_stream(
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
    _EXT.decompress_into_current_stream(
        src,
        int(used_bytes),
        dst,
        float(rate or ZfpCompressionConfig().rate),
    )
