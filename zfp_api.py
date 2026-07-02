"""
Thin Python wrapper for ZFP compression extension.

Selects CUDA or HIP backend automatically depending on the PyTorch build
and/or device type.
"""

from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Optional
import torch

@dataclass(frozen=True)
class ZfpCompressionConfig:
    rate: float = float(os.getenv("DDP_ZFP_RATE", "8.0"))

def _load_ext():
    # If you want an override knob:
    backend = os.getenv("DDP_ZFP_BACKEND", "auto").lower()

    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    if backend == "auto":
        backend = "hip" if is_rocm else "cuda"

    errors = []

    if backend == "hip":
        try:
            import framework_allreduce_zfp_hip as ext
            return ext
        except Exception as e:
            errors.append(("framework_allreduce_zfp_hip", e))

    if backend == "cuda":
        try:
            import framework_allreduce_zfp_cuda as ext
            return ext
        except Exception as e:
            errors.append(("framework_allreduce_zfp_cuda", e))

    # Fallback: try both if override was wrong / auto misdetected
    for name in ("framework_allreduce_zfp_hip", "framework_allreduce_zfp_cuda"):
        try:
            ext = __import__(name)
            return ext
        except Exception as e:
            errors.append((name, e))

    msg = ["Could not import any ZFP extension backend."]
    msg.append(f"torch.version.cuda={getattr(torch.version,'cuda',None)} "
               f"torch.version.hip={getattr(torch.version,'hip',None)} "
               f"DDP_ZFP_BACKEND={os.getenv('DDP_ZFP_BACKEND')!r}")
    msg.append("Tried:")
    for name, e in errors:
        msg.append(f"  - {name}: {type(e).__name__}: {e}")
    raise ImportError("\n".join(msg))

_EXT = _load_ext()

def max_output_bytes(tensor: torch.Tensor, rate: Optional[float] = None) -> int:
    return int(_EXT.max_output_bytes(tensor.contiguous(), float(rate or ZfpCompressionConfig().rate)))

def compress_into(src: torch.Tensor, dst: torch.Tensor, rate: Optional[float] = None) -> int:
    return int(_EXT.compress_into(src.contiguous(), dst, float(rate or ZfpCompressionConfig().rate)))

def decompress_into(src: torch.Tensor, used_bytes: int, dst: torch.Tensor, rate: Optional[float] = None) -> None:
    _EXT.decompress_into(src, int(used_bytes), dst, float(rate or ZfpCompressionConfig().rate))

def compress_into_current_stream(src: torch.Tensor, dst: torch.Tensor, rate: Optional[float] = None) -> int:
    return int(_EXT.compress_into_current_stream(src.contiguous(), dst, float(rate or ZfpCompressionConfig().rate)))

def decompress_into_current_stream(src: torch.Tensor, used_bytes: int, dst: torch.Tensor, rate: Optional[float] = None) -> None:
    _EXT.decompress_into_current_stream(src, int(used_bytes), dst, float(rate or ZfpCompressionConfig().rate))
