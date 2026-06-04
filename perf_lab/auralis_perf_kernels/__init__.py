"""Experimental Auralis performance kernels.

These wrappers are intentionally outside `src/auralis` until they prove faster
and numerically safe.
"""

from .chunked_ce import (
    chunked_linear_cross_entropy,
    triton_forward_linear_cross_entropy,
    triton_fused_linear_cross_entropy,
)
from .liger_ce import liger_available, liger_linear_cross_entropy
from .rmsnorm import rmsnorm
from .triton_ce import (
    triton_linear_ce_backward,
    triton_linear_ce_backward_atomic_lowmem,
    triton_linear_ce_backward_atomic_mixed,
    triton_linear_ce_backward_partial_weight,
    triton_linear_ce_backward_split_hidden,
    triton_linear_ce_backward_split_weight,
    triton_linear_ce_forward,
)

__all__ = [
    "chunked_linear_cross_entropy",
    "liger_available",
    "liger_linear_cross_entropy",
    "rmsnorm",
    "triton_forward_linear_cross_entropy",
    "triton_fused_linear_cross_entropy",
    "triton_linear_ce_backward",
    "triton_linear_ce_backward_atomic_lowmem",
    "triton_linear_ce_backward_atomic_mixed",
    "triton_linear_ce_backward_partial_weight",
    "triton_linear_ce_backward_split_hidden",
    "triton_linear_ce_backward_split_weight",
    "triton_linear_ce_forward",
]
