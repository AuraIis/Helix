"""Mamba-2 layer with optional CUDA kernel from ``mamba_ssm``.

Two back-ends, same ``forward(x) -> (out, state)`` contract:

- **native** (default on CPU / when ``AURALIS_USE_CUDA_KERNELS != "1"``):
  pure-PyTorch selective-scan reference. Slow, correct, portable.
- **mamba_ssm** (auto-selected when env flag is on and CUDA is available):
  wraps ``mamba_ssm.Mamba2`` which uses the official CUDA kernel
  (parallel selective-scan) and optional ``causal-conv1d``.

The parameter layout differs between the two back-ends — swap is only safe
**before training** (fresh init). If you trained with one back-end, reload
with the same one.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional CUDA back-end
try:
    from mamba_ssm import Mamba2 as _Mamba2SSM
    _MAMBA_SSM_AVAILABLE = True
except Exception:
    _Mamba2SSM = None
    _MAMBA_SSM_AVAILABLE = False


def _cuda_kernels_enabled() -> bool:
    """Per-layer opt-in. AURALIS_USE_MAMBA_KERNEL=1 enables mamba_ssm.
    AURALIS_USE_CUDA_KERNELS=1 enables ALL kernel back-ends at once.

    Note: on Blackwell (sm_120) Triton inside mamba_ssm sometimes fails to
    compile with cu128. Leave AURALIS_USE_MAMBA_KERNEL unset on Blackwell
    until upstream catches up.
    """
    if os.environ.get("AURALIS_USE_MAMBA_KERNEL", "") == "1":
        return True
    return os.environ.get("AURALIS_USE_CUDA_KERNELS", "0") == "1"


# ---------------------------------------------------------------------------
# Pure-PyTorch reference implementation (original)
# ---------------------------------------------------------------------------
class _Mamba2Native(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand_factor: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.d_inner = expand_factor * d_model
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.reset_special_parameters()

    def reset_special_parameters(self) -> None:
        """Restore Mamba-specific parameters that generic init must not zero.

        The model-wide initializer touches every ``nn.Linear`` and zeros its
        bias. ``dt_proj.bias`` is not a normal bias: it is the inverse-softplus
        parameterisation for the SSM time step, so it must be reset after the
        generic init pass.
        """
        with torch.no_grad():
            log_min = torch.log(torch.tensor(self.dt_min, dtype=torch.float32))
            log_max = torch.log(torch.tensor(self.dt_max, dtype=torch.float32))
            dt = torch.exp(torch.rand(self.d_inner, device=self.dt_proj.bias.device) * (log_max - log_min) + log_min)
            # inverse softplus: softplus(bias) == dt
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x, ssm_state=None):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_in = x_in.transpose(1, 2)
        x_in = self.conv1d(x_in)[..., :L]
        x_in = x_in.transpose(1, 2)
        x_in = F.silu(x_in)
        x_dbl = self.x_proj(x_in)
        dt, B_ssm, C_ssm = x_dbl.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        y, new_state = self._selective_scan(x_in, dt, A, B_ssm, C_ssm, self.D, ssm_state)
        y = y * F.silu(z)
        return self.out_proj(y), new_state

    def _selective_scan(self, x, dt, A, Bp, Cp, D, state):
        B, L, d_inner = x.shape
        d_state = A.shape[1]
        dtype = x.dtype
        dA = torch.exp(dt.unsqueeze(-1) * A)
        dB = dt.unsqueeze(-1) * Bp.unsqueeze(-2)
        h = (torch.zeros(B, d_inner, d_state, device=x.device, dtype=dtype)
             if state is None else state.to(dtype))
        outputs = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            outputs.append((h * Cp[:, t].unsqueeze(-2)).sum(dim=-1))
        y = torch.stack(outputs, dim=1)
        y = y + x * D
        return y, h


# ---------------------------------------------------------------------------
# mamba_ssm.Mamba2 wrapper (CUDA)
# ---------------------------------------------------------------------------
class _Mamba2CUDA(nn.Module):
    """Thin wrapper over ``mamba_ssm.Mamba2`` exposing our ``(out, state)`` API."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand_factor: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        assert _Mamba2SSM is not None
        # mamba_ssm has strict shape constraints. d_model*expand must be
        # divisible by its internal "headdim" (default 64). For the
        # configs we ship (d_model in {512, 768, 1280}, expand=2) that is
        # always satisfied.
        self.inner = _Mamba2SSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand_factor,
        )

    def forward(self, x, ssm_state=None):
        # mamba_ssm Mamba2 returns just the output tensor; no explicit state
        # is exposed through the simple forward path we use for training.
        out = self.inner(x)
        return out, ssm_state


# ---------------------------------------------------------------------------
# Public Mamba2Layer: picks back-end at construction time
# ---------------------------------------------------------------------------
class Mamba2Layer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand_factor: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        use_cuda = _cuda_kernels_enabled() and _MAMBA_SSM_AVAILABLE and torch.cuda.is_available()
        self.backend = "mamba_ssm" if use_cuda else "native"
        impl = _Mamba2CUDA if use_cuda else _Mamba2Native
        self._impl = impl(
            d_model=d_model, d_state=d_state, d_conv=d_conv,
            expand_factor=expand_factor, dt_min=dt_min, dt_max=dt_max,
        )

    @property
    def out_proj(self) -> nn.Module:
        """Expose out_proj so the scaled-output init finds it in both back-ends."""
        inner = self._impl
        # Native has a direct out_proj. mamba_ssm Mamba2 nests it at .inner.out_proj.
        if hasattr(inner, "out_proj"):
            return inner.out_proj
        if hasattr(inner, "inner") and hasattr(inner.inner, "out_proj"):
            return inner.inner.out_proj
        raise AttributeError("no out_proj")

    def forward(self, x, ssm_state=None):
        return self._impl(x, ssm_state)

    def reset_special_parameters(self) -> None:
        reset = getattr(self._impl, "reset_special_parameters", None)
        if callable(reset):
            reset()


__all__ = ["Mamba2Layer"]
