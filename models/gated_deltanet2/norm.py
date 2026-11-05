
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RMSNorm", "RMSNormGated", "layer_norm_fn", "TRITON_NORM_AVAILABLE"]

try:  # pragma: no cover - optional acceleration
    from mamba_ssm.ops.triton.layer_norm import layer_norm_fn as _triton_layer_norm_fn

    TRITON_NORM_AVAILABLE = True
except Exception:  # pragma: no cover
    _triton_layer_norm_fn = None
    TRITON_NORM_AVAILABLE = False


class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype) * self.weight).to(dtype)

    def extra_repr(self) -> str:
        return f"{self.weight.numel()}, eps={self.eps}"


class RMSNormGated(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        norm_before_gate: bool = False,
        group_size: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.group_size = group_size if group_size is not None else hidden_size
        assert hidden_size % self.group_size == 0
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, self.group_size).float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.reshape(shape)

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        if z is not None and not self.norm_before_gate:
            x = x * F.silu(z.float())
        out = self._rms(x).to(dtype) * self.weight
        if z is not None and self.norm_before_gate:
            out = out * F.silu(z)
        return out


def layer_norm_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    is_rms_norm: bool = False,
):
    if _triton_layer_norm_fn is not None and x.is_cuda:  # pragma: no cover
        return _triton_layer_norm_fn(
            x, weight, bias, residual=residual, eps=eps, prenorm=prenorm,
            residual_in_fp32=residual_in_fp32, is_rms_norm=is_rms_norm,
        )

    dtype = x.dtype
    if residual is not None:
        x = x + residual
    residual_out = x.to(torch.float32) if residual_in_fp32 else x

    xf = x.float()
    if is_rms_norm:
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    else:
        xf = (xf - xf.mean(-1, keepdim=True)) * torch.rsqrt(xf.var(-1, keepdim=True, unbiased=False) + eps)
    out = xf.to(dtype) * weight
    if bias is not None:
        out = out + bias
    return (out, residual_out) if prenorm else out
