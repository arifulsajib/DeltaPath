from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm, layer_norm_fn

__all__ = ["Block", "GatedMLP", "init_weights"]


class GatedMLP(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 activation=F.silu, bias=False, multiple_of=128, device=None, dtype=None):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else int(8 * in_features / 3)
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias, **factory)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        return self.fc2(y * self.activation(gate))


class Block(nn.Module):

    def __init__(self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm,
                 fused_add_norm=False, residual_in_fp32=False):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None

    def forward(self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None,
                inference_params=None, **mixer_kwargs):
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states, self.norm.weight, self.norm.bias,
                residual=residual, prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps, is_rms_norm=isinstance(self.norm, RMSNorm),
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states, self.norm2.weight, self.norm2.bias,
                    residual=residual, prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps, is_rms_norm=isinstance(self.norm2, RMSNorm),
                )
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True,
                 n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Scale residual-projection weights by 1/sqrt(N_residual_layers) (GPT-2 style).
        for name, p in module.named_parameters():
            if name in ("out_proj.weight", "fc2.weight", "R_out_proj.weight"):
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)
