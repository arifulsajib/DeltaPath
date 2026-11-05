
from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .gated_deltanet2 import causal_conv1d, _init_dt_bias
from .ops import chunk_gated_delta_rule, recurrent_gated_delta_rule

__all__ = ["LightGatedDeltaNet2"]


class LightGatedDeltaNet2(nn.Module):

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, headdim=None,
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, A_init_range=(1, 16),
                 conv_bias=True, bias=False, chunk_size=64, use_qk_l2norm=True,
                 beta_init=1.0, device=None, dtype=None, **unused):
        factory = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.headdim = headdim or self.d_inner
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.chunk_size = chunk_size
        self.use_qk_l2norm = use_qk_l2norm

        # [z, x, B, C, dt, beta]
        d_in_proj = 2 * self.d_inner + 2 * self.d_state + 2 * self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=bias, **factory)
        conv_dim = self.d_inner + 2 * self.d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=d_conv, groups=conv_dim,
                                padding=d_conv - 1, bias=conv_bias, **factory)

        self.dt_bias = nn.Parameter(_init_dt_bias(self.nheads, dt_min, dt_max, dt_init_floor, **factory))
        self.dt_bias._no_weight_decay = True
        self.beta_bias = nn.Parameter(torch.full((self.nheads,), float(beta_init), device=device))
        self.beta_bias._no_weight_decay = True
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A).to(dtype=dtype))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory)

    def forward(self, x, padding_mask=None, mode="chunk", **kwargs):
        zxbcdt = self.in_proj(x)
        z, xBC, dt, beta_raw = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads, self.nheads], dim=-1
        )
        if padding_mask is not None:
            xBC = xBC.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        xBC = causal_conv1d(xBC, self.conv1d)
        v_flat, B, C = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)

        A = -torch.exp(self.A_log.float())
        g = F.softplus(dt + self.dt_bias) * A
        beta = torch.sigmoid(beta_raw + self.beta_bias)
        v = rearrange(v_flat, "b l (h p) -> b l h p", p=self.headdim)
        if padding_mask is not None:
            pm = padding_mask.unsqueeze(-1)
            g = g.masked_fill(pm, 0.0)
            beta = beta.masked_fill(pm, 0.0)
            v = v.masked_fill(pm.unsqueeze(-1), 0.0)

        kwargs_ = dict(q=C.unsqueeze(2), k=B.unsqueeze(2), v=v, g=g, beta=beta,
                       initial_state=None, output_final_state=False,
                       use_qk_l2norm=self.use_qk_l2norm)
        if mode == "chunk":
            y, _ = chunk_gated_delta_rule(chunk_size=self.chunk_size, **kwargs_)
        else:
            y, _ = recurrent_gated_delta_rule(**kwargs_)
        y = y + self.D.unsqueeze(-1) * v
        y = rearrange(y.to(x.dtype), "b l h p -> b l (h p)")
        return self.out_proj(y * F.silu(z))
