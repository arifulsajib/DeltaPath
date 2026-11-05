from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .norm import RMSNormGated
from .ops import chunk_gated_delta_rule, recurrent_gated_delta_rule

__all__ = ["GatedDeltaNet2", "causal_conv1d"]


def causal_conv1d(x: torch.Tensor, conv: nn.Conv1d, activation: str = "silu") -> torch.Tensor:
    d_conv = conv.kernel_size[0]
    y = conv(x.transpose(1, 2)).transpose(1, 2)
    if d_conv > 1:
        y = y[:, : -(d_conv - 1)]
    return F.silu(y) if activation in ("silu", "swish") else y


class GatedDeltaNet2(nn.Module):

    def __init__(
        self,
        d_model,
        d_inner=0,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        chunk_size=64,
        use_qk_l2norm=True,
        beta_init=1.0,
        layer_idx=None,
        device=None,
        dtype=None,
        **unused,
    ):
        factory = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = d_inner if d_inner else self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm
        self.ngroups = ngroups
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_qk_l2norm = use_qk_l2norm
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt, beta]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + 2 * self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim, out_channels=conv_dim, bias=conv_bias,
            kernel_size=d_conv, groups=conv_dim, padding=d_conv - 1, **factory,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        self.act = nn.SiLU()

        self.dt_bias = nn.Parameter(_init_dt_bias(self.nheads, dt_min, dt_max, dt_init_floor, **factory))
        self.dt_bias._no_weight_decay = True
        self.beta_bias = nn.Parameter(torch.full((self.nheads,), float(beta_init), **_f32(factory)))
        self.beta_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A).to(dtype=dtype))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory)

    # ------------------------------------------------------------------ #
    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None,
                inference_params=None, padding_mask=None, mode="chunk", **kwargs):
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, _ = u.shape
        else:
            batch = u.shape[0] // seqlen
            u = rearrange(u, "(b l) d -> b l d", l=seqlen)

        zxbcdt = self.in_proj(u)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - 2 * self.nheads) // 2
        z0, x0, z, xBC, dt, beta_raw = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state,
             self.nheads, self.nheads],
            dim=-1,
        )
        if padding_mask is not None:
            xBC = xBC.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        xBC = causal_conv1d(xBC, self.conv1d, self.activation)
        x, B, C = torch.split(
            xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1
        )

        y = self._scan(x, B, C, dt, beta_raw, padding_mask, mode)
        y = self.norm(y, z) if self.rmsnorm else y * F.silu(z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        if seqlen_og is not None:
            y = rearrange(y, "b l d -> (b l) d")
        return self.out_proj(y)

    # ------------------------------------------------------------------ #
    def _scan(self, x, B, C, dt, beta_raw, padding_mask=None, mode="chunk"):
        A = -torch.exp(self.A_log.float())                       # (H,), < 0
        dt = F.softplus(dt + self.dt_bias)
        if self.dt_limit != (0.0, float("inf")):
            dt = dt.clamp(self.dt_limit[0], self.dt_limit[1])
        g = dt * A                                               # log gate <= 0
        beta = torch.sigmoid(beta_raw + self.beta_bias)

        v = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        if padding_mask is not None:
            pm = padding_mask.unsqueeze(-1)
            g = g.masked_fill(pm, 0.0)
            beta = beta.masked_fill(pm, 0.0)
            v = v.masked_fill(pm.unsqueeze(-1), 0.0)

        kwargs = dict(
            q=rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            k=rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            v=v, g=g, beta=beta,
            initial_state=None, output_final_state=False,
            use_qk_l2norm=self.use_qk_l2norm,
        )
        if mode == "chunk":
            y, _ = chunk_gated_delta_rule(chunk_size=self.chunk_size, **kwargs)
        else:
            y, _ = recurrent_gated_delta_rule(**kwargs)
        D = rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D.unsqueeze(-1)
        y = y + D * v
        return rearrange(y.to(x.dtype), "b l h p -> b l (h p)")

    # ------------------------------------------------------------------ #
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        dtype = dtype or self.out_proj.weight.dtype
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        conv_state = torch.zeros(batch_size, conv_dim, self.d_conv, device=device, dtype=dtype)
        ssm_state = torch.zeros(batch_size, self.nheads, self.d_state, self.headdim,
                                device=device, dtype=dtype)
        return conv_state, ssm_state

    @torch.no_grad()
    def step(self, u, conv_state, ssm_state):
        zxbcdt = self.in_proj(u.squeeze(1))
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - 2 * self.nheads) // 2
        z0, x0, z, xBC, dt, beta_raw = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state,
             self.nheads, self.nheads], dim=-1,
        )
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = xBC
        xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
        if self.conv1d.bias is not None:
            xBC = xBC + self.conv1d.bias
        xBC = self.act(xBC)
        x, B, C = torch.split(
            xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1
        )

        A = -torch.exp(self.A_log.float())
        dt = F.softplus(dt + self.dt_bias)
        alpha = torch.exp(dt * A).unsqueeze(-1).unsqueeze(-1)            # (B,H,1,1)
        beta = torch.sigmoid(beta_raw + self.beta_bias).unsqueeze(-1)    # (B,H,1)

        from .ops import _expand_groups, _l2norm
        k = _expand_groups(rearrange(B, "b (g n) -> b 1 g n", g=self.ngroups), self.nheads)[:, 0]
        q = _expand_groups(rearrange(C, "b (g n) -> b 1 g n", g=self.ngroups), self.nheads)[:, 0]
        if self.use_qk_l2norm:
            k, q = _l2norm(k), _l2norm(q)
        v = rearrange(x, "b (h p) -> b h p", p=self.headdim)

        pred = torch.einsum("bhn,bhnp->bhp", k, ssm_state)
        delta = beta * (v - alpha.squeeze(-1) * pred)
        ssm_state.copy_(alpha * ssm_state + torch.einsum("bhn,bhp->bhnp", k, delta))
        y = torch.einsum("bhn,bhnp->bhp", q, ssm_state)
        D = rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D.unsqueeze(-1)
        y = rearrange(y + D * v, "b h p -> b (h p)")

        y = self.norm(y, z) if self.rmsnorm else y * F.silu(z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        return self.out_proj(y).unsqueeze(1), conv_state, ssm_state


# --------------------------------------------------------------------------- #
def _f32(factory):
    f = dict(factory)
    f["dtype"] = torch.float32
    return f


def _init_dt_bias(nheads, dt_min, dt_max, dt_init_floor, device=None, dtype=None):
    dt = torch.exp(
        torch.rand(nheads, device=device, dtype=dtype) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    )
    dt = torch.clamp(dt, min=dt_init_floor)
    return dt + torch.log(-torch.expm1(-dt))  # inverse softplus
