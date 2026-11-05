from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .gated_deltanet2 import causal_conv1d, _init_dt_bias
from .norm import RMSNormGated
from .ops import chunk_gated_delta_rule, recurrent_gated_delta_rule

__all__ = ["DeltaPathBlock"]


class DeltaPathBlock(nn.Module):

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
        aux_feature_size=0,
        use_S4_bcdt=False,       # input-independent (S4-style) B, C, Delta
        add_bcdt_item=True,      # re-introduce an input-independent additive term
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

        self.use_S4_bcdt = use_S4_bcdt
        # ``no_gen_bcdt``: B/C/Delta/beta are *not* generated from the token
        # stream itself -- they either arrive from outside (movement features)
        # or are input-independent learnable vectors (S4 form).
        self.no_gen_bcdt = bool(aux_feature_size) or (aux_feature_size == 0 and use_S4_bcdt)
        self.add_bcdt_item = add_bcdt_item

        self.B = self.C = self.dt = self.beta = None
        self.R_B = self.R_C = self.R_dt = self.R_beta = None
        if self.no_gen_bcdt and (self.use_S4_bcdt or self.add_bcdt_item):
            init_BC = torch.randn(4, self.ngroups * self.d_state, dtype=torch.float32, device=device)
            init_dt = torch.randn(2, self.nheads, dtype=torch.float32, device=device)
            init_beta = torch.randn(2, self.nheads, dtype=torch.float32, device=device)
            for name, value in (
                ("B", init_BC[0]), ("C", init_BC[1]), ("dt", init_dt[0]), ("beta", init_beta[0]),
                ("R_B", init_BC[2]), ("R_C", init_BC[3]), ("R_dt", init_dt[1]), ("R_beta", init_beta[1]),
            ):
                p = nn.Parameter(value.clone())
                p._no_weight_decay = True
                setattr(self, name, p)

        # Order: [z, x, (B, C, dt, beta)]
        extra = 0 if self.no_gen_bcdt else 2 * self.ngroups * self.d_state + 2 * self.nheads
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner + extra, bias=bias, **factory)

        conv_dim = self.d_ssm if self.no_gen_bcdt else self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim, out_channels=conv_dim, bias=conv_bias,
            kernel_size=d_conv, groups=conv_dim, padding=d_conv - 1, **factory,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        self.act = nn.SiLU()

        # ---------------- GPS-DeltaNet ---------------- #
        self.dt_bias = nn.Parameter(_init_dt_bias(self.nheads, dt_min, dt_max, dt_init_floor, **factory))
        self.dt_bias._no_weight_decay = True
        self.beta_bias = nn.Parameter(torch.full((self.nheads,), float(beta_init), device=device))
        self.beta_bias._no_weight_decay = True
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A).to(dtype=dtype))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        # ---------------- Road-DeltaNet ---------------- #
        # B', C', Delta', beta' = Param(Y^G)
        self.R_bcdt_proj = nn.Linear(
            self.d_ssm, 2 * self.ngroups * self.d_state + 2 * self.nheads, bias=bias, **factory
        )
        self.R_dt_bias = nn.Parameter(_init_dt_bias(self.nheads, dt_min, dt_max, dt_init_floor, **factory))
        self.R_dt_bias._no_weight_decay = True
        self.R_beta_bias = nn.Parameter(torch.full((self.nheads,), float(beta_init), device=device))
        self.R_beta_bias._no_weight_decay = True
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        self.R_A_log = nn.Parameter(torch.log(A).to(dtype=dtype))
        self.R_A_log._no_weight_decay = True
        self.R_D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.R_D._no_weight_decay = True

        if self.rmsnorm:
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory)

        assert d_model % 2 == 0, "d_model must be even: the two branches own one half each"
        self.out_proj = nn.Linear(self.d_inner, self.d_model // 2, bias=bias, **factory)
        self.R_out_proj = nn.Linear(self.d_inner, self.d_model // 2, bias=bias, **factory)

    # ------------------------------------------------------------------ #
    def _delta_scan(self, x, B, C, dt, beta_raw, A_log, dt_bias, beta_bias, D,
                    padding_mask=None, mode="chunk"):
        A = -torch.exp(A_log.float())
        dt = F.softplus(dt + dt_bias)
        if self.dt_limit != (0.0, float("inf")):
            dt = dt.clamp(self.dt_limit[0], self.dt_limit[1])
        g = dt * A
        beta = torch.sigmoid(beta_raw + beta_bias)
        v = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        if padding_mask is not None:
            pm = padding_mask.unsqueeze(-1)
            g = g.masked_fill(pm, 0.0)
            beta = beta.masked_fill(pm, 0.0)
            v = v.masked_fill(pm.unsqueeze(-1), 0.0)

        kwargs = dict(
            q=rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            k=rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            v=v, g=g, beta=beta, initial_state=None, output_final_state=False,
            use_qk_l2norm=self.use_qk_l2norm,
        )
        if mode == "chunk":
            y, _ = chunk_gated_delta_rule(chunk_size=self.chunk_size, **kwargs)
        else:
            y, _ = recurrent_gated_delta_rule(**kwargs)
        Dp = rearrange(D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else D.unsqueeze(-1)
        y = y + Dp * v
        return rearrange(y.to(x.dtype), "b l h p -> b l (h p)")

    # ------------------------------------------------------------------ #
    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None,
                B=None, C=None, dt=None, beta=None, padding_mask=None, mode="chunk", **kwargs):
        assert not (B is None and self.no_gen_bcdt and not self.use_S4_bcdt), (
            "GPS-DeltaNet parameters are neither generated nor supplied"
        )
        assert not (B is not None and not self.no_gen_bcdt), (
            "this block generates its own B/C/Delta; do not pass them in"
        )

        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, _ = u.shape
        else:
            batch = u.shape[0] // seqlen
            u = rearrange(u, "(b l) d -> b l d", l=seqlen)
        zxbcdt = self.in_proj(u)

        if B is not None and self.add_bcdt_item:
            B, C, dt = B + self.B, C + self.C, dt + self.dt
            beta = beta + self.beta if beta is not None else repeat(self.beta, "h -> b l h", b=batch, l=seqlen)
        elif B is None and self.no_gen_bcdt:  # S4 form: input-independent parameters
            B = repeat(self.B, "n -> b l n", b=batch, l=seqlen).float()
            C = repeat(self.C, "n -> b l n", b=batch, l=seqlen).float()
            dt = repeat(self.dt, "h -> b l h", b=batch, l=seqlen).float()
            beta = repeat(self.beta, "h -> b l h", b=batch, l=seqlen).float()

        if self.no_gen_bcdt:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm) // 2
            z0, x0, z, xBC = torch.split(zxbcdt, [d_mlp, d_mlp, self.d_ssm, self.d_ssm], dim=-1)
        else:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm
                     - 2 * self.ngroups * self.d_state - 2 * self.nheads) // 2
            z0, x0, z, xBC, dt, beta = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state,
                 self.nheads, self.nheads], dim=-1,
            )

        if padding_mask is not None:
            xBC = xBC.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        xBC = causal_conv1d(xBC, self.conv1d, self.activation)
        if self.no_gen_bcdt:
            x = xBC
        else:
            x, B, C = torch.split(
                xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1
            )

        # ---- GPS-DeltaNet ---- #
        y = self._delta_scan(x, B, C, dt, beta, self.A_log, self.dt_bias, self.beta_bias,
                             self.D, padding_mask, mode)

        # ---- Road-DeltaNet: parameters derived from the GPS branch output ---- #
        R_x = self.act(z)
        R_bcdt = self.R_bcdt_proj(y)
        R_B, R_C, R_dt, R_beta = torch.split(
            R_bcdt,
            [self.ngroups * self.d_state, self.ngroups * self.d_state, self.nheads, self.nheads],
            dim=-1,
        )
        if self.add_bcdt_item and self.B is not None:
            R_B, R_C, R_dt, R_beta = R_B + self.B, R_C + self.C, R_dt + self.dt, R_beta + self.beta
        R_y = self._delta_scan(R_x, R_B, R_C, R_dt, R_beta, self.R_A_log, self.R_dt_bias,
                               self.R_beta_bias, self.R_D, padding_mask, mode)

        # ---- cross-perspective gating ---- #
        y = self.norm(y, z) if self.rmsnorm else y * F.silu(z)

        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            R_y = torch.cat([F.silu(z0) * x0, R_y], dim=-1)
        if seqlen_og is not None:
            y = rearrange(y, "b l d -> (b l) d")
            R_y = rearrange(R_y, "b l d -> (b l) d")

        return torch.cat([self.R_out_proj(R_y), self.out_proj(y)], dim=-1)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        dtype = dtype or self.out_proj.weight.dtype
        conv_dim = self.conv1d.in_channels
        conv_state = torch.zeros(batch_size, conv_dim, self.d_conv, device=device, dtype=dtype)
        gps_state = torch.zeros(batch_size, self.nheads, self.d_state, self.headdim,
                                device=device, dtype=dtype)
        road_state = torch.zeros_like(gps_state)
        return conv_state, gps_state, road_state
