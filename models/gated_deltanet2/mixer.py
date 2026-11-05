
from __future__ import annotations

from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from .block import Block, GatedMLP, init_weights
from .deltapath_block import DeltaPathBlock
from .gated_deltanet2 import GatedDeltaNet2
from .norm import RMSNorm, layer_norm_fn

__all__ = ["DeltaPathMixerModel", "create_block"]

BLOCK_TYPES = ("DeltaPath", "GatedDeltaNet2")


def create_block(
    d_model,
    d_intermediate,
    aux_feature_size=0,
    d_state=128,
    headdim=64,
    d_inner=0,
    use_S4_form=False,
    add_bcdt_item=False,
    block_type="DeltaPath",
    chunk_size=64,
    use_qk_l2norm=True,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    assert block_type in BLOCK_TYPES, f"block_type must be one of {BLOCK_TYPES}"
    ssm_cfg = ssm_cfg or {}
    factory = {"device": device, "dtype": dtype}
    if block_type == "DeltaPath":
        mixer_cls = partial(
            DeltaPathBlock,
            d_inner=d_inner, d_state=d_state, headdim=headdim,
            aux_feature_size=aux_feature_size, use_S4_bcdt=use_S4_form,
            add_bcdt_item=add_bcdt_item, chunk_size=chunk_size,
            use_qk_l2norm=use_qk_l2norm, layer_idx=layer_idx, **ssm_cfg, **factory,
        )
    else:
        mixer_cls = partial(
            GatedDeltaNet2,
            d_inner=d_inner, d_state=d_state, headdim=headdim,
            chunk_size=chunk_size, use_qk_l2norm=use_qk_l2norm,
            layer_idx=layer_idx, **ssm_cfg, **factory,
        )
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory)
    mlp_cls = nn.Identity if d_intermediate == 0 else partial(
        GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory
    )
    block = Block(d_model, mixer_cls, mlp_cls, norm_cls=norm_cls,
                  fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block


class DeltaPathMixerModel(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        aux_feature_size: int,
        d_state: int = 128,
        headdim: int = 64,
        d_inner: int = 0,
        use_S4_form: bool = False,
        add_bcdt_item: bool = False,
        block_type: str = "DeltaPath",
        chunk_size: int = 64,
        use_qk_l2norm: bool = True,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm: bool = True,
        residual_in_fp32: bool = True,
        device=None,
        dtype=None,
        bias: bool = False,
        verbose: bool = True,
    ) -> None:
        factory = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        assert block_type in BLOCK_TYPES
        self.block_type = block_type

        self.layers = nn.ModuleList([
            create_block(
                d_model, d_intermediate=d_intermediate, aux_feature_size=aux_feature_size,
                d_state=d_state, headdim=headdim, d_inner=d_inner, use_S4_form=use_S4_form,
                add_bcdt_item=add_bcdt_item, block_type=block_type, chunk_size=chunk_size,
                use_qk_l2norm=use_qk_l2norm, ssm_cfg=ssm_cfg, norm_epsilon=norm_epsilon,
                rms_norm=rms_norm, residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm, layer_idx=i, **factory,
            )
            for i in range(n_layer)
        ])

        # Shared projection producing the input-dependent GDN-2 parameters.
        if aux_feature_size and block_type == "DeltaPath":
            if verbose:
                print("Using the trajectory's high-order movement features to construct GDN-2 params B, C, Delta, beta.")
            self.bcdt_proj_outdim = 0
            self.bcdt_dim_list = []
            for layer in self.layers:
                mixer = layer.mixer
                b_dim = c_dim = mixer.ngroups * mixer.d_state
                dt_dim = beta_dim = mixer.nheads
                self.bcdt_proj_outdim += b_dim + c_dim + dt_dim + beta_dim
                self.bcdt_dim_list.extend([b_dim, c_dim, dt_dim, beta_dim])
            self.bcdt_proj = nn.Linear(aux_feature_size, self.bcdt_proj_outdim, bias=bias, **factory)
            if verbose and self.layers[0].mixer.add_bcdt_item:
                print("Additionally re-introducing the input-independent parameterisation.")
        else:
            self.bcdt_proj = None
            if verbose and block_type == "DeltaPath":
                print(f"Constructing GDN-2 params with {'S4' if use_S4_form else 'S6'} form.")

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(d_model, eps=norm_epsilon, **factory)
        self.apply(partial(init_weights, n_layer=n_layer,
                           **(initializer_cfg or {}),
                           n_residuals_per_layer=1 if d_intermediate == 0 else 2))

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
                for i, layer in enumerate(self.layers)}

    def forward(self, hidden_states: torch.Tensor, aux_features: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None, inference_params=None,
                mode: str = "chunk", **mixer_kwargs):
        residual = None
        if self.bcdt_proj is not None:
            assert aux_features is not None, "this encoder expects movement features"
            all_bcdt = torch.split(self.bcdt_proj(aux_features), self.bcdt_dim_list, dim=-1)
            for i, layer in enumerate(self.layers):
                B, C, dt, beta = all_bcdt[4 * i: 4 * i + 4]
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params,
                    B=B, C=C, dt=dt, beta=beta, padding_mask=padding_mask, mode=mode,
                    **mixer_kwargs,
                )
        else:
            for layer in self.layers:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params,
                    padding_mask=padding_mask, mode=mode, **mixer_kwargs,
                )

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            hidden_states = layer_norm_fn(
                hidden_states, self.norm_f.weight, getattr(self.norm_f, "bias", None),
                eps=self.norm_f.eps, residual=residual, prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )
        return hidden_states
