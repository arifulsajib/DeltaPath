
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .encode import PositionalEmbedding

__all__ = ["TransformerMixerModel"]


class TransformerMixerModel(nn.Module):
    def __init__(self, d_model: int, n_layer: int, d_intermediate: int, aux_feature_size: int,
                 nhead: int = 4, dropout: float = 0.1, device=None, dtype=None,
                 verbose: bool = True, **unused):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.aux_proj = (nn.Linear(aux_feature_size, d_model, **factory)
                         if aux_feature_size else None)
        self.pos = PositionalEmbedding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_intermediate if d_intermediate else 4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True, **factory,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer, enable_nested_tensor=False)
        self.norm_f = nn.LayerNorm(d_model, **factory)
        if verbose:
            print(f"[ablation] Using a {n_layer}-layer Transformer encoder instead of DeltaPath blocks.")

    def forward(self, hidden_states: torch.Tensor, aux_features: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None, **kwargs):
        B, L, _ = hidden_states.shape
        positions = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, L).float()
        h = hidden_states + self.pos(positions)
        if self.aux_proj is not None and aux_features is not None:
            h = h + self.aux_proj(aux_features)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        return self.norm_f(torch.nan_to_num(h, nan=0.0))
