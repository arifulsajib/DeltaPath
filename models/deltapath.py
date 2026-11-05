
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from data import COL_I, MAX_TRIP_LEN
from utils import get_batch_mask, lamda_scheduler, pack_input, pad_input, tokenize_timestamp

from .encode import FourierEncode
from .gated_deltanet2 import DeltaPathMixerModel, LightGatedDeltaNet2

__all__ = ["DeltaPath", "GateMask", "MEC", "build_seq_encoder"]


def build_seq_encoder(encoder_type: str, **kwargs):
    if encoder_type == "Transformer":
        from .transformer import TransformerMixerModel

        return TransformerMixerModel(**kwargs)
    kwargs = dict(kwargs)
    kwargs["block_type"] = encoder_type
    if encoder_type == "GatedDeltaNet2":
        # The vanilla block derives B/C/Delta from its own token stream.
        kwargs["aux_feature_size"] = 0
    return DeltaPathMixerModel(**kwargs)


# --------------------------------------------------------------------------- #
class GateMask(nn.Module):

    def __init__(self, d_model, expand: int = 2, sigma: float = 0.5,
                 pooling_method: str = "sigmoid", topk: int = 3, device=None):
        super().__init__()
        self.mu = nn.Parameter(torch.randn(MAX_TRIP_LEN, d_model))
        self.sigma = sigma
        self.based = 0.5
        self.topk = topk
        self.pooling_method = pooling_method
        self.device = device
        self.trend_func = LightGatedDeltaNet2(d_model=d_model, expand=expand,
                                              device=device, dtype=torch.float32)
        # A standalone mixer has no block-level normalisation, so we add one to
        # keep the intermediate activations from drifting.
        self.pre_norm = nn.LayerNorm(d_model, eps=1e-5, device=device)
        self.final_norm = nn.LayerNorm(d_model, eps=1e-5, device=device)

    def forward(self, hidden_state, valid_lens, batch_arr, batch_mask):
        mu = self.mu[: hidden_state.size(1)].unsqueeze(0)
        mu_ = self.refactor_mask(mu, hidden_state, batch_mask)            # (B, L, H)
        mu_ = mu_.topk(k=min(self.topk, mu_.size(-1)), dim=-1)[0].mean(dim=-1)  # (B, L)

        mask = mu_ if not self.training else \
            mu_ + torch.normal(0, self.sigma, mu_.shape, device=mu_.device)
        input_mask = self.hard_sigmoid(mask.masked_fill(batch_mask, 0))
        # The origin and the destination always carry travel-purpose semantics.
        input_mask[:, 0] = 1.0
        input_mask[batch_arr, valid_lens - 1] = 1.0

        new_valid_lens = (input_mask != 0).sum(dim=-1)
        new_L = int(new_valid_lens.max().item())
        return input_mask, new_valid_lens, new_L, mu_

    def refactor_mask(self, mu, x, batch_mask=None):
        residual = x
        output = self.pre_norm(residual)
        output = self.trend_func(output, padding_mask=batch_mask)
        trend = self.final_norm(output + residual)
        mu = mu + self.based
        return mu * torch.sigmoid(mu * trend)

    @staticmethod
    def hard_sigmoid(x):
        return torch.clamp(x, 0.0, 1.0)


# --------------------------------------------------------------------------- #
class MEC(nn.Module):

    def __init__(self, embed_dim, num_epoch, num_iter, batch_size,
                 hidden_size=256, n=5, eps=512, warmup_epoch=10):
        super().__init__()
        # A non-linear predictor on top of the student encoder.
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, embed_dim),
        )
        self.n = n
        lamda = 1 / (batch_size * eps / embed_dim)  # d / (N * eps^2)
        self.lamda_schedule = lamda_scheduler(8 / lamda, 1 / lamda, num_epoch, num_iter,
                                              warmup_epochs=warmup_epoch)

    def forward(self, z, p, batch_size, it):
        z = self.predictor(z)
        lamda_inv = self.lamda_schedule[min(it, len(self.lamda_schedule) - 1)]
        loss = self.mec(p, z, lamda_inv) / batch_size
        return -1 * loss * lamda_inv

    def mec(self, view1, view2, lamda_inv):
        view1, view2 = F.normalize(view1), F.normalize(view2)
        c = torch.mm(view1, view2.transpose(0, 1)) / lamda_inv  # (B, B)
        power = c
        sum_p = torch.zeros_like(power)
        for k in range(1, self.n + 1):
            if k > 1:
                power = torch.mm(power, c)
            if (k + 1) % 2 == 0:
                sum_p += power / k
            else:
                sum_p -= power / k
        return torch.trace(sum_p)


# --------------------------------------------------------------------------- #
class DeltaPath(nn.Module):

    def __init__(self, embed_size, d_model, num_road,
                 spatial_border, high_order_feature_border, temporal_border,
                 device, use_higher_features=True, n_layer=4, d_state=128, headdim=64,
                 d_inner=0, use_S4_form=False, add_bcdt_item=False, mask_weight=0.5,
                 encoder_type="DeltaPath", chunk_size=64, use_qk_l2norm=True,
                 use_mask_generator=True, soft_mask_scaling=False, mask_sigma=0.5, **kwargs):
        super().__init__()
        assert d_model % 2 == 0
        self.output_size = d_model
        self.num_road = num_road
        self.spatial_size = len(COL_I["spatial"])
        self.temporal_size = len(COL_I["temporal"])
        self.high_order_feature_size = len(COL_I["high_order_features"])
        self.device = device
        self.spatial_border = nn.Parameter(torch.tensor(spatial_border), requires_grad=False)
        self.temporal_border = nn.Parameter(torch.tensor(temporal_border), requires_grad=False)
        self.high_order_feature_border = nn.Parameter(torch.tensor(high_order_feature_border),
                                                      requires_grad=False)
        self.use_higher_features = use_higher_features
        self.use_mask_generator = use_mask_generator
        self.soft_mask_scaling = soft_mask_scaling

        self.mask_generator = GateMask(
            d_model=self.spatial_size + self.temporal_size + 1, sigma=mask_sigma, device=device
        ) if use_mask_generator else None

        ST_embed_size = d_model // 2
        self.traj_view = nn.ModuleDict({
            "spatial_embed_layer": nn.Sequential(
                nn.Linear(len(COL_I["spatial"]), embed_size), nn.LeakyReLU(),
                nn.Linear(embed_size, ST_embed_size)),
            "temporal_embed_modules": nn.ModuleList([FourierEncode(embed_size) for _ in range(5)]),
            # fine-grained temporal features: delta_t and the raw timestamp
            "fine_temporal_embed_layer": nn.Sequential(
                nn.LeakyReLU(), nn.Linear(embed_size * 2, ST_embed_size)),
            # coarse-grained cyclic temporal features: day-in-week / hour-in-day / minute-in-hour
            "coarse_temporal_embed_layer": nn.Sequential(
                nn.LeakyReLU(), nn.Linear(embed_size * 3, ST_embed_size)),
            "road_index_embed_layer": nn.Sequential(
                nn.Embedding(num_road, embed_size), nn.LayerNorm(embed_size),
                nn.Linear(embed_size, ST_embed_size)),
            "seq_encoder": build_seq_encoder(
                encoder_type, d_model=d_model, n_layer=n_layer, d_intermediate=0,
                aux_feature_size=self.high_order_feature_size if self.use_higher_features else 0,
                d_state=d_state, headdim=headdim, d_inner=d_inner, use_S4_form=use_S4_form,
                add_bcdt_item=add_bcdt_item, chunk_size=chunk_size, use_qk_l2norm=use_qk_l2norm,
                device=device, dtype=torch.float32),
        })
        self.mask_weight = mask_weight

    # ------------------------------------------------------------------ #
    def _normalise(self, input_seq):
        spatial = input_seq[:, :, COL_I["spatial"]]                      # (B, L, 2)
        temporal = input_seq[:, :, COL_I["temporal"]]                    # (B, L, 2)
        road = input_seq[:, :, COL_I["road"]].long()                     # (B, L)
        norm_spatial = (spatial - self.spatial_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.spatial_border[1] - self.spatial_border[0]).unsqueeze(0).unsqueeze(0)
        norm_temporal = (temporal - self.temporal_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.temporal_border[1] - self.temporal_border[0]).unsqueeze(0).unsqueeze(0)
        norm_road = road / (self.num_road - 1)
        mask_generator_input = torch.cat(
            [norm_spatial, norm_temporal, norm_road.unsqueeze(-1)], dim=-1)
        high_order_features = input_seq[:, :, COL_I["high_order_features"]]   # (B, L, 3)
        norm_high = (high_order_features - self.high_order_feature_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.high_order_feature_border[1] - self.high_order_feature_border[0]).unsqueeze(0).unsqueeze(0)
        return norm_spatial, temporal, road, norm_high, mask_generator_input

    def _embed_points(self, norm_spatial, temporal, road):
        spatial_e = self.traj_view["spatial_embed_layer"](norm_spatial)
        temporal_tokens = [self.traj_view["temporal_embed_modules"][i](tok)
                           for i, tok in enumerate(tokenize_timestamp(temporal))]
        fine_temporal_e = self.traj_view["fine_temporal_embed_layer"](
            torch.cat(temporal_tokens[-2:], -1))
        coarse_temporal_e = self.traj_view["coarse_temporal_embed_layer"](
            torch.cat(temporal_tokens[:-2], -1))
        # Negative temporal values mark a masked-out feature.
        temporal_mask = temporal[..., :1] < 0
        fine_temporal_e = fine_temporal_e.masked_fill(temporal_mask, 0)
        coarse_temporal_e = coarse_temporal_e.masked_fill(temporal_mask, 0)
        road_e = self.traj_view["road_index_embed_layer"](road)
        return torch.cat([road_e + coarse_temporal_e, spatial_e + fine_temporal_e], dim=-1)

    # ------------------------------------------------------------------ #
    def forward(self, input_seq, valid_lens):
        norm_spatial, temporal, road, norm_high, mg_input = self._normalise(input_seq)
        traj_h, _, _ = self.cal_traj_h(norm_spatial, temporal, road, norm_high, valid_lens, mg_input)
        return traj_h

    def cal_traj_h(self, norm_spatial, temporal, road, aux_features, valid_lens,
                   mask_generator_input=None):
        B, L = norm_spatial.size(0), norm_spatial.size(1)
        batch_mask = get_batch_mask(B, L, valid_lens)
        batch_arr = torch.arange(end=B, dtype=torch.long, device=valid_lens.device)

        traj_h = self._embed_points(norm_spatial, temporal, road)

        if self.mask_generator is not None:
            mask_generator_input = mask_generator_input.masked_fill(batch_mask.unsqueeze(-1), 0.0)
            input_mask, new_valid_lens, new_L, mu_ = self.mask_generator(
                mask_generator_input, valid_lens, batch_arr, batch_mask)
            if self.soft_mask_scaling:
                traj_h = traj_h * input_mask.unsqueeze(-1)
            # Gather the retained points and re-pad them into a dense batch.
            packed_h, indices = pack_input(traj_h, input_mask)
            packed_aux = rearrange(aux_features, "b s ... -> (b s) ...")[indices]
            new_batch_mask = get_batch_mask(B, new_L, new_valid_lens)
            _, new_indices = pack_input(
                torch.ones(B, new_L, 1, device=traj_h.device), ~new_batch_mask)
            traj_h = pad_input(packed_h, new_indices, B, new_L)
            aux_features = pad_input(packed_aux, new_indices, B, new_L)
        else:  # w/o Compress
            new_valid_lens, new_L, new_batch_mask = valid_lens, L, batch_mask
            mu_ = None

        encoded = self.traj_view["seq_encoder"](
            traj_h, aux_features if self.use_higher_features else None,
            padding_mask=new_batch_mask)
        encoded = encoded.masked_fill(new_batch_mask.unsqueeze(-1), 0.0)
        traj_h = encoded.sum(1) / new_valid_lens.unsqueeze(-1)
        return traj_h, new_valid_lens, mu_

    # ------------------------------------------------------------------ #
    def loss(self, input_seq, valid_lens, teacher_traj_h, MEC_weight=0, MEC_loss=None, it=None):
        s_time = time.time()
        norm_spatial, temporal, road, norm_high, mg_input = self._normalise(input_seq)
        e_time = time.time()

        traj_h, new_valid_lens, mu_ = self.cal_traj_h(
            norm_spatial, temporal, road, norm_high, valid_lens, mg_input)

        mask_loss = self.cal_mask_loss(mu_, valid_lens) if mu_ is not None else 0.0
        mec_loss = MEC_loss(traj_h, teacher_traj_h, traj_h.size(0), it) if MEC_weight else 0.0
        loss = self.mask_weight * mask_loss + MEC_weight * mec_loss
        return loss, new_valid_lens, (e_time - s_time)

    def cal_mask_loss(self, mu_, valid_lens):
        if len(mu_.shape) > 1:
            batch_mask = get_batch_mask(mu_.size(0), mu_.size(1), valid_lens)
            mu_ = mu_.masked_fill(batch_mask, 0)
        reg = 0.5 + 0.5 * torch.erf(mu_ / (self.mask_generator.sigma * np.sqrt(2)))
        return reg.mean()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward_on_search_mode(self, input_seq, valid_lens):
        s_time = time.time()
        norm_spatial, temporal, road, norm_high, mg_input = self._normalise(input_seq)
        e_time = time.time()

        start_time = time.time()
        traj_h, _, _ = self.cal_traj_h(norm_spatial, temporal, road, norm_high, valid_lens, mg_input)
        end_time = time.time()
        return traj_h, (end_time - start_time), (e_time - s_time)

    def forward_w_processtime_cal(self, input_seq, valid_lens):
        return self.forward_on_search_mode(input_seq, valid_lens)

    @torch.no_grad()
    def cal_compressed_traj(self, input_seq, valid_lens):
        _, _, _, _, mg_input = self._normalise(input_seq)
        B, L = input_seq.size(0), input_seq.size(1)
        batch_mask = get_batch_mask(B, L, valid_lens)
        batch_arr = torch.arange(end=B, dtype=torch.long, device=valid_lens.device)
        mg_input = mg_input.masked_fill(batch_mask.unsqueeze(-1), 0.0)
        input_mask, new_valid_lens, new_L, _ = self.mask_generator(
            mg_input, valid_lens, batch_arr, batch_mask)

        new_batch_mask = get_batch_mask(B, new_L, new_valid_lens)
        _, new_indices = pack_input(torch.ones(B, new_L, 1, device=input_seq.device), ~new_batch_mask)
        packed_seq, _ = pack_input(input_seq, input_mask)
        return pad_input(packed_seq, new_indices, B, new_L), new_valid_lens, new_L
