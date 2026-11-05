
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from data import COL_I
from utils import cal_tensor_geo_distance, get_batch_mask, tokenize_timestamp

from .encode import FourierEncode
from .gated_deltanet2 import DeltaPathMixerModel
from .deltapath import build_seq_encoder

__all__ = ["DeltaClip", "Attention"]


class Attention(nn.Module):

    def __init__(self, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.attn = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.v = nn.Linear(self.hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, neighbors_mask, batch_mask):
        N = encoder_outputs.shape[2]
        hidden = hidden.unsqueeze(2).repeat(1, 1, N, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=-1)))
        attention = self.v(energy).squeeze(-1)                         # (B, L, N)
        attention = attention.masked_fill(batch_mask.unsqueeze(-1) | neighbors_mask, -1e10)
        return F.softmax(attention, dim=-1)


class DeltaClip(nn.Module):

    def __init__(self, embed_size, d_model,
                 road_embed, road_neighbors, poi_embed, poi_coors,
                 spatial_border, high_order_feature_border, device,
                 use_higher_features=True, n_layer=4, d_state=128, headdim=64, d_inner=0,
                 use_S4_form=False, add_bcdt_item=False,
                 road_weight=1, poi_weight=1, pois_dist_thres=300, poi_max_neighbors=10,
                 road_transW_lambda=1.0, poi_distW_lambda=0.5,
                 view_n_layer=2, encoder_type="DeltaPath", chunk_size=64, use_qk_l2norm=True,
                 pretrain_task="purpose", **kwargs):
        super().__init__()
        assert d_model % 2 == 0
        self.device = device
        self.output_size = d_model
        self.pretrain_task = pretrain_task

        self.road_neighbors = nn.Parameter(
            torch.from_numpy(road_neighbors["road_neighbors_sets"]), requires_grad=False)
        self.road_neighbors_num = nn.Parameter(
            torch.from_numpy(road_neighbors["road_neighbors_num"]), requires_grad=False)
        # [length weight, transition weight]
        self.road_neighbors_weight = nn.Parameter(
            torch.from_numpy(road_neighbors["road_neighbors_weight"]).float(), requires_grad=False)

        self.poi_coors = nn.Parameter(torch.from_numpy(poi_coors).float(), requires_grad=False)
        self.pois_dist_thres = pois_dist_thres
        self.poi_max_neighbors = poi_max_neighbors
        self.construct_poi_neighbors()

        self.spatial_border = nn.Parameter(torch.tensor(spatial_border), requires_grad=False)
        self.high_order_feature_border = nn.Parameter(
            torch.tensor(high_order_feature_border), requires_grad=False)

        self.road_weight = road_weight
        self.poi_weight = poi_weight
        self.use_higher_features = use_higher_features

        ST_embed_size = d_model // 2
        self.traj_view = nn.ModuleDict({
            "spatial_embed_layer": nn.Sequential(
                nn.Linear(len(COL_I["spatial"]), embed_size), nn.LeakyReLU(),
                nn.Linear(embed_size, ST_embed_size)),
            "temporal_embed_modules": nn.ModuleList([FourierEncode(embed_size) for _ in range(5)]),
            "fine_temporal_embed_layer": nn.Sequential(
                nn.LeakyReLU(), nn.Linear(embed_size * 2, ST_embed_size)),
            "coarse_temporal_embed_layer": nn.Sequential(
                nn.LeakyReLU(), nn.Linear(embed_size * 3, ST_embed_size)),
            "road_index_embed_layer": nn.Sequential(
                nn.Embedding(road_embed.shape[0], embed_size), nn.LayerNorm(embed_size),
                nn.Linear(embed_size, ST_embed_size)),
            "seq_encoder": build_seq_encoder(
                encoder_type, d_model=d_model, n_layer=n_layer, d_intermediate=0,
                aux_feature_size=len(COL_I["high_order_features"]) if use_higher_features else 0,
                d_state=d_state, headdim=headdim, d_inner=d_inner, use_S4_form=use_S4_form,
                add_bcdt_item=add_bcdt_item, chunk_size=chunk_size, use_qk_l2norm=use_qk_l2norm,
                device=device, dtype=torch.float32),
        })

        view_d_inner = d_inner if d_model <= d_inner else d_model
        self.road_transW_lambda = road_transW_lambda
        road_embed_mat = nn.Embedding(*road_embed.shape)
        road_embed_mat.weight = nn.Parameter(torch.from_numpy(road_embed).float(), requires_grad=False)
        self.road_view = nn.ModuleDict({
            "text_embed_mat": road_embed_mat,
            "text_embed_layer": nn.Sequential(nn.LayerNorm(road_embed.shape[1]),
                                              nn.Linear(road_embed.shape[1], d_model)),
            "road_neighbors_att": Attention(d_model),
            "neighbors_weight_layer": nn.Linear(d_model, d_model, bias=False),
            "OD_weight_layer": nn.Linear(d_model, d_model, bias=False),
            "merge_norm": nn.Sequential(nn.BatchNorm1d(d_model), nn.ReLU(inplace=True)),
            "seq_encoder": DeltaPathMixerModel(
                d_model=d_model, n_layer=view_n_layer, d_intermediate=0, aux_feature_size=0,
                d_state=d_state, headdim=headdim, d_inner=view_d_inner,
                block_type="GatedDeltaNet2", chunk_size=chunk_size, use_qk_l2norm=use_qk_l2norm,
                device=device, dtype=torch.float32, verbose=False),
        })

        self.poi_distW_lambda = poi_distW_lambda
        poi_embed_mat = nn.Embedding(*poi_embed.shape)
        poi_embed_mat.weight = nn.Parameter(torch.from_numpy(poi_embed).float(), requires_grad=False)
        self.poi_view = nn.ModuleDict({
            "text_embed_mat": poi_embed_mat,
            "text_embed_layer": nn.Sequential(nn.LayerNorm(poi_embed.shape[1]),
                                              nn.Linear(poi_embed.shape[1], d_model)),
            "poi_neighbors_att": Attention(d_model),
            "nearpoint_weight_layer": nn.Linear(d_model, d_model, bias=False),
            "OD_weight_layer": nn.Linear(d_model, d_model, bias=False),
            "merge_norm": nn.Sequential(nn.BatchNorm1d(d_model), nn.ReLU(inplace=True)),
            "index_embed_layer": nn.Sequential(
                nn.Embedding(poi_embed.shape[0], embed_size), nn.LayerNorm(embed_size),
                nn.Linear(embed_size, d_model)),
            "seq_encoder": DeltaPathMixerModel(
                d_model=d_model, n_layer=view_n_layer, d_intermediate=0, aux_feature_size=0,
                d_state=d_state, headdim=headdim, d_inner=view_d_inner,
                block_type="GatedDeltaNet2", chunk_size=chunk_size, use_qk_l2norm=use_qk_l2norm,
                device=device, dtype=torch.float32, verbose=False),
        })

        if self.pretrain_task == "reconstruct":  # w/o Purpose ablation
            self.recon_head = nn.ModuleDict({
                "gps": nn.Sequential(nn.Linear(d_model, d_model), nn.LeakyReLU(),
                                     nn.Linear(d_model, len(COL_I["spatial"]))),
                "road": nn.Sequential(nn.Linear(d_model, d_model), nn.LeakyReLU(),
                                      nn.Linear(d_model, road_embed.shape[0])),
            })

        # Learned log-parameterised temperature, as in CLIP.
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cross_entropy = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------ #
    def _normalise(self, input_seq):
        spatial = input_seq[:, :, COL_I["spatial"]]
        temporal = input_seq[:, :, COL_I["temporal"]]
        road = input_seq[:, :, COL_I["road"]].long()
        norm_spatial = (spatial - self.spatial_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.spatial_border[1] - self.spatial_border[0]).unsqueeze(0).unsqueeze(0)
        high_order_features = input_seq[:, :, COL_I["high_order_features"]]
        norm_high = (high_order_features - self.high_order_feature_border[0].unsqueeze(0).unsqueeze(0)) / \
            (self.high_order_feature_border[1] - self.high_order_feature_border[0]).unsqueeze(0).unsqueeze(0)
        return spatial, norm_spatial, temporal, road, norm_high

    def cal_traj_h(self, norm_spatial, temporal, road, aux_features, valid_lens,
                   batch_mask, return_sequence=False):
        spatial_e = self.traj_view["spatial_embed_layer"](norm_spatial)
        temporal_tokens = [self.traj_view["temporal_embed_modules"][i](tok)
                           for i, tok in enumerate(tokenize_timestamp(temporal))]
        fine_temporal_e = self.traj_view["fine_temporal_embed_layer"](torch.cat(temporal_tokens[-2:], -1))
        coarse_temporal_e = self.traj_view["coarse_temporal_embed_layer"](torch.cat(temporal_tokens[:-2], -1))
        fine_temporal_e = fine_temporal_e.masked_fill(temporal[..., :1] < 0, 0)
        coarse_temporal_e = coarse_temporal_e.masked_fill(temporal[..., :1] < 0, 0)
        road_e = self.traj_view["road_index_embed_layer"](road)

        traj_h = torch.cat([road_e + coarse_temporal_e, spatial_e + fine_temporal_e], dim=-1)
        traj_h = self.traj_view["seq_encoder"](
            traj_h, aux_features if self.use_higher_features else None, padding_mask=batch_mask)
        traj_h = traj_h.masked_fill(batch_mask.unsqueeze(-1), 0.0)
        pooled = traj_h.sum(1) / valid_lens.unsqueeze(-1)
        return (pooled, traj_h) if return_sequence else pooled

    def forward(self, input_seq, valid_lens):
        B, L, _ = input_seq.shape
        batch_mask = get_batch_mask(B, L, valid_lens)
        _, norm_spatial, temporal, road, norm_high = self._normalise(input_seq)
        return self.cal_traj_h(norm_spatial, temporal, road, norm_high, valid_lens, batch_mask)

    # ------------------------------------------------------------------ #
    def loss(self, input_seq, valid_lens):
        B, L, _ = input_seq.shape
        batch_mask = get_batch_mask(B, L, valid_lens)
        batch_arr = torch.arange(end=B, dtype=torch.long, device=valid_lens.device)

        s_time = time.time()
        spatial, norm_spatial, temporal, road, norm_high = self._normalise(input_seq)
        e_time = time.time()

        if self.pretrain_task == "reconstruct":  # w/o Purpose
            _, seq_h = self.cal_traj_h(norm_spatial, temporal, road, norm_high, valid_lens,
                                       batch_mask, return_sequence=True)
            keep = ~batch_mask
            gps_loss = F.mse_loss(self.recon_head["gps"](seq_h)[keep], norm_spatial[keep])
            road_loss = F.cross_entropy(self.recon_head["road"](seq_h)[keep], road[keep])
            return gps_loss + road_loss, (e_time - s_time)

        traj_h = self.cal_traj_h(norm_spatial, temporal, road, norm_high, valid_lens, batch_mask)

        # destination weight w^d = delta_t_i / delta_t_n, origin weight w^o = 1 - w^d
        denom = temporal[..., 1][batch_arr, valid_lens - 1].unsqueeze(-1).clamp_min(1e-6)
        time_ratio = temporal[..., 1] / denom                                     # (B, L)

        road_h = self.cal_road_h(road, valid_lens, time_ratio, batch_arr, batch_mask)
        poi_h = self.cal_poi_h(spatial, valid_lens, time_ratio, batch_arr, batch_mask)

        traj_h = traj_h / traj_h.norm(dim=1, keepdim=True)
        road_h = road_h / road_h.norm(dim=1, keepdim=True)
        poi_h = poi_h / poi_h.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logit_road = logit_scale * traj_h @ road_h.t()
        logit_poi = logit_scale * traj_h @ poi_h.t()

        label = torch.arange(B).long().to(input_seq.device)
        loss_road = (self.cross_entropy(logit_road, label) + self.cross_entropy(logit_road.t(), label)) / 2
        loss_poi = (self.cross_entropy(logit_poi, label) + self.cross_entropy(logit_poi.t(), label)) / 2
        loss = self.road_weight * loss_road + self.poi_weight * loss_poi
        return loss, (e_time - s_time)

    # ------------------------------------------------------------------ #
    def cal_road_h(self, road, valid_lens, time_ratio, batch_arr, batch_mask):
        B, L = road.shape
        road_text_e = self.road_view["text_embed_layer"](self.road_view["text_embed_mat"](road))

        valid_neighbors_num = self.road_neighbors_num[road]                      # (B, L)
        N = int(valid_neighbors_num.max())
        neighbors_mask = torch.arange(end=N, device=valid_lens.device).unsqueeze(0).unsqueeze(0) \
            >= valid_neighbors_num.unsqueeze(-1)
        neighbors_text_e = self.road_view["text_embed_layer"](
            self.road_view["text_embed_mat"](self.road_neighbors[road][:, :, :N]))
        neighbors_weight = self.road_view["road_neighbors_att"](
            road_text_e, neighbors_text_e, neighbors_mask, batch_mask)
        static_neighbors_weight = self.road_neighbors_weight[road][:, :, :N]
        # attention relevance + alpha * transition probability
        neighbors_weight = neighbors_weight + self.road_transW_lambda * static_neighbors_weight[..., 1]
        neighbors_text_e = torch.matmul(neighbors_weight.unsqueeze(2), neighbors_text_e).squeeze(2)

        OD_text_e = (1 - time_ratio).unsqueeze(-1) * road_text_e[:, 0, :].unsqueeze(1) + \
            time_ratio.unsqueeze(-1) * road_text_e[batch_arr, valid_lens - 1].unsqueeze(1)

        merged = self.road_view["neighbors_weight_layer"](neighbors_text_e) + \
            self.road_view["OD_weight_layer"](OD_text_e)
        road_h = road_text_e + self.road_view["merge_norm"](merged.permute(0, 2, 1)).permute(0, 2, 1)

        road_h = self.road_view["seq_encoder"](road_h, None, padding_mask=batch_mask)
        road_h = road_h.masked_fill(batch_mask.unsqueeze(-1), 0.0)
        return road_h.sum(1) / valid_lens.unsqueeze(-1)

    def cal_poi_h(self, spatial, valid_lens, time_ratio, batch_arr, batch_mask):
        B, L, _ = spatial.shape
        nearest_poi = ((self.poi_coors.unsqueeze(0).unsqueeze(0) - spatial.unsqueeze(2)) ** 2) \
            .sum(-1).argmin(dim=-1)                                              # (B, L)

        poi_index_e = self.poi_view["index_embed_layer"](nearest_poi)
        poi_text_e = self.poi_view["text_embed_layer"](self.poi_view["text_embed_mat"](nearest_poi))

        valid_neighbors_num = self.poi_neighbors_num[nearest_poi]
        N = int(valid_neighbors_num.max())
        neighbors_mask = torch.arange(end=N, device=valid_lens.device).unsqueeze(0).unsqueeze(0) \
            >= valid_neighbors_num.unsqueeze(-1)
        neighbors_text_e = self.poi_view["text_embed_layer"](
            self.poi_view["text_embed_mat"](self.poi_neighbors[nearest_poi][:, :, :N]))
        neighbors_weight = self.poi_view["poi_neighbors_att"](
            poi_text_e, neighbors_text_e, neighbors_mask, batch_mask)
        neighbors_weight = neighbors_weight + \
            self.poi_distW_lambda * self.poi_neighbors_distW[nearest_poi][:, :, :N]
        nearpoint_merge = torch.matmul(neighbors_weight.unsqueeze(2), neighbors_text_e).squeeze(2)

        OD_merge = (1 - time_ratio).unsqueeze(-1) * poi_text_e[:, 0, :].unsqueeze(1) + \
            time_ratio.unsqueeze(-1) * poi_text_e[batch_arr, valid_lens - 1].unsqueeze(1)

        merged = self.poi_view["nearpoint_weight_layer"](nearpoint_merge) + \
            self.poi_view["OD_weight_layer"](OD_merge)
        poi_h = poi_index_e + poi_text_e + \
            self.poi_view["merge_norm"](merged.permute(0, 2, 1)).permute(0, 2, 1)

        poi_h = self.poi_view["seq_encoder"](poi_h, None, padding_mask=batch_mask)
        poi_h = poi_h.masked_fill(batch_mask.unsqueeze(-1), 0.0)
        return poi_h.sum(1) / valid_lens.unsqueeze(-1)

    # ------------------------------------------------------------------ #
    def construct_poi_neighbors(self):
        dist_m = cal_tensor_geo_distance(self.poi_coors.unsqueeze(0), self.poi_coors.unsqueeze(1))
        dist_m = dist_m.fill_diagonal_(np.inf)
        neighbors_id, neighbors_dist, neighbors_num = [], [], []
        for i in range(len(self.poi_coors)):
            neighbors_indices = torch.where(dist_m[i] <= self.pois_dist_thres)[0]
            if len(neighbors_indices) > self.poi_max_neighbors:
                value, indices = torch.topk(dist_m[i][neighbors_indices],
                                            self.poi_max_neighbors, largest=False)
                neighbors_id.append(neighbors_indices[indices])
                neighbors_dist.append(value)
                neighbors_num.append(self.poi_max_neighbors)
            else:
                neighbors_id.append(neighbors_indices)
                neighbors_dist.append(dist_m[i][neighbors_indices])
                neighbors_num.append(len(neighbors_indices))

        self.poi_neighbors = nn.Parameter(pad_sequence(neighbors_id, batch_first=True),
                                          requires_grad=False)
        self.poi_neighbors_num = nn.Parameter(torch.tensor(neighbors_num), requires_grad=False)
        assert self.poi_neighbors_num.min() > 0, "pois_dist_thres is set too small!"

        poi_neighbors_distW = pad_sequence(neighbors_dist, batch_first=True)
        # f(dist) = exp(-dist / max_dist), then L1-normalised over the neighbour set.
        poi_neighbors_distW = torch.exp(
            -poi_neighbors_distW / poi_neighbors_distW.max(dim=-1, keepdim=True).values
        ).masked_fill(
            get_batch_mask(self.poi_neighbors.shape[0], self.poi_neighbors.shape[1],
                           self.poi_neighbors_num), 0)
        self.poi_neighbors_distW = nn.Parameter(
            (poi_neighbors_distW / poi_neighbors_distW.sum(dim=-1, keepdim=True)).float(),
            requires_grad=False)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward_w_processtime_cal(self, input_seq, valid_lens):
        start_time = time.time()
        B, L, _ = input_seq.shape
        batch_mask = get_batch_mask(B, L, valid_lens)
        s_time = time.time()
        _, norm_spatial, temporal, road, norm_high = self._normalise(input_seq)
        e_time = time.time()
        traj_h = self.cal_traj_h(norm_spatial, temporal, road, norm_high, valid_lens, batch_mask)
        end_time = time.time()
        return traj_h, (end_time - start_time - (e_time - s_time)), (e_time - s_time)

    def forward_on_search_mode(self, input_seq, valid_lens):
        return self.forward_w_processtime_cal(input_seq, valid_lens)
