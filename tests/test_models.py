
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.delta_clip import DeltaClip  # noqa: E402
from models.deltapath import DeltaPath  # noqa: E402
from models.predictor import MlpPredictor  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

B, L, NUM_ROAD, NUM_POI, D_TEXT, E = 4, 30, 40, 25, 16, 64
SPATIAL_BORDER = [[104.0, 30.6], [104.2, 30.8]]
TEMPORAL_BORDER = [[0.0, 0.0], [600000.0, 3600.0]]
HIGH_BORDER = [[0.0, -5.0, 0.0], [30.0, 5.0, 360.0]]

COMMON = dict(embed_size=16, d_model=E, n_layer=2, d_state=16, headdim=32, d_inner=E,
              add_bcdt_item=True, chunk_size=8)


def _batch():
    lng = torch.rand(B, L) * 0.2 + 104.0
    lat = torch.rand(B, L) * 0.2 + 30.6
    ts = torch.cumsum(torch.rand(B, L) * 10 + 1, dim=1) + 1000.0
    dt = ts - ts[:, :1]
    road = torch.randint(0, NUM_ROAD, (B, L)).float()
    speed = torch.rand(B, L) * 20
    acc = torch.randn(B, L)
    angle = torch.rand(B, L) * 360
    seq = torch.stack([lng, lat, ts, dt, road, speed, acc, angle], dim=-1)
    valid_lens = torch.randint(L // 2, L + 1, (B,))
    valid_lens[0] = L
    return seq, valid_lens


def _make_deltapath(**over):
    kwargs = dict(COMMON)
    kwargs.update(over)
    return DeltaPath(num_road=NUM_ROAD, spatial_border=SPATIAL_BORDER,
                     high_order_feature_border=HIGH_BORDER, temporal_border=TEMPORAL_BORDER,
                     device="cpu", **kwargs)


def _make_deltaclip(**over):
    kwargs = dict(COMMON)
    kwargs.update(over)
    road_neighbors = {
        "road_neighbors_sets": np.random.randint(0, NUM_ROAD, (NUM_ROAD, 4)).astype(np.int64),
        "road_neighbors_num": np.full(NUM_ROAD, 4, dtype=np.int64),
        "road_neighbors_weight": np.abs(np.random.rand(NUM_ROAD, 4, 2)),
    }
    poi_coors = np.stack([np.random.rand(NUM_POI) * 0.2 + 104.0,
                          np.random.rand(NUM_POI) * 0.2 + 30.6], axis=-1)
    return DeltaClip(road_embed=np.random.randn(NUM_ROAD, D_TEXT).astype(np.float32),
                     road_neighbors=road_neighbors,
                     poi_embed=np.random.randn(NUM_POI, D_TEXT).astype(np.float32),
                     poi_coors=poi_coors, spatial_border=SPATIAL_BORDER,
                     high_order_feature_border=HIGH_BORDER, device="cpu",
                     pois_dist_thres=1e9, poi_max_neighbors=5, view_n_layer=2, **kwargs)


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("encoder_type", ["DeltaPath", "GatedDeltaNet2", "Transformer"])
def test_deltapath_forward_backward(encoder_type):
    model = _make_deltapath(encoder_type=encoder_type)
    seq, valid_lens = _batch()
    out = model(seq, valid_lens)
    assert out.shape == (B, E)
    assert torch.isfinite(out).all()
    out.sum().backward()


def test_deltapath_without_mask_generator():
    model = _make_deltapath(use_mask_generator=False)
    seq, valid_lens = _batch()
    out = model(seq, valid_lens)
    assert out.shape == (B, E)
    assert model.mask_generator is None


def test_deltapath_distillation_loss():
    model = _make_deltapath()
    seq, valid_lens = _batch()
    teacher = torch.randn(B, E)

    class _FakeMEC:
        def __call__(self, z, p, bs, it):
            return ((z - p) ** 2).mean()

    loss, new_valid_lens, _ = model.loss(seq, valid_lens, teacher, MEC_weight=0.5,
                                         MEC_loss=_FakeMEC(), it=0)
    assert torch.isfinite(loss)
    loss.backward()
    # Compression can only shorten trajectories, and never below 1 point.
    assert (new_valid_lens <= valid_lens).all()
    assert (new_valid_lens >= 1).all()


def test_mask_generator_keeps_origin_and_destination():
    model = _make_deltapath().eval()
    seq, valid_lens = _batch()
    compressed, new_valid_lens, _ = model.cal_compressed_traj(seq, valid_lens)
    batch_arr = torch.arange(B)
    assert torch.allclose(compressed[:, 0], seq[:, 0])
    assert torch.allclose(compressed[batch_arr, new_valid_lens - 1],
                          seq[batch_arr, valid_lens - 1])


@pytest.mark.parametrize("pretrain_task", ["purpose", "reconstruct"])
def test_deltaclip_loss(pretrain_task):
    model = _make_deltaclip(pretrain_task=pretrain_task)
    seq, valid_lens = _batch()
    loss, _ = model.loss(seq, valid_lens)
    assert torch.isfinite(loss)
    loss.backward()
    assert model(seq, valid_lens).shape == (B, E)


def test_teacher_weights_transfer_to_student():
    teacher = _make_deltaclip()
    student = _make_deltapath()
    student.traj_view.load_state_dict(teacher.traj_view.state_dict())


@pytest.mark.parametrize("pred_type,out_size,cols", [("classification", NUM_ROAD, 1),
                                                     ("regression", 2, 2),
                                                     ("regression", 1, 1)])
def test_prediction_heads(pred_type, out_size, cols):
    head = MlpPredictor(input_size=E, hidden_size=E, output_size=out_size,
                        pred_type=pred_type, spatial_border=SPATIAL_BORDER)
    traj_h = torch.randn(B, E)
    label = (torch.randint(0, NUM_ROAD, (B, 1)).float() if pred_type == "classification"
             else torch.rand(B, cols))
    assert torch.isfinite(head.loss(traj_h, label))
