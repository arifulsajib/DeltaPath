
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils  # noqa: E402
from data import (  # noqa: E402
    DataPreprocessor, DpPadder, FEATURE_COLS, MIN_TRIP_LEN, PRED_LEN, TrajectorySearchTestdata,
)


def _trip(n, trip_id=0):
    rng = np.random.default_rng(trip_id)
    df = pd.DataFrame({
        "trip": trip_id,
        "lng": 104.0 + np.cumsum(rng.random(n)) * 1e-3,
        "lat": 30.6 + np.cumsum(rng.random(n)) * 1e-3,
        "timestamp": 1.5e9 + np.cumsum(rng.integers(3, 15, n)).astype(float),
        "road": np.repeat(np.arange(n), 1)[:n] // 2,
    })
    return DataPreprocessor.cal_high_order_features(df)


# --------------------------------------------------------------------------- #
def test_min_trip_len_leaves_an_input_point():
    assert MIN_TRIP_LEN > PRED_LEN


@pytest.mark.parametrize("n", [PRED_LEN, PRED_LEN + 1, PRED_LEN + 2, 30])
@pytest.mark.parametrize("pred_len", [0, PRED_LEN])
def test_compression_never_crashes_and_keeps_targets(n, pred_len):
    group = _trip(n)
    out = DataPreprocessor.__new__(DataPreprocessor).gen_compressed_traj(group.copy(), pred_len)
    assert 1 <= len(out) <= n
    if pred_len and n > pred_len:
        np.testing.assert_allclose(out[["lng", "lat"]].to_numpy()[-pred_len:],
                                   group[["lng", "lat"]].to_numpy()[-pred_len:])


def test_dp_padder_rejects_too_short_trips():
    padder = DpPadder(device="cpu", pred_len=PRED_LEN, pred_cols=["road"])
    with pytest.raises(ValueError):
        padder([_trip(PRED_LEN)])
    traj, valid_lens, label = padder([_trip(PRED_LEN + 3)])
    assert int(valid_lens[0]) == 3 and traj.shape[-1] == len(FEATURE_COLS)


# --------------------------------------------------------------------------- #
def _old_cal_pres_and_labels(query, target, negs):
    """The original dense implementation, kept here as a reference."""
    nq = query.shape[0]
    dist_qt = np.linalg.norm(query[:, None, :] - target[None, :, :], ord=2, axis=2)
    dist_qn = np.linalg.norm(query[:, None, :] - negs, ord=2, axis=2)
    return -np.concatenate([dist_qt[np.eye(nq).astype(bool)][:, None], dist_qn], axis=1)


def test_chunked_search_scores_match_dense_reference():
    rng = np.random.default_rng(0)
    q, t = rng.normal(size=(37, 16)).astype(np.float32), rng.normal(size=(37, 16)).astype(np.float32)
    db = rng.normal(size=(200, 16)).astype(np.float32)
    neg = np.stack([rng.choice(200, 50, replace=False) for _ in range(37)])
    scores, labels = TrajectorySearchTestdata.cal_pres_and_labels(q, t, db, neg, chunk_size=8)
    np.testing.assert_allclose(scores, _old_cal_pres_and_labels(q, t, db[neg]), rtol=1e-6)
    assert (labels == 0).all()


def test_search_metric_matches_argsort_rank_without_ties():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(100, 51))
    old_rank = scores.argsort(axis=1)[:, ::-1].argsort(axis=1)[:, 0] + 1
    m = utils.cal_search_metric(scores)
    assert m["mean_rank"] == pytest.approx(old_rank.mean())
    assert m["acc@1"] == pytest.approx(np.mean(old_rank <= 1))
    assert m["acc@5"] == pytest.approx(np.mean(old_rank <= 5))


def test_search_metric_does_not_reward_a_collapsed_encoder():
    scores = np.zeros((10, 21))          # every trajectory embedded to the same point
    m = utils.cal_search_metric(scores)
    assert m["acc@1"] == 0.0 and m["mean_rank"] == 21.0
