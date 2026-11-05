
import math
import os
import random
import string
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
)


# --------------------------------------------------------------------------- #
# housekeeping
# --------------------------------------------------------------------------- #
def get_datetime_key():
    return "D" + datetime.now().strftime("%Y_%m_%dT%H_%M_%S_") + get_random_string(4)


def get_random_string(length):
    return "".join(random.choice(string.ascii_uppercase) for _ in range(length))


def create_if_noexists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def cal_courseAngle(a_coor, b_coor):
    a_coor, b_coor = np.radians(a_coor), np.radians(b_coor)
    a_x, a_y = a_coor[..., 0], a_coor[..., 1]
    b_x, b_y = b_coor[..., 0], b_coor[..., 1]
    d_x = b_x - a_x
    y = np.sin(d_x) * np.cos(b_y)
    x = np.cos(a_y) * np.sin(b_y) - np.sin(a_y) * np.cos(b_y) * np.cos(d_x)
    bearing = 180 * np.arctan2(y, x) / np.pi
    return np.where(bearing < 0, bearing + 360, bearing)


def cal_geo_distance(a_coor, b_coor):
    a_coor, b_coor = np.radians(a_coor), np.radians(b_coor)
    a_x, a_y = a_coor[..., 0], a_coor[..., 1]
    b_x, b_y = b_coor[..., 0], b_coor[..., 1]
    d_x, d_y = a_x - b_x, a_y - b_y
    a = np.sin(d_y / 2) ** 2 + np.cos(a_y) * np.cos(b_y) * np.sin(d_x / 2) ** 2
    return 2 * np.arcsin(np.sqrt(a)) * 6371 * 1000


def cal_tensor_geo_distance(a_coor: torch.Tensor, b_coor: torch.Tensor):
    a_coor, b_coor = torch.deg2rad(a_coor), torch.deg2rad(b_coor)
    a_x, a_y = a_coor[..., 0], a_coor[..., 1]
    b_x, b_y = b_coor[..., 0], b_coor[..., 1]
    d_x, d_y = a_x - b_x, a_y - b_y
    a = torch.sin(d_y / 2) ** 2 + torch.cos(a_y) * torch.cos(b_y) * torch.sin(d_x / 2) ** 2
    return 2 * torch.arcsin(torch.sqrt(a)) * 6371 * 1000


# --------------------------------------------------------------------------- #
# masking / packing
# --------------------------------------------------------------------------- #
def get_batch_mask(B, L, valid_len):
    return torch.arange(end=L, device=valid_len.device).unsqueeze(0) >= valid_len.unsqueeze(-1)


def tokenize_timestamp(t):
    week = t[..., 0] % (7 * 24 * 60 * 60) / (24 * 60 * 60)
    hour = t[..., 0] % (24 * 60 * 60) / (60 * 60)
    minute = t[..., 0] % (60 * 60) / 60
    d_minute = t[..., 1] / 60
    timestamp = t[..., 0] / 60
    return week, hour, minute, d_minute, timestamp


def pack_input(hidden_states, mask):
    hidden_states = rearrange(hidden_states, "b s ... -> (b s) ...")
    indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
    return hidden_states[indices], indices


def pad_input(hidden_states, indices, batch, seqlen):
    output = torch.zeros(batch * seqlen, *hidden_states.shape[1:],
                         device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def cal_regression_metric(label, pres):
    rmse = math.sqrt(mean_squared_error(label, pres))
    mae = mean_absolute_error(label, pres)
    mape = mean_absolute_percentage_error(label, pres)
    return pd.Series([rmse, mae, mape], index=["rmse", "mae", "mape"])


def distance_mae(distance, null_val=np.nan):
    return np.mean(np.abs(distance))


def distance_mse(distance, null_val=np.nan):
    return np.mean(distance ** 2)


def distance_rmse(distance, null_val=np.nan):
    return np.sqrt(distance_mse(distance, null_val))


def cal_distance_metric(label, pres, lng_col, lat_col):
    distance = cal_geo_distance(label[..., [lng_col, lat_col]], pres[..., [lng_col, lat_col]])
    return pd.Series([distance_rmse(distance, 0.0), distance_mae(distance, 0.0)],
                     index=["distance_rmse", "distance_mae"])


def top_n_accuracy(truths, preds, n):
    best_n = np.argsort(-preds, axis=1)[:, :n]
    successes = sum(1 for i, truth in enumerate(truths) if truth in best_n[i, :])
    return float(successes) / truths.shape[0]


def cal_classification_metric(labels, pres):
    pres_index = pres.argmax(-1)
    macro_f1 = f1_score(labels, pres_index, average="macro", zero_division=0)
    macro_recall = recall_score(labels, pres_index, average="macro", zero_division=0)
    acc = accuracy_score(labels, pres_index)
    n_list = [5, 10]
    top_n_acc = [top_n_accuracy(labels, pres, n) for n in n_list]
    return pd.Series([macro_f1, macro_recall, acc] + top_n_acc,
                     index=["macro_f1", "macro_rec"] + [f"acc@{n}" for n in [1] + n_list])


def cal_search_metric(scores, target_col=0, n_list=(1, 5, 10)):
    """Similar-trajectory-search metrics from a (num_query, 1 + num_neg) score matrix.

    The rank of the true target is ``1 + #{negatives scoring >= target}``, so a
    tie with a negative counts *against* the model.  (An argsort-based rank
    breaks ties by column order, which puts the target -- column 0 -- first and
    lets a collapsed encoder that maps everything to one point score 100%.)
    Without ties this equals the usual rank.
    """
    target = scores[:, target_col][:, None]
    others = np.delete(scores, target_col, axis=1)
    ranks = 1 + (others >= target).sum(axis=1)
    values = [float(np.mean(ranks <= n)) for n in n_list] + [float(np.mean(ranks))]
    return pd.Series(values, index=[f"acc@{n}" for n in n_list] + ["mean_rank"])


# --------------------------------------------------------------------------- #
# model bookkeeping
# --------------------------------------------------------------------------- #
def cal_model_size(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters() if p.requires_grad)
    return param_size / 1024 ** 2


def lamda_scheduler(start_warmup_value, base_value, epochs, niter_per_ep, warmup_epochs=5):
    warmup_epochs = min(warmup_epochs, epochs)
    warmup_iters = warmup_epochs * niter_per_ep
    warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters) if warmup_epochs > 0 else np.array([])
    schedule = np.concatenate((warmup_schedule, np.ones(epochs * niter_per_ep - warmup_iters) * base_value))
    assert len(schedule) == epochs * niter_per_ep
    return schedule
