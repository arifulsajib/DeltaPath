
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from tqdm import tqdm, trange

from data import TrajectorySearchTestdata, X_COL, Y_COL
from models.deltapath import MEC
from utils import cal_classification_metric, cal_distance_metric, cal_regression_metric


# --------------------------------------------------------------------------- #
def pretrain_model(model, teacher_model, dataset, dataloader,
                   unshuffle_teacher_dataloader, teacher_dataloader,
                   num_epoch_1, num_epoch_2, lr_1, lr_2, MEC_weight=0, MEC_config=None,
                   teacher_prepared=False, teacher_savename=None):
    num_iter = len(unshuffle_teacher_dataloader)
    batch_size = unshuffle_teacher_dataloader.batch_size
    if MEC_weight:
        MEC_loss = MEC(model.output_size, num_epoch_2, num_iter, batch_size,
                       **(MEC_config or {})).to(model.device)
    else:
        MEC_loss = None
        print("Not using the MEC loss.")

    pretrain_time_1 = None
    if not teacher_prepared:
        optimizer_t = torch.optim.Adam(teacher_model.parameters(), lr=lr_1)
        teacher_model.train()

        bar_desc = "Pre-training stage 1 (travel purpose), avg loss: %.5f"
        whole_process_time = []
        start_time = time.time()
        with trange(num_epoch_1, desc=bar_desc % 0.0, position=0) as bar:
            for _ in bar:
                loss_values, traj_process_time = [], []
                for batch in tqdm(teacher_dataloader, desc="-->Traversing", leave=False, ncols=60):
                    optimizer_t.zero_grad()
                    loss, process_time = teacher_model.loss(*batch)
                    loss.backward()
                    optimizer_t.step()
                    loss_values.append(loss.item())
                    traj_process_time.append(process_time)
                bar.set_description(bar_desc % np.mean(loss_values))
                whole_process_time.append(np.sum(traj_process_time))
        end_time = time.time()
        pretrain_time_1 = ((end_time - start_time) - np.sum(whole_process_time)) / (num_epoch_1 * 60)
        print(f"Stage 1 mean pre-train time: {pretrain_time_1:.4f} min/epoch")
        if teacher_savename is not None:
            torch.save(teacher_model.state_dict(), teacher_savename)

    # The student inherits the teacher's trajectory encoder weights.
    model.traj_view.load_state_dict(copy.deepcopy(teacher_model.traj_view).state_dict())
    params = list(model.parameters()) + (list(MEC_loss.parameters()) if MEC_weight else [])
    optimizer = torch.optim.Adam(params, lr=lr_2)
    teacher_model.eval()
    model.train()

    bar_desc = "Pre-training stage 2 (distillation), avg loss: %.5f, avg compressed ratio: %.3f%%"
    whole_process_time = []
    start_time = time.time()

    teacher_embeds, traj_process_time = [], []
    with torch.no_grad():
        for batch in tqdm(unshuffle_teacher_dataloader, desc="Generating teacher embeds", ncols=60):
            teacher_traj_h, _, process_time = teacher_model.forward_w_processtime_cal(*batch)
            teacher_embeds.append(teacher_traj_h.cpu().numpy())
            traj_process_time.append(process_time)
    teacher_embeds = np.concatenate(teacher_embeds, 0)
    whole_process_time.append(np.sum(traj_process_time))

    dataloader = dataloader(dataset=dataset(teacher_embeds=teacher_embeds))
    with trange(num_epoch_2, desc=bar_desc % (0.0, 100.0), position=0) as bar:
        for epoch_i in bar:
            loss_values, traj_process_time, compressed_ratios = [], [], []
            for batch_i, batch in enumerate(tqdm(dataloader, desc="-->Traversing", leave=False, ncols=60)):
                it = num_iter * epoch_i + batch_i
                # The MEC predictor contains BatchNorm1d, which raises on a batch of one
                # (happens when len(train) % batch_size == 1).
                if MEC_weight and batch[1].size(0) < 2:
                    continue
                optimizer.zero_grad()
                loss, compressed_valid_len, process_time = model.loss(
                    *batch[0], batch[1], MEC_weight, MEC_loss, it)
                if not torch.is_tensor(loss) or not loss.requires_grad:
                    raise ValueError("Stage-2 loss has no trainable term: enable MEC_weight "
                                     "or the mask generator (mask_weight > 0).")
                loss.backward()
                optimizer.step()
                loss_values.append(loss.item())
                traj_process_time.append(process_time)
                compressed_ratios.append(compressed_valid_len / batch[0][1])

            compressed_ratios = torch.cat(compressed_ratios, 0)
            bar.set_description(bar_desc % (np.mean(loss_values), compressed_ratios.mean().item() * 100))
            whole_process_time.append(np.sum(traj_process_time))

    end_time = time.time()
    pretrain_time_2 = ((end_time - start_time) - np.sum(whole_process_time)) / (num_epoch_2 * 60)
    print(f"Stage 2 mean pre-train time: {pretrain_time_2:.4f} min/epoch")
    if pretrain_time_1 is not None:
        total = (pretrain_time_1 * num_epoch_1 + pretrain_time_2 * num_epoch_2) / (num_epoch_1 + num_epoch_2)
        print(f"Overall mean pre-train time: {total:.4f} min/epoch")


# --------------------------------------------------------------------------- #
def finetune_model(model, pred_head, dataloader, num_epoch, lr, ft_encoder=True,
                   weight_decay=0.0, clip=None, val_dataloader=None, valid_metrics_list=None,
                   patience=10, denormalize=False, pred_cols=None):
    valid_metrics_list = valid_metrics_list if valid_metrics_list is not None else []
    pred_head.train()
    if ft_encoder:
        optimizer = torch.optim.Adam(list(model.parameters()) + list(pred_head.parameters()),
                                     lr=lr, weight_decay=weight_decay)
        model.train()
    else:
        optimizer = torch.optim.Adam(pred_head.parameters(), lr=lr, weight_decay=weight_decay)
        model.eval()

    best_model_state, best_pred_head_state, best_epoch = None, None, -1
    wait, min_eval_metric = 0, float("inf")
    bar_desc = "Fine-tuning, avg loss: %.5f"
    with trange(num_epoch, desc=bar_desc % 0.0, position=0) as bar:
        for epoch_i in bar:
            loss_values = []
            for batch in tqdm(dataloader, desc="-->Traversing", leave=False, ncols=60):
                *input_batch, label = batch
                optimizer.zero_grad()
                traj_h = model(*input_batch)
                if not ft_encoder:
                    traj_h = traj_h.detach()
                loss = pred_head.loss(traj_h, label, denormalize)
                loss.backward()
                if clip is not None:
                    if ft_encoder:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    torch.nn.utils.clip_grad_norm_(pred_head.parameters(), clip)
                optimizer.step()
                loss_values.append(loss.item())
            bar.set_description(bar_desc % np.mean(loss_values))

            if val_dataloader is None:
                continue

            predictions, targets = test_model(model, pred_head, val_dataloader, denormalize)
            pred_head.train()
            if ft_encoder:
                model.train()

            if pred_head.pred_type == "regression":
                if denormalize:
                    lng_col, lat_col = pred_cols.index(X_COL), pred_cols.index(Y_COL)
                    valid_metrics = cal_distance_metric(targets, predictions, lng_col, lat_col)
                    eval_metric = valid_metrics["distance_mae"]
                else:
                    valid_metrics = cal_regression_metric(targets, predictions)
                    eval_metric = valid_metrics["mae"]
            elif pred_head.pred_type == "classification":
                valid_metrics = cal_classification_metric(targets, predictions)
                eval_metric = -valid_metrics["acc@1"]
            else:
                raise NotImplementedError(f"No prediction type: {pred_head.pred_type}.")

            print(f"eval_metric={eval_metric}")
            valid_metrics_list.append(valid_metrics)
            if eval_metric < min_eval_metric:  # lower is better
                wait, min_eval_metric, best_epoch = 0, eval_metric, epoch_i
                best_model_state = copy.deepcopy(model).state_dict()
                best_pred_head_state = copy.deepcopy(pred_head).state_dict()
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stop at epoch {epoch_i + 1}.")
                    break

    if best_model_state is not None:
        print(f"Best models are at epoch {best_epoch + 1}; restoring their state.")
        model.load_state_dict(best_model_state)
        pred_head.load_state_dict(best_pred_head_state)


@torch.no_grad()
def test_model(model, pred_head, dataloader, denormalize=False):
    model.eval()
    pred_head.eval()

    predictions, targets = [], []
    for batch in tqdm(dataloader, "Testing", ncols=60):
        *input_batch, target = batch
        pred = pred_head(model(*input_batch))
        if denormalize:
            pred = pred * (pred_head.spatial_border[1] - pred_head.spatial_border[0]).unsqueeze(0) + \
                pred_head.spatial_border[0].unsqueeze(0)
        predictions.append(pred.cpu().numpy())
        targets.append(target.cpu().numpy())
    return np.concatenate(predictions, 0), np.concatenate(targets, 0)


@torch.no_grad()
def test_model_on_search(model, traj_dataloader, qrytgt_dataloader, neg_indices,
                         test_embed_time=False, set_name="test", loop_times=20):
    model.eval()

    qrytgt_embeds = []
    for batch_meta in tqdm(qrytgt_dataloader, desc=f"Embedding queries/targets on {set_name}",
                           total=len(qrytgt_dataloader), ncols=60):
        qrytgt_embeds.append(model(*batch_meta).detach().cpu().numpy())
    qrytgt_embeds = np.concatenate(qrytgt_embeds, 0)
    qry_indices, tgt_indices = TrajectorySearchTestdata.parse_label(len(qrytgt_embeds))

    if test_embed_time:
        embed_time_list, process_time_list = [], []
        with trange(loop_times, desc=f"Timing embedding over {loop_times} loops",
                    position=0, ncols=60) as bar:
            for _ in bar:
                enc_time_list = []
                for batch_meta in traj_dataloader:
                    _, enc_time, process_time = model.forward_on_search_mode(*batch_meta)
                    enc_time_list.append(enc_time)
                    process_time_list.append(process_time)
                embed_time_list.append(np.sum(enc_time_list))
        embed_time_list = np.array(embed_time_list)
        print(f"Total embedding time over {loop_times} loops: {embed_time_list.sum():.3f}s")
        print(f"Mean embedding time on the {set_name} set: {embed_time_list.mean():.3f}s")
        print(f"Min  embedding time on the {set_name} set: {embed_time_list.min():.3f}s")
        print(f"Feature preparation time: {np.sum(process_time_list):.3f}s")

    embeds, whole_enc_time, traj_process_time = [], [], []
    for batch_meta in tqdm(traj_dataloader, desc=f"Embedding database on {set_name}",
                           total=len(traj_dataloader), ncols=60):
        encodes, enc_time, process_time = model.forward_on_search_mode(*batch_meta)
        embeds.append(encodes.detach().cpu().numpy())
        whole_enc_time.append(enc_time)
        traj_process_time.append(process_time)
    print("Embedding time: {:.3f}s".format(np.sum(whole_enc_time)))
    print("Feature preparation time: {:.3f}s".format(np.sum(traj_process_time)))
    embeds = np.concatenate(embeds, 0)

    return TrajectorySearchTestdata.cal_pres_and_labels(
        qrytgt_embeds[qry_indices], qrytgt_embeds[tgt_indices], embeds, neg_indices
    )
