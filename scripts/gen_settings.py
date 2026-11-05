#!/usr/bin/env python
"""Generate every JSON experiment configuration under ./settings.

    python scripts/gen_settings.py

Edit the constants below and re-run instead of hand-editing the JSON files.
"""

import json
import os

SETTINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings")

# --------------------------------------------------------------------------- #
# Dataset registry.  num_road must equal the number of rows of road_embed.
# --------------------------------------------------------------------------- #
DATASETS = {
    "chengdu": dict(num_road=4315, poi_df="./raw_data/chengdu.h5",
                    poi_embed="./raw_data/chengdu_poi_embed.npy",
                    road_embed="./raw_data/chengdu_road_embed.npy"),
    "xian": dict(num_road=3392, poi_df="./raw_data/xian.h5",
                 poi_embed="./raw_data/xian_poi_embed.npy",
                 road_embed="./raw_data/xian_road_embed.npy"),
    # Synthetic city produced by scripts/make_toy_data.py (default 12x12 grid).
    "toy": dict(num_road=528, poi_df="./samples/toy.h5",
                poi_embed="./samples/toy_poi_embed.npy",
                road_embed="./samples/toy_road_embed.npy"),
}

# Optimal hyper-parameters: L = 5, E = 256, N = 32, H = 4 (headdim 64, d_inner 256).
DELTAPATH = dict(embed_size=64, d_model=256, n_layer=5, d_state=32, headdim=64,
                 d_inner=256, add_bcdt_item=True, chunk_size=64, use_qk_l2norm=True)
DELTACLIP = dict(road_weight=1, poi_weight=1, pois_dist_thres=300, poi_max_neighbors=10,
                 road_transW_lambda=1.0, poi_distW_lambda=0.5)

BATCH = 128
EPOCHS = 15
LR1, LR2 = 1e-3, 1e-4
SEED = 2025


def dataset_block(city, *, pretrain_suffix="compressed_keep-0", finetune_suffix="compressed_keep-5",
                  test_suffix="compressed_keep-5"):
    d = DATASETS[city]
    return {
        "city": city,
        "train_traj_df": f"{city}_train.h5",
        "poi_df": d["poi_df"],
        "poi_embed": d["poi_embed"],
        "road_embed": d["road_embed"],
        "road_neighbors": f"{city}_road_neighbors_info.npz",
        "compressed_pretrain_traj_df": f"{city}_train_{pretrain_suffix}.h5",
        "compressed_finetune_traj_df": f"{city}_train_{finetune_suffix}.h5",
        "val_traj_df": f"{city}_valid_{finetune_suffix}.h5",
        "test_traj_df": f"{city}_test_{test_suffix}.h5",
    }


TASKS = {
    # name -> (pred_cols, pred_type, output_size_fn, task)
    "dp_road": (["road"], "classification", lambda d: d["num_road"], "destination_prediction"),
    "dp_gps": (["lng", "lat"], "regression", lambda d: 2, "destination_prediction"),
    "ate": (["delta_t"], "regression", lambda d: 1, "arrival_time_estimation"),
}


def pred_head_block(city, task, d_model):
    pred_cols, pred_type, out_fn, _ = TASKS[task]
    # The head reads the encoder output, so it must follow d_model (this used to be
    # fixed to the default 256, which broke the embedding-size sweep).
    return {"input_size": d_model, "hidden_size": d_model,
            "output_size": out_fn(DATASETS[city]), "dropout": 0.3, "pred_type": pred_type}


def pretrain_block(*, load=False, save=True, mec_weight=0.5, batch=BATCH,
                   epochs=EPOCHS, teacher_savename=None, pretrain_save_name=None):
    block = {
        "load": load, "save": save,
        "dataloader": {"batch_size": batch, "shuffle": True, "num_workers": 0},
        "padder": {},
        "config": {
            "num_epoch_1": epochs, "num_epoch_2": epochs, "lr_1": LR1, "lr_2": LR2,
            "MEC_weight": mec_weight,
            "MEC_config": {"hidden_size": 256, "n": 5, "warmup_epoch": 5, "eps": 512},
        },
    }
    if teacher_savename:
        block["teacher_model_savename"] = teacher_savename
    if pretrain_save_name:
        block["pretrain_save_name"] = pretrain_save_name
    return block


def finetune_block(task, city, *, ft_encoder=True, epochs=20, patience=5, batch=BATCH, prop=None):
    pred_cols, _, _, _ = TASKS[task]
    block = {
        "load": False, "save": True,
        "dataloader": {"batch_size": batch, "shuffle": True, "num_workers": 0},
        "val_dataloader": {"batch_size": batch, "num_workers": 0},
        "padder": {"name": "dp", "params": {"pred_len": 5, "pred_cols": pred_cols}},
        "config": {"num_epoch": epochs, "lr": LR1, "ft_encoder": ft_encoder, "patience": patience},
    }
    if prop is not None:
        block["prop"] = prop
    return block


def test_block(task, *, batch=BATCH, save=True):
    pred_cols, _, _, task_name = TASKS[task]
    return {
        "save": save, "task": task_name,
        "dataloader": {"batch_size": batch, "num_workers": 0},
        "padder": {"name": "dp", "params": {"pred_len": 5, "pred_cols": pred_cols}},
    }


def sts_test_block(*, batch=BATCH, num_target=1000, num_negative=5000, test_embed_time=True):
    return {
        "task": "search", "save": True, "test_embed_time": test_embed_time,
        "search_data_params": {"num_target": num_target, "num_negative": num_negative,
                               "same_OD_thres": 50},
        "dataloader": {"batch_size": batch, "num_workers": 0},
        "padder": {"name": "pretrain", "params": {}},
    }


def make_setting(name, city, task, *, deltapath_over=None, deltaclip_over=None,
                 dataset_over=None, pretrain=None, finetune=True, test=True,
                 ft_encoder=True, batch=BATCH, prop=None):
    dp = dict(DELTAPATH)
    dp.update(deltapath_over or {})
    dc = dict(DELTACLIP)
    dc.update(deltaclip_over or {})
    ds = dataset_block(city)
    ds.update(dataset_over or {})

    setting = {
        "save_name": name, "seed": SEED,
        "dataset": ds,
        "delta_clip": dc,
        "deltapath": dp,
        "pred_head": pred_head_block(city, task if task in TASKS else "dp_road", dp["d_model"]),
    }
    if pretrain is not None:
        setting["pretrain"] = pretrain
    if finetune and task in TASKS:
        setting["finetune"] = finetune_block(task, city, ft_encoder=ft_encoder,
                                             batch=batch, prop=prop)
    if test:
        setting["test"] = sts_test_block(batch=batch) if task == "sts" else test_block(task, batch=batch)
    return setting


def write(path, settings):
    full = os.path.join(SETTINGS_DIR, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as fp:
        json.dump(settings, fp, indent=4)
    print("wrote", os.path.relpath(full, SETTINGS_DIR))


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(SETTINGS_DIR, exist_ok=True)

    # ---- quick smoke tests on the synthetic toy city ---- #
    small = make_setting(
        "local_test", "toy", "dp_road",
        deltapath_over={}, pretrain=pretrain_block(batch=16, epochs=2), batch=16)
    small["finetune"]["config"].update(num_epoch=2, patience=2)
    write("local_test.json", [small])

    small_search = make_setting(
        "local_test_search", "toy", "sts",
        dataset_over={"test_traj_df": "toy_test_compressed_keep-0.h5"},
        pretrain=pretrain_block(load=True, teacher_savename="local_test",
                                pretrain_save_name="local_test", batch=16),
        finetune=False, batch=16)
    small_search["test"]["search_data_params"].update(num_target=30, num_negative=17)
    write("local_test_search.json", [small_search])

    # Smoke test of the architectural variants on the toy city, so that the
    # ablation code paths are exercised without the full datasets.
    variants = []
    for tag, over in (("gdn2", {"encoder_type": "GatedDeltaNet2"}),
                      ("transformer", {"encoder_type": "Transformer"}),
                      ("nomask", {"use_mask_generator": False})):
        s = make_setting(f"local_test_{tag}", "toy", "dp_road",
                         deltapath_over=over,
                         pretrain=pretrain_block(batch=16, epochs=1), batch=16)
        s["finetune"]["config"].update(num_epoch=1, patience=1)
        variants.append(s)
    write("local_test_variants.json", variants)

    # ---- main experiments ---- #
    for city in ("chengdu", "xian"):
        # Stage 1 + stage 2 pre-training, saved once and reused by every task.
        write(f"{city}_pretrain.json", [
            make_setting(f"{city}_pretrain", city, "dp_road",
                         pretrain=pretrain_block(), finetune=False, test=False)
        ])

        for task in ("dp_road", "dp_gps", "ate"):
            # fine-tune setting (default)
            write(f"{city}_{task}.json", [
                make_setting(f"{city}_{task}", city, task,
                             pretrain=pretrain_block(load=True,
                                                     pretrain_save_name=f"{city}_pretrain"))
            ])
            # without fine-tune ("w/o ft"): only the predictor is trained
            write(f"{city}_{task}_noft.json", [
                make_setting(f"{city}_{task}_noft", city, task, ft_encoder=False,
                             pretrain=pretrain_block(load=True,
                                                     pretrain_save_name=f"{city}_pretrain"))
            ])

        # Similar trajectory search: parameters are frozen after pre-training.
        write(f"{city}_sts.json", [
            make_setting(f"{city}_sts", city, "sts",
                         dataset_over={"test_traj_df": f"{city}_test_compressed_keep-0.h5"},
                         pretrain=pretrain_block(load=True,
                                                 pretrain_save_name=f"{city}_pretrain"),
                         finetune=False)
        ])

    # ---- ablation variants (Chengdu) ---- #
    ablations = {
        # Replace DeltaPath blocks with plain Gated DeltaNet-2 blocks
        "v_gdn2": dict(deltapath_over={"encoder_type": "GatedDeltaNet2"}),
        # Replace every sequence mixer with vanilla Transformer layers.
        "v_transformer": dict(deltapath_over={"encoder_type": "Transformer"}),
        # Pre-train the teacher with reconstruction instead of the travel-purpose views.
        "wo_purpose": dict(deltaclip_over={"pretrain_task": "reconstruct"}),
        # Drop the distillation objective in stage 2 (mask penalty only).
        "wo_kd": dict(mec_weight=0.0),
        # Remove the mask generator; embed the full (rule-filtered) trajectory.
        "wo_compress": dict(deltapath_over={"use_mask_generator": False}),
        # Replace the mask generator with Douglas-Peucker compression.
        "mg_trans_dp": dict(deltapath_over={"use_mask_generator": False},
                            dataset_over={
                                "compressed_pretrain_traj_df": "chengdu_train_RDPcompressed_keep-0.h5",
                                "compressed_finetune_traj_df": "chengdu_train_RDPcompressed_keep-5.h5",
                                "val_traj_df": "chengdu_valid_RDPcompressed_keep-5.h5",
                                "test_traj_df": "chengdu_test_RDPcompressed_keep-5.h5"}),
        # Replace the mask generator with uniform down-sampling at a 60% keep ratio.
        "mg_trans_ds": dict(deltapath_over={"use_mask_generator": False},
                            dataset_over={
                                "compressed_pretrain_traj_df": "chengdu_train_DScompressed-0.6_keep-0.h5",
                                "compressed_finetune_traj_df": "chengdu_train_DScompressed-0.6_keep-5.h5",
                                "val_traj_df": "chengdu_valid_DScompressed-0.6_keep-5.h5",
                                "test_traj_df": "chengdu_test_DScompressed-0.6_keep-5.h5"}),
        # Remove the rule-based filtering preprocessing feeding the mask generator.
        "wo_filter": dict(dataset_over={
            "compressed_pretrain_traj_df": "chengdu_train.h5",
            "compressed_finetune_traj_df": "chengdu_train.h5",
            "val_traj_df": "chengdu_valid.h5",
            "test_traj_df": "chengdu_test.h5"}),
    }
    for name, over in ablations.items():
        mec_weight = over.pop("mec_weight", 0.5)
        # Pre-train the variant once, then reuse the checkpoint for every task.
        write(f"ablation/{name}_pretrain.json", [
            make_setting(f"ablation_{name}_pretrain", "chengdu", "dp_road",
                         pretrain=pretrain_block(mec_weight=mec_weight),
                         finetune=False, test=False, **over)
        ])
        reuse = pretrain_block(load=True, pretrain_save_name=f"ablation_{name}_pretrain")
        for task in ("dp_road", "dp_gps", "ate"):
            write(f"ablation/{name}_{task}.json", [
                make_setting(f"ablation_{name}_{task}", "chengdu", task,
                             pretrain=dict(reuse), **over)
            ])
        write(f"ablation/{name}_sts.json", [
            make_setting(f"ablation_{name}_sts", "chengdu", "sts",
                         pretrain=dict(reuse), finetune=False, **over)
        ])

    # ---- loss-weight study ---- #
    for mec_w, mask_w in ((1.0, 0.0), (0.7, 0.3), (0.3, 0.7), (0.5, 0.5)):
        tag = f"mec{mec_w}_mask{mask_w}".replace(".", "")
        write(f"sweep/weights_{tag}.json", [
            make_setting(f"weights_{tag}", "chengdu", "dp_road",
                         deltapath_over={"mask_weight": mask_w},
                         pretrain=pretrain_block(mec_weight=mec_w))
        ])

    # ---- hyper-parameter study ---- #
    for L in (2, 3, 4, 5, 6):
        write(f"sweep/layers_L{L}.json", [
            make_setting(f"layers_L{L}", "chengdu", "dp_road",
                         deltapath_over={"n_layer": L}, pretrain=pretrain_block())
        ])
    for E in (64, 128, 256, 512):
        write(f"sweep/embed_E{E}.json", [
            make_setting(f"embed_E{E}", "chengdu", "dp_road",
                         deltapath_over={"d_model": E, "d_inner": E},
                         pretrain=pretrain_block())
        ])
    for N in (8, 16, 32, 64, 128, 256):
        write(f"sweep/state_N{N}.json", [
            make_setting(f"state_N{N}", "chengdu", "dp_road",
                         deltapath_over={"d_state": N}, pretrain=pretrain_block())
        ])
    for H in (2, 4, 8, 16, 32, 64):
        write(f"sweep/heads_H{H}.json", [
            make_setting(f"heads_H{H}", "chengdu", "dp_road",
                         deltapath_over={"headdim": DELTAPATH["d_inner"] // H},
                         pretrain=pretrain_block())
        ])

    # ---- scalability study (training-set proportion) ---- #
    for prop in (0.2, 0.4, 0.6, 0.8):
        tag = str(prop).replace(".", "")
        write(f"sweep/scale_prop{tag}.json", [
            make_setting(f"scale_prop{tag}", "chengdu", "dp_road", prop=prop,
                         pretrain=pretrain_block(load=True,
                                                 pretrain_save_name="chengdu_pretrain"))
        ])


if __name__ == "__main__":
    main()
