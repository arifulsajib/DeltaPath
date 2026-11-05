
import json
import os
from argparse import ArgumentParser
from functools import partial

import numpy as np
import pandas as pd

parser = ArgumentParser(description="DeltaPath: efficient, semantic-rich vehicle trajectory learning")
parser.add_argument("-s", "--settings", type=str, default="local_test",
                    help="name of the settings file in ./settings (without .json)")
parser.add_argument("--cuda", type=int, default=0,
                    help="index of the CUDA device to use; pass -1 to force CPU")
args = parser.parse_args()

if args.cuda is not None and args.cuda >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import utils  # noqa: E402
from data import (  # noqa: E402
    DT_COL, ROAD_COL, ROAD_META_DIR, SEARCH_META_DIR, TRAJ_META_DIR, X_COL, Y_COL,
    DpPadder, PretrainDatasetsUnion, PretrainPadder, SearchPadder, TrajectoryDataset,
    TrajectorySearchDataset, UnionPretrainPadder, fetch_task_padder, load_trajSearch_testdata,
)
from models.delta_clip import DeltaClip  # noqa: E402
from models.deltapath import DeltaPath  # noqa: E402
from models.predictor import MlpPredictor  # noqa: E402
from pipeline import finetune_model, pretrain_model, test_model, test_model_on_search  # noqa: E402

SETTINGS_CACHE_DIR = os.environ.get("SETTINGS_CACHE_DIR", os.path.join("settings", "cache"))
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "saved_model")
PRED_SAVE_DIR = os.environ.get("PRED_SAVE_DIR", "predictions")


def main():
    device = "cuda:0" if torch.cuda.is_available() and os.environ["CUDA_VISIBLE_DEVICES"] else "cpu"

    datetime_key = utils.get_datetime_key()
    print(f"==== START EXPERIMENT, DATETIME KEY: {datetime_key}, DEVICE: {device} ====")

    with open(os.path.join("settings", f"{args.settings}.json"), "r") as fp:
        settings = json.load(fp)
    utils.create_if_noexists(SETTINGS_CACHE_DIR)
    with open(os.path.join(SETTINGS_CACHE_DIR, f"{datetime_key}.json"), "w") as fp:
        json.dump(settings, fp, indent=2)

    for setting_i, setting in enumerate(settings):
        print(f"=== SETTING {setting_i}/{len(settings)}: {setting.get('save_name')} ===")
        SAVE_NAME = setting.get("save_name", None)

        if "seed" in setting:
            utils.setup_seed(setting["seed"])

        # ---------------- data ---------------- #
        train_traj_df = pd.read_hdf(
            os.path.join(TRAJ_META_DIR, setting["dataset"]["train_traj_df"]), key="trips")
        train_dataset = TrajectoryDataset(traj_df=train_traj_df)

        road_embed = np.load(setting["dataset"]["road_embed"])
        poi_df = pd.read_hdf(setting["dataset"]["poi_df"], key="pois")
        poi_embed = np.load(setting["dataset"]["poi_embed"])
        poi_coors = poi_df[[X_COL, Y_COL]].to_numpy()
        road_neighbors = np.load(
            os.path.join(ROAD_META_DIR, setting["dataset"]["road_neighbors"]), allow_pickle=True)

        if setting["pred_head"]["pred_type"] == "classification" and \
                setting["pred_head"]["output_size"] != road_embed.shape[0]:
            raise ValueError(f"pred_head.output_size={setting['pred_head']['output_size']} but the road "
                             f"network has {road_embed.shape[0]} segments; they must match.")
        if setting["pred_head"]["input_size"] != setting["deltapath"]["d_model"]:
            raise ValueError("pred_head.input_size must equal deltapath.d_model.")

        # ---------------- models ---------------- #
        deltapath_model = DeltaPath(
            num_road=road_embed.shape[0],
            spatial_border=train_dataset.spatial_border,
            high_order_feature_border=train_dataset.high_order_feature_border,
            temporal_border=train_dataset.temporal_border,
            device=device, **setting["deltapath"]).to(device)
        delta_clip = DeltaClip(
            road_embed=road_embed, road_neighbors=road_neighbors,
            poi_embed=poi_embed, poi_coors=poi_coors,
            spatial_border=train_dataset.spatial_border,
            high_order_feature_border=train_dataset.high_order_feature_border,
            device=device, **setting["deltapath"], **setting["delta_clip"]).to(device)
        pred_head = MlpPredictor(spatial_border=train_dataset.spatial_border,
                                 **setting["pred_head"]).to(device)
        print(f"DeltaPath model size: {utils.cal_model_size(deltapath_model):.3f} MBytes.")
        print(f"DeltaCLIP model size: {utils.cal_model_size(delta_clip):.3f} MBytes.")

        # ---------------- pre-training ---------------- #
        if "pretrain" in setting:
            teacher_prepared = False
            teacher_savename = setting["pretrain"].get("teacher_model_savename", None)
            if teacher_savename is not None:
                delta_clip.load_state_dict(torch.load(
                    os.path.join(MODEL_CACHE_DIR, f"{teacher_savename}_deltaclip.pretrain"),
                    map_location=device))
                teacher_prepared = True

            if setting["pretrain"].get("load", False):
                PRETRAIN_SAVE_NAME = setting["pretrain"].get("pretrain_save_name", SAVE_NAME)
                if not teacher_prepared:
                    delta_clip.load_state_dict(torch.load(
                        os.path.join(MODEL_CACHE_DIR, f"{PRETRAIN_SAVE_NAME}_deltaclip.pretrain"),
                        map_location=device))
                deltapath_model.load_state_dict(torch.load(
                    os.path.join(MODEL_CACHE_DIR, f"{PRETRAIN_SAVE_NAME}_deltapath.pretrain"),
                    map_location=device))
            else:
                compressed_pretrain_traj_df = pd.read_hdf(
                    os.path.join(TRAJ_META_DIR, setting["dataset"]["compressed_pretrain_traj_df"]),
                    key="trips")
                compressed_pretrain_dataset = TrajectoryDataset(traj_df=compressed_pretrain_traj_df)
                # Teacher embeddings are paired with compressed trips by position,
                # so both files must list the same trips in the same order.
                if not np.array_equal(compressed_pretrain_dataset.traj_ids, train_dataset.traj_ids):
                    raise ValueError("compressed_pretrain_traj_df and train_traj_df do not contain "
                                     "the same trips in the same order.")

                pretrain_dataset = partial(PretrainDatasetsUnion,
                                           compressed_dataset=compressed_pretrain_dataset)
                pretrain_dataloader = partial(
                    DataLoader,
                    collate_fn=UnionPretrainPadder(device=device, **setting["pretrain"]["padder"]),
                    **setting["pretrain"]["dataloader"])

                teacher_dataloader = DataLoader(
                    train_dataset,
                    collate_fn=PretrainPadder(device=device, **setting["pretrain"]["padder"]),
                    **setting["pretrain"]["dataloader"]) if not teacher_prepared else None

                loader_cfg = dict(setting["pretrain"]["dataloader"])
                loader_cfg.pop("shuffle", None)
                unshuffle_teacher_dataloader = DataLoader(
                    train_dataset, shuffle=False,
                    collate_fn=PretrainPadder(device=device, **setting["pretrain"]["padder"]),
                    **loader_cfg)

                utils.create_if_noexists(MODEL_CACHE_DIR)
                pretrain_model(
                    model=deltapath_model, teacher_model=delta_clip,
                    dataset=pretrain_dataset, dataloader=pretrain_dataloader,
                    unshuffle_teacher_dataloader=unshuffle_teacher_dataloader,
                    teacher_dataloader=teacher_dataloader, teacher_prepared=teacher_prepared,
                    teacher_savename=os.path.join(MODEL_CACHE_DIR, f"{SAVE_NAME}_deltaclip.pretrain"),
                    **setting["pretrain"]["config"])

                if setting["pretrain"].get("save", True):
                    if not teacher_prepared:
                        torch.save(delta_clip.state_dict(),
                                   os.path.join(MODEL_CACHE_DIR, f"{SAVE_NAME}_deltaclip.pretrain"))
                    torch.save(deltapath_model.state_dict(),
                               os.path.join(MODEL_CACHE_DIR, f"{SAVE_NAME}_deltapath.pretrain"))

        # ---------------- fine-tuning ---------------- #
        if "finetune" in setting:
            if setting["finetune"].get("check_teacher", False):
                print("\nFine-tuning the teacher (DeltaCLIP):")
                model, model_name, finetune_dataset = delta_clip, "deltaclip", train_dataset
            else:
                print("\nFine-tuning DeltaPath:")
                model, model_name = deltapath_model, "deltapath"
                compressed_finetune_traj_df = pd.read_hdf(
                    os.path.join(TRAJ_META_DIR, setting["dataset"]["compressed_finetune_traj_df"]),
                    key="trips")
                finetune_dataset = TrajectoryDataset(traj_df=compressed_finetune_traj_df)

            if setting["finetune"].get("load", False):
                model.load_state_dict(torch.load(
                    os.path.join(MODEL_CACHE_DIR, f"{SAVE_NAME}_{model_name}.finetune"),
                    map_location=device))
                pred_head.load_state_dict(torch.load(
                    os.path.join(MODEL_CACHE_DIR, f"{SAVE_NAME}_predhead.finetune"),
                    map_location=device))
            else:
                if setting["finetune"].get("prop", 1.0) < 1.0:
                    print("Training set proportion =", setting["finetune"]["prop"])
                    finetune_dataset.use_partial_data(setting["finetune"]["prop"])

                finetune_padder = fetch_task_padder(
                    padder_name=setting["finetune"]["padder"]["name"], device=device,
                    padder_params=setting["finetune"]["padder"]["params"])
                finetune_dataloader = DataLoader(finetune_dataset, collate_fn=finetune_padder,
                                                 **setting["finetune"]["dataloader"])

                val_dataloader = None
                if setting["dataset"].get("val_traj_df", False):
                    val_traj_df = pd.read_hdf(
                        os.path.join(TRAJ_META_DIR, setting["dataset"]["val_traj_df"]), key="trips")
                    val_dataloader = DataLoader(TrajectoryDataset(traj_df=val_traj_df),
                                                collate_fn=finetune_padder,
                                                **setting["finetune"]["val_dataloader"])
                valid_metrics_list = []

                if_denormalize, pred_cols = False, None
                if isinstance(finetune_padder, DpPadder) and \
                        sorted(finetune_padder.pred_cols) == sorted([Y_COL, X_COL]):
                    if_denormalize, pred_cols = True, finetune_padder.pred_cols

                finetune_model(model=model, pred_head=pred_head, dataloader=finetune_dataloader,
                               val_dataloader=val_dataloader, valid_metrics_list=valid_metrics_list,
                               denormalize=if_denormalize, pred_cols=pred_cols,
                               **setting["finetune"]["config"])

                if setting["finetune"].get("save", True):
                    utils.create_if_noexists(MODEL_CACHE_DIR)
                    torch.save(model.state_dict(),
                               os.path.join(MODEL_CACHE_DIR, f"{SAVE_NAME}_{model_name}.finetune"))
                    torch.save(pred_head.state_dict(),
                               os.path.join(MODEL_CACHE_DIR, f"{SAVE_NAME}_predhead.finetune"))
                if len(valid_metrics_list):
                    utils.create_if_noexists(os.path.join(PRED_SAVE_DIR, SAVE_NAME))
                    pd.concat(valid_metrics_list, axis=1).T.to_csv(
                        os.path.join(PRED_SAVE_DIR, SAVE_NAME, "valid_metrics.csv"), index=False)

        # ---------------- evaluation ---------------- #
        if "test" in setting:
            test_traj_df = pd.read_hdf(
                os.path.join(TRAJ_META_DIR, setting["dataset"]["test_traj_df"]), key="trips")
            test_dataset = TrajectoryDataset(traj_df=test_traj_df)
            test_padder = fetch_task_padder(
                padder_name=setting["test"]["padder"].get("name", "pretrain"), device=device,
                padder_params=setting["test"]["padder"]["params"])
            test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=test_padder,
                                         **setting["test"]["dataloader"])

            if setting["test"].get("check_teacher", False):
                print("\nTesting the teacher (DeltaCLIP):")
                model = delta_clip
            else:
                print("\nTesting DeltaPath:")
                model = deltapath_model

            down_task = setting["test"].get("task", "destination_prediction")

            if down_task in ("destination_prediction", "arrival_time_estimation"):
                if_denormalize = sorted(test_padder.pred_cols) == sorted([Y_COL, X_COL])
                predictions, targets = test_model(model=model, pred_head=pred_head,
                                                  dataloader=test_dataloader,
                                                  denormalize=if_denormalize)
                print(f"{SAVE_NAME} predictions: {predictions.shape}")

                if test_padder.pred_cols == [ROAD_COL]:
                    cal_metric, metric_filename = utils.cal_classification_metric, "road_classification"
                elif sorted(test_padder.pred_cols) == sorted([Y_COL, X_COL]):
                    lng_col = test_padder.pred_cols.index(X_COL)
                    lat_col = test_padder.pred_cols.index(Y_COL)
                    cal_metric = partial(utils.cal_distance_metric, lng_col=lng_col, lat_col=lat_col)
                    metric_filename = "gps_regression"
                elif test_padder.pred_cols == [DT_COL]:
                    cal_metric, metric_filename = utils.cal_regression_metric, "delta_t_regression"
                else:
                    raise NotImplementedError(f'No predict columns called "{test_padder.pred_cols}".')
                metric = cal_metric(targets, predictions)
                print(f"Test metric for {test_padder.pred_cols}:")
                print(metric)

            elif down_task == "search":
                eval_dataset = os.path.basename(setting["dataset"]["test_traj_df"]).split(".")[0]
                search_meta_dir = os.path.join(SEARCH_META_DIR, eval_dataset)
                try:
                    _, hopqrytgt, neg_indices = load_trajSearch_testdata(
                        search_meta_dir, **setting["test"]["search_data_params"])
                except FileNotFoundError:
                    raise SystemExit("No meta for similar trajectory search; run data.py first.")

                search_loader = partial(DataLoader, shuffle=False,
                                        collate_fn=SearchPadder(device=device),
                                        **setting["test"]["dataloader"])
                trajqrytgt_dataloader = search_loader(TrajectorySearchDataset(hopqrytgt))

                predictions, targets = test_model_on_search(
                    model=model, traj_dataloader=test_dataloader,
                    qrytgt_dataloader=trajqrytgt_dataloader, neg_indices=neg_indices,
                    test_embed_time=setting["test"].get("test_embed_time", False))
                metric_filename = "similar_trajectory_search"
                metric = utils.cal_search_metric(predictions)
                print("Test metric for similar trajectory search:")
                print(metric)
            else:
                raise NotImplementedError(f'No downstream task called "{down_task}".')

            if setting["test"].get("save", False):
                utils.create_if_noexists(os.path.join(PRED_SAVE_DIR, SAVE_NAME))
                np.save(os.path.join(PRED_SAVE_DIR, SAVE_NAME, "predictions.npy"), predictions)
                np.save(os.path.join(PRED_SAVE_DIR, SAVE_NAME, "targets.npy"), targets)
                metric.to_hdf(os.path.join(PRED_SAVE_DIR, SAVE_NAME, f"{metric_filename}.h5"),
                              key="metric", format="table")


if __name__ == "__main__":
    main()
