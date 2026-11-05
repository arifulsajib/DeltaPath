#!/usr/bin/env python

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="DeltaPath efficiency benchmark")
parser.add_argument("-s", "--settings", type=str, default="local_test")
parser.add_argument("--cuda", type=int, default=0, help="-1 for CPU")
parser.add_argument("--loops", type=int, default=20)
parser.add_argument("--scaling-only", action="store_true",
                    help="only run the synthetic length-scaling benchmark")
parser.add_argument("--lengths", type=int, nargs="+", default=[30, 60, 120, 240, 480, 960])
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--out", type=str, default=None, help="write results to this JSON file")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda) if args.cuda is not None and args.cuda >= 0 else ""

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import utils  # noqa: E402
from data import (  # noqa: E402
    ROAD_META_DIR, TRAJ_META_DIR, X_COL, Y_COL,
    PretrainPadder, TrajectoryDataset,
)
from models.delta_clip import DeltaClip  # noqa: E402
from models.deltapath import DeltaPath  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() and os.environ["CUDA_VISIBLE_DEVICES"] else "cpu"


def _sync():
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
def length_scaling(encoder_types=("DeltaPath", "GatedDeltaNet2", "Transformer"),
                   lengths=(30, 60, 120, 240, 480, 960), batch=32, repeats=5):
    from models.deltapath import build_seq_encoder

    rows = []
    for encoder_type in encoder_types:
        enc = build_seq_encoder(encoder_type, d_model=256, n_layer=5, d_intermediate=0,
                                aux_feature_size=3, d_state=32, headdim=64, d_inner=256,
                                add_bcdt_item=True, chunk_size=64, device=DEVICE,
                                dtype=torch.float32, verbose=False).to(DEVICE).eval()
        size_mb = utils.cal_model_size(enc)
        for L in lengths:
            x = torch.randn(batch, L, 256, device=DEVICE)
            aux = torch.randn(batch, L, 3, device=DEVICE)
            with torch.no_grad():
                enc(x, aux)  # warm-up
                _sync()
                t0 = time.time()
                for _ in range(repeats):
                    enc(x, aux)
                _sync()
                dt = (time.time() - t0) / repeats
            rows.append({"encoder": encoder_type, "length": L, "batch": batch,
                         "seconds_per_forward": round(dt, 6),
                         "ms_per_trajectory": round(dt / batch * 1000, 4),
                         "encoder_size_MB": round(size_mb, 3)})
            print(f"{encoder_type:<16} L={L:<5} {dt * 1000:8.2f} ms/batch")
        del enc
    return rows


# --------------------------------------------------------------------------- #
def dataset_benchmark(setting, loops):
    train_traj_df = pd.read_hdf(os.path.join(TRAJ_META_DIR, setting["dataset"]["train_traj_df"]),
                                key="trips")
    train_dataset = TrajectoryDataset(traj_df=train_traj_df)
    road_embed = np.load(setting["dataset"]["road_embed"])
    poi_df = pd.read_hdf(setting["dataset"]["poi_df"], key="pois")
    poi_embed = np.load(setting["dataset"]["poi_embed"])
    road_neighbors = np.load(os.path.join(ROAD_META_DIR, setting["dataset"]["road_neighbors"]),
                             allow_pickle=True)

    model = DeltaPath(num_road=road_embed.shape[0],
                      spatial_border=train_dataset.spatial_border,
                      high_order_feature_border=train_dataset.high_order_feature_border,
                      temporal_border=train_dataset.temporal_border,
                      device=DEVICE, **setting["deltapath"]).to(DEVICE).eval()
    teacher = DeltaClip(road_embed=road_embed, road_neighbors=road_neighbors,
                        poi_embed=poi_embed, poi_coors=poi_df[[X_COL, Y_COL]].to_numpy(),
                        spatial_border=train_dataset.spatial_border,
                        high_order_feature_border=train_dataset.high_order_feature_border,
                        device=DEVICE, **setting["deltapath"], **setting["delta_clip"]).to(DEVICE)

    test_traj_df = pd.read_hdf(os.path.join(TRAJ_META_DIR, setting["dataset"]["test_traj_df"]),
                               key="trips")
    test_dataset = TrajectoryDataset(traj_df=test_traj_df)
    loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                        collate_fn=PretrainPadder(device=DEVICE))

    embed_times = []
    with torch.no_grad():
        for batch in loader:  # warm-up
            model.forward_on_search_mode(*batch)
            break
        for _ in range(loops):
            _sync()
            t0 = time.time()
            for batch in loader:
                model.forward_on_search_mode(*batch)
            _sync()
            embed_times.append(time.time() - t0)

    result = {
        "settings": args.settings,
        "device": DEVICE,
        "n_test_trajectories": len(test_dataset),
        "deltapath_size_MB": round(utils.cal_model_size(model), 4),
        "deltaclip_size_MB": round(utils.cal_model_size(teacher), 4),
        "embed_time_mean_s": round(float(np.mean(embed_times)), 4),
        "embed_time_min_s": round(float(np.min(embed_times)), 4),
        "embed_time_std_s": round(float(np.std(embed_times)), 4),
    }
    print(json.dumps(result, indent=2))
    return result


# --------------------------------------------------------------------------- #
def main():
    results = {"device": DEVICE}
    if not args.scaling_only:
        with open(os.path.join("settings", f"{args.settings}.json")) as fp:
            settings = json.load(fp)
        results["dataset_benchmark"] = [dataset_benchmark(s, args.loops) for s in settings]

    print("\n--- length scaling (synthetic) ---")
    results["length_scaling"] = length_scaling(lengths=args.lengths)

    if args.out:
        with open(args.out, "w") as fp:
            json.dump(results, fp, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
