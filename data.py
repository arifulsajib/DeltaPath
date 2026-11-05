
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import utils

TRAJ_ID_COL = "trip"
X_COL = "lng"
Y_COL = "lat"
T_COL = "timestamp"
DT_COL = "delta_t"
ROAD_COL = "road"
V_COL = "speed"
ACC_COL = "acc"
ANGLE_COL = "courseAngle"
FEATURE_COLS = [X_COL, Y_COL, T_COL, DT_COL, ROAD_COL, V_COL, ACC_COL, ANGLE_COL]

COL_I = {
    "spatial": [0, 1],
    "temporal": [2, 3],
    "road": 4,
    "high_order_features": [5, 6, 7],
}

FEATURE_PAD = 0
# Downstream tasks hide the last PRED_LEN points, so a trip needs more than
# PRED_LEN points to leave any input.  (With 5, a 5-point trip crashed the
# keep-5 compression step and would have produced an empty input sequence.)
PRED_LEN = 5
MIN_TRIP_LEN = PRED_LEN + 1
MAX_TRIP_LEN = 120
# Re-base timestamps so that they stay small enough to survive the cast to float32.
BASE_TIME = pd.to_datetime("2018-09-29 00:00:00")

RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", "./raw_data")
# Synthetic toy cities (scripts/make_toy_data.py) live here.
SAMPLE_DATA_DIR = os.environ.get("SAMPLE_DATA_DIR", "./samples")
PROCESSED_DATA_DIR = os.environ.get("PROCESSED_DATA_DIR", "./processed_data")
TRAJ_META_DIR = os.path.join(PROCESSED_DATA_DIR, "traj_meta")
ROAD_META_DIR = os.path.join(PROCESSED_DATA_DIR, "road_meta")
SEARCH_META_DIR = os.path.join(PROCESSED_DATA_DIR, "search_meta")


def _require(pkg_name, import_name=None):
    import importlib

    try:
        return importlib.import_module(import_name or pkg_name)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"'{pkg_name}' is required for this step. Install it with `pip install {pkg_name}`."
        ) from exc


# --------------------------------------------------------------------------- #
class DataPreprocessor:

    def __init__(self, city):
        self.city = city
        # Synthetic toy cities live in ./samples, real datasets in ./raw_data.
        self.raw_data_dir = SAMPLE_DATA_DIR if city.startswith("toy") else RAW_DATA_DIR
        self.dataset_category = ["train", "valid", "test"]
        self.base_timestamp = get_last_Monday(BASE_TIME)
        utils.create_if_noexists(PROCESSED_DATA_DIR)

    # ------------------------------------------------------------------ #
    def preprocess_traj(self):
        utils.create_if_noexists(TRAJ_META_DIR)

        for category in self.dataset_category:
            data_filename = f"{self.city}_{category}.h5"
            trajs_df = pd.read_hdf(os.path.join(self.raw_data_dir, data_filename), key="trips")
            trajs_df[T_COL] = trajs_df["time"].apply(lambda x: x.timestamp())

            preprocessed_trajs = []
            groups = trajs_df.groupby(TRAJ_ID_COL)
            for _, group in tqdm(groups, desc=f"Preprocessing trips [{category}]",
                                 total=len(groups), ncols=70):
                if (not group.isna().any().any()) and MIN_TRIP_LEN <= group.shape[0] <= MAX_TRIP_LEN:
                    preprocessed_trajs.append(self.cal_high_order_features(group))

            pd.concat(preprocessed_trajs).to_hdf(
                os.path.join(TRAJ_META_DIR, data_filename), key="trips"
            )

    @staticmethod
    def cal_high_order_features(group):
        group = group.copy()
        group[DT_COL] = group[T_COL] - group[T_COL].iloc[0]

        ST_features = group[[X_COL, Y_COL, T_COL]].to_numpy()
        time_diff = ST_features[1:, -1] - ST_features[:-1, -1]
        time_diff = np.where(time_diff != 0, time_diff, 1)  # avoid division by zero

        dist = utils.cal_geo_distance(ST_features[:-1, :-1], ST_features[1:, :-1])
        group["distance"] = np.insert(dist, 0, values=-1)

        speed = dist / time_diff
        group[V_COL] = speed = np.append(speed, values=0)  # the destination is at rest

        acc = (speed[1:] - speed[:-1]) / time_diff
        group[ACC_COL] = np.append(acc, values=0)

        courseAngle = utils.cal_courseAngle(ST_features[:-1, :-1], ST_features[1:, :-1])
        group[ANGLE_COL] = np.append(courseAngle, values=0)
        return group

    # ------------------------------------------------------------------ #
    def compress_traj(self, dataset_mode="train", pred_len=0):
        data_filename = f"{self.city}_{dataset_mode}"
        trajs_df = pd.read_hdf(os.path.join(TRAJ_META_DIR, data_filename + ".h5"), key="trips")

        compressed_trajs = []
        groups = trajs_df.groupby(TRAJ_ID_COL)
        for _, group in tqdm(groups, desc=f"Compressing trips [{dataset_mode}]",
                             total=len(groups), ncols=70):
            compressed_trajs.append(self.gen_compressed_traj(group, pred_len))

        pd.concat(compressed_trajs).to_hdf(
            os.path.join(TRAJ_META_DIR, f"{data_filename}_compressed_keep-{pred_len}.h5"), key="trips"
        )

    def gen_compressed_traj(self, group, pred_len):
        # Nothing to compress: only the origin and the kept prediction targets remain.
        if len(group) <= pred_len + 1:
            return group.reset_index(drop=True)

        drop_index = self.find_intermediate_stay_point(group, pred_len)

        # Some trajectories never move; keep at least one point besides the origin.
        if len(group) - len(drop_index) == pred_len + 1:
            if len(drop_index):
                drop_index.pop()
            group = group.drop(drop_index).reset_index(drop=True)
            return group

        group = group.drop(drop_index).reset_index(drop=True)
        group = group.drop(self.find_steady_pace_point(group, pred_len)).reset_index(drop=True)
        return self.cal_high_order_features(group)

    @staticmethod
    def find_intermediate_stay_point(group, pred_len):
        last_index_for_compress = group.index[-pred_len - 1] if pred_len else group.index[-1]

        drop_index = []
        new_stay = True
        for index in group[group.distance == 0].index:
            if new_stay:
                start = index - 1
                if start >= last_index_for_compress:
                    break
                new_stay = False

            if group.loc[index, V_COL]:  # the vehicle moves again: the stop ends
                if last_index_for_compress < index - 1:
                    break
                new_stay = True
                if start + 1 != index:  # refresh the acceleration of the first stop point
                    group.loc[start, ACC_COL] = (
                        group.loc[index, V_COL] - group.loc[start, V_COL]
                    ) / (group.loc[index, DT_COL] - group.loc[start, DT_COL])
            elif index <= last_index_for_compress:
                drop_index.append(index)

        return drop_index

    @staticmethod
    def find_steady_pace_point(group, pred_len):
        group["road_diff"] = group[ROAD_COL].diff(1)
        last_index_for_compress = group.index[-pred_len - 1] if pred_len else group.index[-2]

        drop_index = []
        new_road = True
        pre_index = group.index[0]
        for index in group[group.road_diff == 0].index:
            if index > last_index_for_compress:
                break
            if new_road:
                pre_index = index - 1
                new_road = False

            if (abs(group.loc[index, V_COL] - group.loc[pre_index, V_COL]) < 1.5
                    and abs(group.loc[index, ACC_COL] - group.loc[pre_index, ACC_COL]) < 0.3):
                drop_index.append(index)
            else:
                if index != last_index_for_compress:
                    pre_index = index

            if group.loc[index + 1, "road_diff"]:  # last point of the current road segment
                new_road = True

        group.drop(columns=["road_diff"], inplace=True)
        return drop_index

    # ------------------------------------------------------------------ #
    def construct_road_neighbor_sets(self):
        utils.create_if_noexists(ROAD_META_DIR)

        road_df = pd.read_hdf(os.path.join(self.raw_data_dir, f"{self.city}.h5"), key="road_info")
        num_road = int(road_df[ROAD_COL].max() + 1)
        road_neighbors_id, road_neighbors_lenW, road_neighbors_num = [], [], []
        for road in tqdm(range(num_road), desc="Constructing road neighbor sets", ncols=70):
            group = road_df[road_df[ROAD_COL] == road]
            o, d = group["o"].iloc[0], group["d"].iloc[0]
            neighbor_roads = road_df[(road_df["o"] == o) | (road_df["o"] == d)
                                     | (road_df["d"] == o) | (road_df["d"] == d)]
            neighbor_roads = neighbor_roads.drop(
                neighbor_roads[neighbor_roads[ROAD_COL] == road].index
            )  # no self loop
            road_neighbors_id.append(neighbor_roads[ROAD_COL].to_numpy())
            road_neighbors_lenW.append(neighbor_roads["length"].values / neighbor_roads["length"].sum())
            road_neighbors_num.append(len(road_neighbors_id[-1]))

        road_neighbors_id = pad_batch(road_neighbors_id).astype(np.int64)
        road_neighbors_lenW = pad_batch(road_neighbors_lenW)
        road_neighbors_num = np.array(road_neighbors_num)

        # Empirical transition probabilities between adjacent roads; the last column
        # stores the self-transition (consecutive points on the same segment).
        trans_p = np.zeros((road_neighbors_id.shape[0], road_neighbors_id.shape[1] + 1))
        trajs_df = pd.read_hdf(os.path.join(TRAJ_META_DIR, f"{self.city}_train.h5"), key="trips")
        groups = trajs_df.groupby(TRAJ_ID_COL)
        for _, group in tqdm(groups, desc="Calculating road transition frequency",
                             total=len(groups), ncols=70):
            road = group[ROAD_COL].values
            start_points, end_points = road[:-1], road[1:]
            indices = np.where(start_points != end_points)[0].tolist()
            remain_indices = list(set(range(len(start_points))) - set(indices))
            np.add.at(trans_p, (start_points[remain_indices], -1), 1)

            neighbors_sets = road_neighbors_id[start_points[indices]]
            neighbors = end_points[indices]
            neighbors_indices = np.argmax(neighbors_sets == neighbors[:, np.newaxis], axis=1)
            np.add.at(trans_p, (start_points[indices], neighbors_indices), 1)

        road_trans_sum = trans_p.sum(axis=-1, keepdims=True)
        road_trans_sum[road_trans_sum == 0] = 1
        trans_p = trans_p / road_trans_sum
        road_neighbors_weight = np.stack([road_neighbors_lenW, trans_p[:, :-1]], axis=-1)

        np.savez(os.path.join(ROAD_META_DIR, f"{self.city}_road_neighbors_info"),
                 road_neighbors_sets=road_neighbors_id,
                 road_neighbors_num=road_neighbors_num,
                 road_neighbors_weight=road_neighbors_weight)

    # ------------------------------------------------------------------ #
    def Testset_SimTraj_Label(self, same_OD_thres=50, neighbor_area_radius=500, **kwargs):
        fastdtw = _require("fastdtw").fastdtw

        filename = f"{self.city}_test_SimTrips-{same_OD_thres}.h5"
        print(f"Need file: {filename}")
        if os.path.exists(os.path.join(SEARCH_META_DIR, filename)):
            print("File exists.")
            return
        utils.create_if_noexists(SEARCH_META_DIR)

        test_trajs_df = pd.read_hdf(os.path.join(TRAJ_META_DIR, f"{self.city}_test.h5"), key="trips")
        trip_ori, trip_dest, trajs = [], [], []
        groups = test_trajs_df.groupby(TRAJ_ID_COL)
        for trip_id, group in tqdm(groups, desc="Gathering OD of trips", total=len(groups), ncols=70):
            traj = group[FEATURE_COLS].to_numpy()
            trajs.append(traj)
            trip_ori.append(traj[0, :2])
            trip_dest.append(traj[-1, :2])
        trajs = np.array(trajs, dtype=object)
        trip_ori, trip_dest = np.array(trip_ori), np.array(trip_dest)

        # Origin/destination distances are computed one query row at a time:
        # the full N x N matrices need tens of GB for a city-scale test set.
        def od_rows(index):
            ori_d = utils.cal_geo_distance(trip_ori, trip_ori[index][np.newaxis, :])
            dest_d = utils.cal_geo_distance(trip_dest, trip_dest[index][np.newaxis, :])
            return ((ori_d <= same_OD_thres) & (dest_d <= same_OD_thres),
                    (ori_d <= neighbor_area_radius) & (dest_d <= neighbor_area_radius))

        similar_trips = {
            "query_trip_id": np.array(list(groups.groups.keys())),
            "query_trip_index": np.arange(len(groups)),
            "same_OD_trips_ids": [], "same_OD_trips_indices": [],
            "target_trip_id": [], "target_trip_index": [],
            "non_neg_ids": [], "non_neg_indices": [],
        }

        for index, trip_id in enumerate(tqdm(similar_trips["query_trip_id"],
                                             desc="Searching similar trips", ncols=70)):
            same_OD_row, near_trips_label = od_rows(index)
            same_OD_trips_indices = np.nonzero(same_OD_row)[0]
            same_OD_trips_ids = similar_trips["query_trip_id"][same_OD_trips_indices]
            similar_trips["same_OD_trips_ids"].append(same_OD_trips_ids)
            similar_trips["same_OD_trips_indices"].append(same_OD_trips_indices)

            similar_trips["non_neg_ids"].append(similar_trips["query_trip_id"][near_trips_label])
            similar_trips["non_neg_indices"].append(similar_trips["query_trip_index"][near_trips_label])

            if len(same_OD_trips_ids) == 1:
                similar_trips["target_trip_id"].append(trip_id)
                similar_trips["target_trip_index"].append(index)
                continue

            trip = trajs[index]
            dist_list, road_diff_list = [], []
            for same_OD_trip_index in same_OD_trips_indices:
                if same_OD_trip_index == index:
                    distance, _ = fastdtw(trip[::2, :2], trip[1::2, :2], dist=utils.cal_geo_distance)
                    road_diff = len(set(trip[::2, 4]) ^ set(trip[1::2, 4]))
                else:
                    candidate_trip = trajs[same_OD_trip_index]
                    distance, _ = fastdtw(trip[:, :2], candidate_trip[:, :2], dist=utils.cal_geo_distance)
                    road_diff = len(set(trip[:, 4]) ^ set(candidate_trip[:, 4]))
                dist_list.append(distance)
                road_diff_list.append(road_diff)

            dist_list = np.array(dist_list)
            road_diff_list = np.array(road_diff_list, dtype=np.int64)
            # Guard both normalisations: equal values would otherwise divide by zero (NaN scores).
            dist_spread = dist_list.max() - dist_list.min()
            norm_dist = (dist_list - dist_list.min()) / (dist_spread if dist_spread else 1)
            spread = road_diff_list.max() - road_diff_list.min()
            norm_road_diff = (road_diff_list - road_diff_list.min()) / (spread if spread else 1)
            score = np.sum([norm_dist, norm_road_diff], axis=0).tolist()
            best = score.index(min(score))
            similar_trips["target_trip_id"].append(same_OD_trips_ids[best])
            similar_trips["target_trip_index"].append(same_OD_trips_indices[best])

        pd.DataFrame(similar_trips).to_hdf(os.path.join(SEARCH_META_DIR, filename), key="trips")

    def construct_STS_meta(self, test_traj_df_name, num_target=1000, num_negative=5000,
                           same_OD_thres=50, gen_indices=True, only_indices=False,
                           neg_seed=10, **kwargs):
        similar_trips_df = pd.read_hdf(
            os.path.join(SEARCH_META_DIR, f"{self.city}_test_SimTrips-{same_OD_thres}.h5"), key="trips"
        )
        test_trajs_df = pd.read_hdf(os.path.join(TRAJ_META_DIR, f"{self.city}_test.h5"), key="trips")
        assert list(test_trajs_df.groupby(TRAJ_ID_COL).groups.keys()) == similar_trips_df["query_trip_id"].tolist()
        test_trajs_df[T_COL] = test_trajs_df[T_COL] - self.base_timestamp

        compressed_test_trajs_df = None
        if "compressed" in test_traj_df_name:
            compressed_test_trajs_df = pd.read_hdf(
                os.path.join(TRAJ_META_DIR, test_traj_df_name + ".h5"), key="trips"
            )
            assert list(compressed_test_trajs_df.groupby(TRAJ_ID_COL).groups.keys()) == \
                similar_trips_df["query_trip_id"].tolist()
            compressed_test_trajs_df[T_COL] = compressed_test_trajs_df[T_COL] - self.base_timestamp

        meta_dir = os.path.join(SEARCH_META_DIR, test_traj_df_name)
        utils.create_if_noexists(meta_dir)

        all_meta_path = os.path.join(meta_dir, f"all_meta-{same_OD_thres}.npz")
        if os.path.exists(all_meta_path):
            all_meta = np.load(all_meta_path, allow_pickle=True)
            all_targets, all_queries = all_meta["targets"], all_meta["queries"]
            similar_trip_not_self_indices = all_meta["similar_trip_not_self_indices"].tolist()
        else:
            all_queries, all_targets, similar_trip_not_self_indices = [], [], []
            groups = test_trajs_df.groupby(TRAJ_ID_COL)
            for index, (trip_id, group) in enumerate(tqdm(groups, desc="Gathering similar trips",
                                                          total=len(groups), ncols=70)):
                similar_trip_id = similar_trips_df.loc[index, "target_trip_id"]
                if similar_trip_id == trip_id:
                    group = group.reset_index(drop=True)
                    query, target = group[::2].reset_index(drop=True), group[1::2].reset_index(drop=True)
                    query, target = self.cal_high_order_features(query), self.cal_high_order_features(target)
                    if compressed_test_trajs_df is not None:
                        query = self.gen_compressed_traj(query, 0)
                        target = self.gen_compressed_traj(target, 0)
                    all_queries.append(query[FEATURE_COLS].to_numpy())
                    all_targets.append(target[FEATURE_COLS].to_numpy())
                else:
                    similar_trip_not_self_indices.append(index)
                    source = compressed_test_trajs_df if compressed_test_trajs_df is not None else test_trajs_df
                    traj = source[source[TRAJ_ID_COL] == trip_id] if compressed_test_trajs_df is not None else group
                    tgt_traj = source[source[TRAJ_ID_COL] == similar_trip_id]
                    all_queries.append(traj[FEATURE_COLS].to_numpy())
                    all_targets.append(tgt_traj[FEATURE_COLS].to_numpy())

            all_targets = np.array(all_targets, dtype=object)
            all_queries = np.array(all_queries, dtype=object)
            np.savez(all_meta_path, targets=all_targets, queries=all_queries,
                     similar_trip_not_self_indices=np.array(similar_trip_not_self_indices, dtype=np.int64))
        print("Total number of trips whose similar trip is not itself:",
              len(similar_trip_not_self_indices))

        num_target = min(len(similar_trips_df) - 1, num_target)
        num_negative = min(len(similar_trips_df) - num_target, num_negative)
        # 1:3 mix of "similar trip is another trajectory" vs "similar trip is a split of itself".
        sampled_trip_indices = []
        random.seed(10)
        sampled_trip_indices += (random.sample(similar_trip_not_self_indices, num_target // 4)
                                 if len(similar_trip_not_self_indices) >= num_target // 4
                                 else similar_trip_not_self_indices)
        remain_trip_indices = list(np.delete(np.arange(len(similar_trips_df)), similar_trip_not_self_indices))
        random.seed(10)
        sampled_trip_indices += random.sample(remain_trip_indices, num_target - len(sampled_trip_indices))

        qrytgt = np.concatenate((all_queries[sampled_trip_indices], all_targets[sampled_trip_indices]),
                                axis=0, dtype=object)

        # Seeded so that regenerating the benchmark gives the same negatives
        # (the global NumPy RNG used before was never seeded here).
        neg_rng = np.random.RandomState(neg_seed)
        neg_indices_list = []
        for sampled_trip_index in sampled_trip_indices:
            non_neg_indices = similar_trips_df.loc[sampled_trip_index, "non_neg_indices"]
            neg_indices = np.delete(np.arange(len(similar_trips_df)), non_neg_indices)
            if len(neg_indices) < num_negative:
                raise ValueError(
                    f"Query trip {sampled_trip_index} has only {len(neg_indices)} candidate negatives "
                    f"(trips outside its neighbourhood) but num_negative={num_negative}. "
                    "Lower num_negative or the neighbourhood radius.")
            neg_indices_list.append(neg_rng.choice(neg_indices, num_negative, replace=False))
        neg_indices_list = np.array(neg_indices_list)
        print("neg_indices shape:", neg_indices_list.shape)

        if gen_indices:
            np.savez(os.path.join(SEARCH_META_DIR,
                                  f"{self.city}_test_sim_indices-{num_target}-{num_negative}-{same_OD_thres}"),
                     qry_idx=np.array(sampled_trip_indices),
                     tgt_idx=np.array(similar_trips_df.loc[sampled_trip_indices, "target_trip_index"]),
                     neg_idx=neg_indices_list)
            if only_indices:
                return

        np.savez(os.path.join(meta_dir, f"qrytgt_negidx-{num_target}-{num_negative}-{same_OD_thres}"),
                 qrytgt=qrytgt, neg_indices=neg_indices_list)
        print("Saved meta to", meta_dir)

    # ------------------------------------------------------------------ #
    def compress_traj_with_rdp(self, dataset_mode="train", pred_len=0, epsilon=3e-6):
        rdp = _require("rdp").rdp
        data_filename = f"{self.city}_{dataset_mode}"
        trajs_df = pd.read_hdf(os.path.join(TRAJ_META_DIR, data_filename + ".h5"), key="trips")

        compressed_trajs = []
        groups = trajs_df.groupby(TRAJ_ID_COL)
        for _, group in tqdm(groups, desc="Compressing trips by RDP", total=len(groups), ncols=70):
            gps = group[[X_COL, Y_COL]]
            traj_mask = rdp(gps.iloc[:-pred_len] if pred_len else gps,
                            epsilon=epsilon, algo="iter", return_mask=True)
            traj_mask[0], traj_mask[-1] = True, True
            traj_mask = list(traj_mask) + [True] * pred_len
            compressed_trajs.append(group[traj_mask])

        pd.concat(compressed_trajs).to_hdf(
            os.path.join(TRAJ_META_DIR, f"{data_filename}_RDPcompressed_keep-{pred_len}.h5"), key="trips"
        )

    def downsample_traj(self, dataset_mode="train", pred_len=0, keep_ratio=0.6):
        data_filename = f"{self.city}_{dataset_mode}"
        trajs_df = pd.read_hdf(os.path.join(TRAJ_META_DIR, data_filename + ".h5"), key="trips")

        compressed_trajs = []
        groups = trajs_df.groupby(TRAJ_ID_COL)
        for _, group in tqdm(groups, desc=f"Down-sampling trips @ {keep_ratio}",
                             total=len(groups), ncols=70):
            n = len(group)
            head = n - pred_len
            keep = np.zeros(n, dtype=bool)
            n_keep = max(2, int(round(head * keep_ratio)))
            keep[np.unique(np.linspace(0, head - 1, n_keep).round().astype(int))] = True
            keep[0], keep[head - 1] = True, True
            keep[head:] = True
            compressed_trajs.append(group[keep])

        pd.concat(compressed_trajs).to_hdf(
            os.path.join(TRAJ_META_DIR,
                         f"{data_filename}_DScompressed-{keep_ratio}_keep-{pred_len}.h5"), key="trips"
        )


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class TrajectoryDataset(Dataset):

    def __init__(self, traj_df):
        super().__init__()
        self.traj_df = traj_df
        self.traj_ids = self.traj_df[TRAJ_ID_COL].unique()

        self.base_timestamp = get_last_Monday(BASE_TIME)
        self.traj_df[T_COL] = self.traj_df[T_COL] - self.base_timestamp
        self.valid_trajs = self.traj_df.groupby(TRAJ_ID_COL)

        # Feature borders; only those of the *training* split are actually used
        # to normalise, and they are passed to the models in main.py.
        spatial_border = self.traj_df[[X_COL, Y_COL]]
        self.spatial_border = [spatial_border.min().tolist(), spatial_border.max().tolist()]
        temporal_border = self.traj_df[[T_COL, DT_COL]]
        self.temporal_border = [temporal_border.min().tolist(), temporal_border.max().tolist()]
        high_order_feature_border = self.traj_df[[V_COL, ACC_COL, ANGLE_COL]]
        self.high_order_feature_border = [high_order_feature_border.min().tolist(),
                                          high_order_feature_border.max().tolist()]

    def use_partial_data(self, prop=0.2):
        assert 0.0 < prop < 1.0
        sampled = np.sort(np.random.choice(len(self.traj_ids), size=int(prop * len(self.traj_ids)),
                                           replace=False))
        self.traj_ids = self.traj_ids[sampled]

    def reset_data(self):
        self.traj_ids = self.traj_df[TRAJ_ID_COL].unique()

    def __len__(self):
        return self.traj_ids.shape[0]

    def __getitem__(self, index):
        return self.valid_trajs.get_group(self.traj_ids[index])


class PretrainDatasetsUnion(Dataset):

    def __init__(self, teacher_embeds, compressed_dataset: TrajectoryDataset):
        super().__init__()
        assert len(compressed_dataset) == len(teacher_embeds)
        self.teacher_embeds = teacher_embeds
        self.compressed_dataset = compressed_dataset

    def __len__(self):
        return self.compressed_dataset.traj_ids.shape[0]

    def __getitem__(self, index):
        return self.compressed_dataset[index], self.teacher_embeds[index]


class TrajectorySearchDataset(Dataset):

    def __init__(self, trajs):
        super().__init__()
        self.trajs = trajs
        self.valid_trajs = self.trajs

    def compress_trajs_with_rdp(self, epsilon=3e-6):
        rdp = _require("rdp").rdp
        compressed_trajs = []
        for one_traj in tqdm(self.trajs, desc="Compressing trips by RDP",
                             total=len(self.trajs), ncols=70):
            traj_mask = rdp(one_traj[:, :2], epsilon=epsilon, algo="iter", return_mask=True)
            traj_mask[0], traj_mask[-1] = True, True
            compressed_trajs.append(one_traj[traj_mask])
        self.valid_trajs = np.array(compressed_trajs, dtype=object)

    def __len__(self):
        return self.valid_trajs.shape[0]

    def __getitem__(self, index):
        return self.valid_trajs[index]


# --------------------------------------------------------------------------- #
# Collate functions
# --------------------------------------------------------------------------- #
class PretrainPadder:

    def __init__(self, device):
        self.device = device

    def __call__(self, raw_batch):
        traj_batch, valid_lens = [], []
        for row in raw_batch:
            traj = row[FEATURE_COLS].to_numpy()
            traj_batch.append(traj)
            valid_lens.append(traj.shape[0])
        traj_batch = torch.from_numpy(pad_batch(traj_batch)).float().to(self.device)
        valid_lens = torch.tensor(valid_lens).long().to(self.device)
        return traj_batch, valid_lens


class DpPadder:

    def __init__(self, device, pred_len, pred_cols):
        self.device = device
        self.pred_len = pred_len
        self.pred_cols = pred_cols

    def __call__(self, raw_batch):
        traj_batch, valid_lens, label_batch = [], [], []
        for row in raw_batch:
            traj = row[FEATURE_COLS].to_numpy()[: -self.pred_len]
            if traj.shape[0] == 0:
                raise ValueError(
                    f"Trip {row[TRAJ_ID_COL].iloc[0]} has {len(row)} points, so nothing is left "
                    f"after hiding the last {self.pred_len}. Re-run preprocessing (MIN_TRIP_LEN).")
            traj_batch.append(traj)
            valid_lens.append(traj.shape[0])
            label_batch.append(row.iloc[-1][self.pred_cols].to_numpy())
        traj_batch = torch.from_numpy(pad_batch(traj_batch)).float().to(self.device)
        valid_lens = torch.tensor(valid_lens).long().to(self.device)
        label_batch = torch.from_numpy(np.stack(label_batch, 0).astype(float)).float().to(self.device)
        return traj_batch, valid_lens, label_batch


class UnionPretrainPadder:

    def __init__(self, device):
        self.device = device

    def __call__(self, raw_batch):
        compressed_traj_batch, compressed_valid_lens, teacher_embeds_batch = [], [], []
        for compressed_traj, teacher_traj_h in raw_batch:
            teacher_embeds_batch.append(teacher_traj_h)
            compressed_traj = compressed_traj[FEATURE_COLS].to_numpy()
            compressed_traj_batch.append(compressed_traj)
            compressed_valid_lens.append(compressed_traj.shape[0])

        compressed_traj_batch = torch.from_numpy(pad_batch(compressed_traj_batch)).float().to(self.device)
        compressed_valid_lens = torch.tensor(compressed_valid_lens).long().to(self.device)
        teacher_embeds_batch = torch.from_numpy(np.array(teacher_embeds_batch)).float().to(self.device)
        return (compressed_traj_batch, compressed_valid_lens), teacher_embeds_batch


class SearchPadder:

    def __init__(self, device):
        self.device = device

    def __call__(self, raw_batch):
        traj_batch, valid_lens = [], []
        for traj in raw_batch:
            traj_batch.append(traj)
            valid_lens.append(traj.shape[0])
        traj_batch = torch.from_numpy(pad_batch(traj_batch)).float().to(self.device)
        valid_lens = torch.tensor(valid_lens).long().to(self.device)
        return traj_batch, valid_lens


def fetch_task_padder(padder_name, device, padder_params):
    if padder_name == "dp":
        return DpPadder(device, **padder_params)
    if padder_name == "pretrain":
        return PretrainPadder(device)
    if padder_name == "unionpretrain":
        return UnionPretrainPadder(device)
    if padder_name == "search":
        return SearchPadder(device)
    raise NotImplementedError(f"No padder named {padder_name}")


def pad_batch(batch):
    max_len = max(arr.shape[0] for arr in batch)
    if batch[0].ndim > 1:
        padded_batch = np.full((len(batch), max_len, batch[0].shape[-1]), FEATURE_PAD, dtype=float)
    else:
        padded_batch = np.full((len(batch), max_len), FEATURE_PAD, dtype=float)
    for i, arr in enumerate(batch):
        padded_batch[i, : arr.shape[0]] = arr
    return padded_batch


def get_last_Monday(start_time: pd.Timestamp):
    # NOTE: despite the name this anchors the week at the preceding *Sunday*
    # 00:00 (2018-09-23 for BASE_TIME).  Only the phase of the day-of-week
    # token depends on it; kept as-is so that results stay reproducible.
    days_to_monday = (start_time.weekday() + 1) % 7
    last_monday = start_time - pd.Timedelta(days=days_to_monday)
    return last_monday.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


# --------------------------------------------------------------------------- #
class TrajectorySearchTestdata:

    @staticmethod
    def parse_label(length):
        return list(range(length // 2)), list(range(length // 2, length))

    @staticmethod
    def cal_pres_and_labels(query, target, database, neg_indices, chunk_size=64):
        """Score each query against its own target and its own negatives.

        Returns ``(scores, labels)`` where ``scores[:, 0]`` is the (negated L2)
        query-target similarity, ``scores[:, 1:]`` the query-negative ones and
        every label is 0.  Negatives are gathered chunk by chunk from
        ``database`` so that memory stays at ``chunk_size x num_neg x d``
        instead of materialising ``num_query x num_neg x d`` arrays.
        """
        num_queries, num_targets = query.shape[0], target.shape[0]
        print("query:", query.shape, "target:", target.shape, "neg:", neg_indices.shape)
        assert num_queries == num_targets, "Number of queries and targets should be the same."

        dist_qt = np.linalg.norm(query - target, ord=2, axis=1)
        dist_qn = np.empty(neg_indices.shape, dtype=dist_qt.dtype)
        for s in range(0, num_queries, chunk_size):
            e = min(s + chunk_size, num_queries)
            dist_qn[s:e] = np.linalg.norm(query[s:e, None, :] - database[neg_indices[s:e]],
                                          ord=2, axis=2)
        dist_mat = np.concatenate([dist_qt[:, None], dist_qn], axis=1)
        return -1 * dist_mat, np.zeros(num_queries)


def load_trajSearch_testdata(search_meta_dir, num_target=1000, num_negative=5000,
                             same_OD_thres=50, **kwargs):
    alltrajtgt = np.load(os.path.join(search_meta_dir, f"all_meta-{same_OD_thres}.npz"),
                         allow_pickle=True)["targets"]
    qrytgt_negidx = np.load(
        os.path.join(search_meta_dir, f"qrytgt_negidx-{num_target}-{num_negative}-{same_OD_thres}.npz"),
        allow_pickle=True,
    )
    return alltrajtgt, qrytgt_negidx["qrytgt"], qrytgt_negidx["neg_indices"]


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser(description="DeltaPath data preprocessing")
    parser.add_argument("-s", "--settings", type=str, default="local_test_search",
                        help="name of the settings file in ./settings")
    parser.add_argument("--skip-sts", action="store_true",
                        help="skip similar-trajectory-search meta construction")
    parser.add_argument("--ablation-compression", action="store_true",
                        help="also build RDP / down-sampled corpora for the MG-Trans-* ablations")
    args = parser.parse_args()

    with open(os.path.join("settings", f"{args.settings}.json"), "r") as fp:
        settings = json.load(fp)

    for setting_i, setting in enumerate(settings):
        print(f"=== SETTING {setting_i}/{len(settings)} ===")
        preprocessor = DataPreprocessor(setting["dataset"]["city"])
        preprocessor.preprocess_traj()
        preprocessor.construct_road_neighbor_sets()

        # Pre-training corpora (whole trajectory kept).
        preprocessor.compress_traj(dataset_mode="train", pred_len=0)
        preprocessor.compress_traj(dataset_mode="test", pred_len=0)
        # Downstream corpora (last 5 points reserved as prediction targets).
        preprocessor.compress_traj(dataset_mode="train", pred_len=5)
        preprocessor.compress_traj(dataset_mode="valid", pred_len=5)
        preprocessor.compress_traj(dataset_mode="test", pred_len=5)

        if args.ablation_compression:
            # keep-5 corpora for the downstream tasks, keep-0 for pre-training.
            for mode in ("train", "valid", "test"):
                preprocessor.compress_traj_with_rdp(dataset_mode=mode, pred_len=5)
                preprocessor.downsample_traj(dataset_mode=mode, pred_len=5, keep_ratio=0.6)
            for mode in ("train", "test"):
                preprocessor.compress_traj_with_rdp(dataset_mode=mode, pred_len=0)
                preprocessor.downsample_traj(dataset_mode=mode, pred_len=0, keep_ratio=0.6)

        if "test" in setting and not args.skip_sts and "search_data_params" in setting["test"]:
            preprocessor.Testset_SimTraj_Label(**setting["test"]["search_data_params"])
            eval_dataset = os.path.basename(setting["dataset"]["test_traj_df"]).split(".")[0]
            preprocessor.construct_STS_meta(eval_dataset, **setting["test"]["search_data_params"])
