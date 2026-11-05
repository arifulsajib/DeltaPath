#!/usr/bin/env python
"""Generate a small synthetic city so the full pipeline can be smoke-tested
without the real (licensed) trajectory datasets.

The toy city is a grid road network.  Vehicles perform random walks on it and
are sampled every few seconds, with occasional stops and several fixes per road
segment, so every preprocessing branch (stay-point removal, steady-pace
removal, similar-trajectory labelling) is exercised.

    python scripts/make_toy_data.py                # writes ./samples/toy*
    python scripts/make_toy_data.py --out samples --city toy --seed 0

Files written (same layout as the real datasets, see README):
    <out>/<city>.h5                    keys: pois, road_info
    <out>/<city>_{train,valid,test}.h5 key:  trips
    <out>/<city>_road_embed.npy        (num_road, D) road text embeddings
    <out>/<city>_poi_embed.npy         (num_poi, D)  POI text embeddings
"""

import argparse
import os

import numpy as np
import pandas as pd

ORIGIN_LNG, ORIGIN_LAT = 104.00, 30.60     # an arbitrary south-west corner
DEG_PER_M_LAT = 1 / 111_320
DEG_PER_M_LNG = 1 / (111_320 * np.cos(np.radians(ORIGIN_LAT)))


def build_grid(n, spacing_m):
    """Directed grid: every pair of 4-adjacent nodes is joined by two roads."""
    node_xy = {}
    for i in range(n):
        for j in range(n):
            node_xy[i * n + j] = (ORIGIN_LNG + j * spacing_m * DEG_PER_M_LNG,
                                  ORIGIN_LAT + i * spacing_m * DEG_PER_M_LAT)
    roads = []
    for i in range(n):
        for j in range(n):
            u = i * n + j
            for di, dj in ((0, 1), (1, 0)):
                ii, jj = i + di, j + dj
                if ii < n and jj < n:
                    v = ii * n + jj
                    roads.append((u, v))
                    roads.append((v, u))
    road_df = pd.DataFrame(roads, columns=["o", "d"])
    road_df["road"] = np.arange(len(road_df))
    road_df["length"] = float(spacing_m)
    out_edges = {}
    for r, (o, d) in enumerate(roads):
        out_edges.setdefault(o, []).append(r)
    return node_xy, road_df[["road", "o", "d", "length"]], out_edges


def random_trip(rng, trip_id, node_xy, road_df, out_edges, start_time):
    n_roads = int(rng.integers(3, 12))
    node = int(rng.choice(list(node_xy)))
    prev = None
    rows, t = [], start_time
    for _ in range(n_roads):
        choices = [r for r in out_edges[node] if road_df.at[r, "d"] != prev] or out_edges[node]
        road = int(rng.choice(choices))
        o, d = road_df.at[road, "o"], road_df.at[road, "d"]
        (x0, y0), (x1, y1) = node_xy[o], node_xy[d]
        n_fix = int(rng.integers(2, 6))
        for k in range(n_fix):
            a = k / n_fix
            x, y = x0 + a * (x1 - x0), y0 + a * (y1 - y0)
            rows.append((trip_id, t, x, y, road))
            t += pd.Timedelta(seconds=int(rng.integers(3, 15)))
            if rng.random() < 0.08:                      # a stop: repeated fixes
                for _ in range(int(rng.integers(1, 4))):
                    rows.append((trip_id, t, x, y, road))
                    t += pd.Timedelta(seconds=int(rng.integers(5, 30)))
        prev, node = o, d
    x, y = node_xy[node]
    rows.append((trip_id, t, x, y, rows[-1][-1]))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="samples")
    p.add_argument("--city", default="toy")
    p.add_argument("--grid", type=int, default=12, help="nodes per side of the grid")
    p.add_argument("--spacing", type=float, default=250.0, help="metres between nodes")
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--n-valid", type=int, default=80)
    p.add_argument("--n-test", type=int, default=160)
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    node_xy, road_df, out_edges = build_grid(args.grid, args.spacing)

    # POIs: jittered around every node, so each one has neighbours nearby.
    poi_xy = np.array(list(node_xy.values()))
    poi_xy = np.concatenate([poi_xy, poi_xy + rng.normal(0, 30 * DEG_PER_M_LAT, poi_xy.shape)])
    poi_df = pd.DataFrame({"lng": poi_xy[:, 0], "lat": poi_xy[:, 1]})

    base = pd.Timestamp("2018-10-01 00:00:00")
    trip_id = 0
    for split, n in (("train", args.n_train), ("valid", args.n_valid), ("test", args.n_test)):
        rows = []
        while n > 0:
            start = base + pd.Timedelta(seconds=int(rng.integers(0, 30 * 24 * 3600)))
            trip = random_trip(rng, trip_id, node_xy, road_df, out_edges, start)
            if 8 <= len(trip) <= 120:
                rows.extend(trip)
                trip_id += 1
                n -= 1
        df = pd.DataFrame(rows, columns=["trip", "time", "lng", "lat", "road"])
        df.to_hdf(os.path.join(args.out, f"{args.city}_{split}.h5"), key="trips")
        print(f"{split}: {df.trip.nunique()} trips, {len(df)} points")

    meta_path = os.path.join(args.out, f"{args.city}.h5")
    poi_df.to_hdf(meta_path, key="pois")
    road_df.to_hdf(meta_path, key="road_info")
    np.save(os.path.join(args.out, f"{args.city}_road_embed.npy"),
            rng.normal(size=(len(road_df), args.embed_dim)).astype(np.float32))
    np.save(os.path.join(args.out, f"{args.city}_poi_embed.npy"),
            rng.normal(size=(len(poi_df), args.embed_dim)).astype(np.float32))
    print(f"{len(road_df)} roads, {len(poi_df)} POIs -> {args.out}/")


if __name__ == "__main__":
    main()
