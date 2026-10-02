"""Phase 1 profiles, exactly as pre-registered (article/tsfm_match_preregistration.md).

Profiles the 19 held-out GIFT-Eval datasets (training portion only) and the
synthetic part of the TSFM "with" corpus with the supervised study's four
MATCH_FEATURES, each the median over windows of length L = 256. Runs on Elja,
where the gift_eval data and the corpus shards live:

    python tsfm_match_profiles.py --configs tsfm_gift_heldout_configs.txt --out profiles.csv
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

from ts_features import MATCH_FEATURES, describe_series

L = 256
N_GIFT = 400
N_SYNTH = 2000
SEED = 2026
DATA = "/hpchome/hugot/data/melange"
ORIG_CFG = "/hpchome/hugot/melange/checkpoints/synth_abl/51m/sa_with_s2021_51m_0/run_config.json"


def median_profile(windows):
    feats = [describe_series(w) for w in windows]
    out = {}
    for k in MATCH_FEATURES:
        v = np.array([f[k] for f in feats], float)
        v = v[np.isfinite(v)]
        out[k] = float(np.median(v)) if v.size else float("nan")
    return out


def standardise(x):
    mu, sd = np.nanmean(x), np.nanstd(x)
    return (x - mu) / (sd if sd > 1e-9 else 1.0)


def gift_windows(names, rng):
    """Up to N_GIFT windows split equally across the dataset's configs."""
    from gift_eval.data import Dataset
    pools = []
    for name in names:
        ds = Dataset(name=name, term="short", to_univariate=False)
        chans = []
        for e in ds.training_dataset:
            t = np.asarray(e["target"], float)
            t = t if t.ndim == 2 else t[None, :]
            for c in t:
                if len(c) >= L:
                    chans.append(standardise(c))   # training mean/sd per channel
        if chans:
            pools.append(chans)
    if not pools:
        return []
    per = N_GIFT // len(pools)
    wins = []
    for chans in pools:
        for _ in range(per):
            c = chans[int(rng.integers(len(chans)))]
            t0 = int(rng.integers(0, len(c) - L + 1))
            wins.append(np.nan_to_num(c[t0:t0 + L], nan=0.0))
    return wins


def synth_windows(rng):
    """N_SYNTH windows, shard probability proportional to series x window cap."""
    import pandas as pd
    args = json.load(open(ORIG_CFG))["args"]
    caps = dict(kv.split(":") for kv in args["corpus_dataset_window_caps"].split(","))
    idx = pq.read_table(f"{DATA}/processed/corpus_index.parquet",
                        columns=["series_id", "dataset_id", "n_channels", "file_path",
                                 "row_offset_start", "row_count"]).to_pandas()
    idx = idx[idx["dataset_id"].str.startswith("synthetic_")]
    shards = sorted(idx["dataset_id"].unique())
    by_shard = {sh: idx[idx["dataset_id"] == sh] for sh in shards}
    weight = np.array([len(by_shard[s]) * int(caps[s]) for s in shards], float)
    weight /= weight.sum()
    picks = defaultdict(list)          # file -> [(row_offset, row_count, channel, start)]
    for s in rng.choice(len(shards), size=N_SYNTH, p=weight):
        sub = by_shard[shards[s]]
        r = sub.iloc[int(rng.integers(len(sub)))]
        ch = int(rng.integers(int(r["n_channels"])))
        t0 = int(rng.integers(0, int(r["row_count"]) - L + 1))
        path = r["file_path"] if os.path.isabs(r["file_path"]) else f"{DATA}/{r['file_path']}"
        picks[path].append((int(r["row_offset_start"]), int(r["row_count"]), ch, t0))
    # Same draws as above; reading is batched per row-group span and sliced in
    # Arrow (to_pylist over whole row groups made the first version time out).
    wins = []
    for path, items in picks.items():
        pf = pq.ParquetFile(path)
        starts = np.cumsum([0] + [pf.metadata.row_group(i).num_rows
                                  for i in range(pf.num_row_groups)])
        cache = {}
        for off, cnt, ch, t0 in items:
            g0 = int(np.searchsorted(starts, off, side="right") - 1)
            g1 = int(np.searchsorted(starts, off + cnt - 1, side="right") - 1)
            if (g0, g1) not in cache:
                cache[(g0, g1)] = pf.read_row_groups(list(range(g0, g1 + 1)), columns=["values"])
            col = cache[(g0, g1)].column("values").slice(off - int(starts[g0]), cnt).combine_chunks()
            vals = col.flatten().to_numpy(zero_copy_only=False).reshape(cnt, -1)
            series = standardise(vals[:, ch].astype(float))
            wins.append(series[t0:t0 + L])
        print(f"  read {len(items)} windows from {os.path.basename(os.path.dirname(path))}/"
              f"{os.path.basename(path)}", flush=True)
    return wins, dict(zip(shards, weight.round(4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    by_ds = defaultdict(list)
    for line in open(a.configs):
        if line.strip():
            name = line.split(",")[0].strip()
            base = name.split("/")[0]
            if name not in by_ds[base]:
                by_ds[base].append(name)
    rows = []
    for base in sorted(by_ds):
        wins = gift_windows(by_ds[base], rng)
        if not wins:
            print(f"excluded (no series >= {L}): {base}")
            continue
        rows.append({"profile": base, "n_windows": len(wins), **median_profile(wins)})
        print(rows[-1], flush=True)
    wins, weights = synth_windows(rng)
    rows.append({"profile": "SYNTHETIC_CORPUS", "n_windows": len(wins), **median_profile(wins)})
    print(rows[-1])
    print("shard weights:", weights)
    import csv
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
