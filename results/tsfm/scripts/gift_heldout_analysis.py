"""Pre-registered analysis of the held-out GIFT-Eval scores (article/tsfm_gift_preregistration.md).

Reads the six per-config CSVs written by elja_synth_ablation_v2_gift.sh and
reports, for MASE and WQL separately, the ratio r = with / without per config
and seed (r < 1: synthetic data helps):

  primary    geometric mean of r over configs, per seed -> mean +- sd over seeds
  secondary  configs with seed-averaged r < 1 (+ exact two-sided sign test),
             configs with r < 1 on all seeds / r > 1 on all seeds,
             primary statistic per term and per dataset
  reported   each arm's geometric-mean metric normalised by Seasonal Naive

    python scripts/analysis/tsfm_gift_heldout.py --cells-dir reports/gift_heldout
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

SEEDS = (2021, 2022, 2023)
METRICS = ("mase", "wql")


def load(path):
    with open(path) as f:
        return {r["config"]: r for r in csv.DictReader(f)}


def gmean(x):
    return float(np.exp(np.mean(np.log(x))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-dir", required=True)
    ap.add_argument("--configs", default="article/tsfm_gift_heldout_configs.txt")
    a = ap.parse_args()

    expected = [line.strip() for line in open(a.configs) if line.strip()]
    runs = {(arm, s): load(Path(a.cells_dir) / f"sa2_{arm}_s{s}_51m_cells.csv")
            for arm in ("with", "without") for s in SEEDS}
    common = set.intersection(*(set(r) for r in runs.values()))
    configs = sorted(common)
    print(f"expected {len(expected)} configs; scored for all six checkpoints: {len(configs)}")
    for name, rows in runs.items():
        missing = sorted(set(configs) ^ set(rows))
        if missing:
            print(f"  dropped (missing in some run) {name}: {missing}")

    for m in METRICS:
        # ratio[config][seed]
        r = np.array([[float(runs[("with", s)][c][m]) / float(runs[("without", s)][c][m])
                       for s in SEEDS] for c in configs])
        per_seed = [gmean(r[:, i]) for i in range(len(SEEDS))]
        print(f"\n=== {m.upper()}  (ratio with/without; < 1 = synthetic helps)")
        print("primary: geometric-mean ratio per seed "
              + ", ".join(f"{x:.4f}" for x in per_seed)
              + f"  -> {np.mean(per_seed):.4f} +- {np.std(per_seed, ddof=1):.4f}"
              + f"  ({(1 - np.mean(per_seed)) * 100:+.1f}%)")
        avg = np.exp(np.log(r).mean(axis=1))
        k = int((avg < 1).sum())
        print(f"configs improved (seed-averaged): {k}/{len(configs)}, "
              f"sign test p = {binomtest(k, len(configs)).pvalue:.3g}")
        print(f"improved on all seeds: {int((r < 1).all(1).sum())}; "
              f"worse on all seeds: {int((r > 1).all(1).sum())}")
        for term in ("short", "medium", "long"):
            idx = [i for i, c in enumerate(configs) if runs[("with", 2021)][c]["term"] == term]
            if idx:
                ps = [gmean(r[idx, i]) for i in range(len(SEEDS))]
                print(f"  term {term:6s} n={len(idx):2d}: {np.mean(ps):.4f} +- {np.std(ps, ddof=1):.4f}")
        bases = sorted({runs[("with", 2021)][c]["base_dataset"].split("/")[0] for c in configs})
        print("  per dataset (n configs, gmean ratio averaged over seeds):")
        for b in bases:
            idx = [i for i, c in enumerate(configs)
                   if runs[("with", 2021)][c]["base_dataset"].split("/")[0] == b]
            ps = [gmean(r[idx, i]) for i in range(len(SEEDS))]
            print(f"    {b:32s} n={len(idx):2d}  {np.mean(ps):.4f}")
        for arm in ("with", "without"):
            vals = [gmean([float(runs[(arm, s)][c][m]) / float(runs[(arm, s)][c][f"snaive_{m}"])
                           for c in configs]) for s in SEEDS]
            print(f"  {arm:7s} arm, {m} / Seasonal Naive (gmean): "
                  f"{np.mean(vals):.4f} +- {np.std(vals, ddof=1):.4f}")


if __name__ == "__main__":
    main()
