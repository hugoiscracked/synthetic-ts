"""Phase 1 test, as pre-registered (article/tsfm_match_preregistration.md).

Spearman rho between each held-out dataset's profile distance to the synthetic
corpus and its log with/without ratio (MASE primary, WQL secondary), with a
one-sided permutation p-value (200,000 permutations, seed 2026).

    python scripts/analysis/tsfm_match_test.py --profiles profiles.csv --cells-dir DIR
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

FEATURES = ("seasonality_strength", "changepoint_rate", "acf_decay", "tail_index")
SEEDS = (2021, 2022, 2023)
N_PERM = 200_000
SEED = 2026


def dataset_log_ratio(cells_dir, metric):
    """log of the seed-averaged geometric-mean with/without ratio per dataset,
    identical to the per-dataset statistic of tsfm_gift_heldout_analysis.py."""
    runs = {}
    for arm in ("with", "without"):
        for s in SEEDS:
            with open(Path(cells_dir) / f"sa2_{arm}_s{s}_51m_cells.csv") as f:
                runs[(arm, s)] = {r["config"]: r for r in csv.DictReader(f)}
    configs = sorted(set.intersection(*(set(r) for r in runs.values())))
    out = {}
    for base in sorted({runs[("with", 2021)][c]["base_dataset"].split("/")[0] for c in configs}):
        cs = [c for c in configs if runs[("with", 2021)][c]["base_dataset"].split("/")[0] == base]
        per_seed = [np.mean([np.log(float(runs[("with", s)][c][metric])
                                    / float(runs[("without", s)][c][metric])) for c in cs])
                    for s in SEEDS]
        out[base] = float(np.log(np.mean(np.exp(per_seed))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--cells-dir", required=True)
    a = ap.parse_args()
    prof = {r["profile"]: {k: float(r[k]) for k in FEATURES} for r in csv.DictReader(open(a.profiles))}
    synth = prof.pop("SYNTHETIC_CORPUS")
    names = sorted(prof)
    pool = np.array([[prof[n][k] for k in FEATURES] for n in names] + [[synth[k] for k in FEATURES]])
    centre = np.nanmedian(pool, axis=0)
    iqr = np.nanpercentile(pool, 75, axis=0) - np.nanpercentile(pool, 25, axis=0)
    scale = np.where(iqr > 1e-9, iqr, np.nanstd(pool, axis=0))
    z = (pool - centre) / scale
    dist = {n: float(np.linalg.norm(z[i] - z[-1])) for i, n in enumerate(names)}

    print(f"n = {len(names)} datasets; synthetic profile: "
          + ", ".join(f"{k}={synth[k]:.3f}" for k in FEATURES))
    rng = np.random.default_rng(SEED)
    for metric, label in (("mase", "PRIMARY"), ("wql", "secondary")):
        lr = dataset_log_ratio(a.cells_dir, metric)
        x = np.array([dist[n] for n in names])
        y = np.array([lr[n] for n in names])
        rho = spearmanr(x, y).statistic
        null = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(N_PERM)])
        p = (1 + np.sum(null >= rho)) / (1 + N_PERM)
        print(f"\n[{label}] {metric.upper()}: Spearman rho = {rho:+.3f}, one-sided permutation p = {p:.4f}")
        if metric == "mase":
            print(f"  {'dataset':32s} {'distance':>8s} {'ratio':>7s}  " + " ".join(f"{k[:10]:>10s}" for k in FEATURES))
            for n in sorted(names, key=lambda n: dist[n]):
                print(f"  {n:32s} {dist[n]:8.3f} {np.exp(lr[n]):7.4f}  "
                      + " ".join(f"{prof[n][k]:10.3f}" for k in FEATURES))


if __name__ == "__main__":
    main()
