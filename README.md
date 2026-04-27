# When Does Synthetic Data Help? Empirical Evidence from Deep Learning Time Series Forecasters

This repository contains the code, experiment scripts, and results accompanying the paper:

> **When Does Synthetic Data Help? Empirical Evidence from Deep Learning Time Series Forecasters**  
> Hugo Cazaux, Eyjólfur Ingi Ásgeirsson, Hlynur Stefánsson
> Preprint: [arXiv link — to be added]

---

## Overview

We run **4,218 controlled experiments** across five forecasting architectures, seven benchmark datasets, nine prediction horizons and four synthetic data variables to answer the question: _when_ does adding synthetic time-series data to training help, and when does it hurt?

**Key findings:**
- Architecture is the strongest single predictor: channel-mixing models (TimesNet, iTransformer) benefit consistently; channel-independent models (DLinear, PatchTST) are reliably harmed.
- The Seasonal-Trend (ST) bundle is the safest choice across all datasets; hard curriculum switches are catastrophic (+24% MSE vs static mixing).
- In low-resource settings (≤25% real data), a receptive architecture can match or exceed full-data baselines. A cache of 500 pre-generated samples is sufficient and essentially free at inference time.

---

## Repository structure

```
.
├── paper/
│   └── paper.pdf                  # Full paper (PDF)
├── results/
│   ├── all_results.csv            # All 4,229 runs, combined (with Group column)
│   ├── group1_baseline.csv        # Group 1: real-only baselines
│   ├── group2_sparsity.csv        # Group 2: data sparsity
│   ├── group3_augmentation.csv    # Group 3: synthetic augmentation ratio
│   ├── group4_lowresource.csv     # Group 4: low-resource regime
│   ├── group5_difficulty.csv      # Group 5: difficulty conditioning
│   ├── group6_bundle.csv          # Group 6: bundle type (ST/NR/LM/VE)
│   ├── group7_curriculum.csv      # Group 7: curriculum / annealing schedule
│   ├── group8_latent.csv          # Group 8: latent factor cross-channel correlation
│   └── group9_cacheablation.csv   # Group 9: cache size ablation
├── data_provider/
│   ├── data_loader.py             # Dataset_Synthetic, Dataset_Mixed + standard TSLib loaders
│   └── data_factory.py            # data_provider() with mixed/anneal/sparsity modes
├── exp/
│   └── exp_long_term_forecasting.py  # Training loop with per-epoch annealing support
├── models/                        # Five architectures used in the paper
│   ├── DLinear.py
│   ├── iTransformer.py
│   ├── PatchTST.py
│   ├── SegRNN.py
│   └── TimesNet.py
├── layers/                        # Required TSLib layer modules
├── utils/                         # Metrics, tools, time features
├── run.py                         # CLI entry point
├── requirements.txt
└── scripts/
    ├── config.sh                  # Dataset paths, hyperparameter defaults
    ├── run_single.sh              # Run one experiment (with resume logic)
    ├── generate_jobs.sh           # Generate all job scripts for all 9 groups
    └── slurm_array_template.sh   # SLURM array job template (adapt before use)
```

---

## Relation to TSLib

This codebase is built on top of [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library).  
All model files (`models/`, `layers/`) and base dataset loaders (`data_provider/data_loader.py` lines 1270+) come from TSLib unchanged.

**New contributions** (highlighted in `data_provider/data_loader.py`):
- `Dataset_Synthetic` — parameterised synthetic generator with four bundles (ST, NR, LM, VE), difficulty conditioning, and latent-factor cross-channel correlation.
- `Dataset_Mixed` — wraps any real TSLib dataset with synthetic samples at a configurable ratio.

**Modified files:**
- `data_provider/data_factory.py` — adds `mixed`, `anneal`, `anneali`, `synthetic` data modes and sparsity control.
- `exp/exp_long_term_forecasting.py` — adds per-epoch annealing ratio updates.
- `run.py` — adds synthetic-data CLI arguments.

---

## Setup

```bash
git clone https://github.com/hugoiscracked/synthetic-tslib
cd synthetic-tslib
pip install -r requirements.txt
```

Download the benchmark datasets from the [TSLib repository](https://github.com/thuml/Time-Series-Library) and place them under `./dataset/`:

```
dataset/
├── ETT-small/    ETTh1.csv  ETTh2.csv  ETTm1.csv  ETTm2.csv
├── weather/      weather.csv
├── electricity/  electricity.csv
└── traffic/      traffic.csv
```

---

## Quick start

**Real-only baseline (Group 1):**
```bash
python run.py \
  --task_name long_term_forecast --is_training 1 \
  --model iTransformer --model_id test \
  --data ETTh1 --root_path ./dataset/ETT-small --data_path ETTh1.csv \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --data_mode real --use_gpu True
```

**Mixed augmentation — ST bundle, gradual annealing (best config from paper):**
```bash
python run.py \
  --task_name long_term_forecast --is_training 1 \
  --model iTransformer --model_id synth_test \
  --data custom --root_path ./dataset/weather --data_path weather.csv \
  --enc_in 21 --dec_in 21 --c_out 21 --freq h \
  --seq_len 96 --label_len 48 --pred_len 336 \
  --data_mode anneal --anneal_strategy gradual \
  --synth_ratio 1.0 --synth_bundle ST --synth_difficulty uniform \
  --synth_cache_size 500 \
  --use_gpu True
```

---

## Reproducing all experiments

```bash
# 1. Generate all job scripts (run from repo root)
bash scripts/generate_jobs.sh

# 2a. Run locally (sequential, slow)
bash jobs/all_jobs.txt

# 2b. Submit via SLURM (adapt the template first)
#     Edit scripts/slurm_array_template.sh: partition, account, modules, venv path
sbatch scripts/slurm_array_template.sh
```

Resume logic is built in: re-running any job skips experiments already present with a valid MSE in the results CSV.

---

## Results

`results/all_results.csv` contains 4,229 unique successfully completed runs with columns.
The paper's controlled study plans 4,218 runs across Groups 1–9 (Table 3); the 11 additional runs in Group 9 are preliminary cache-ablation trials included here for completeness.

| Column | Description |
|---|---|
| `group` | Experiment group (1–9) |
| `experiment_group` | Group label string |
| `model` | Architecture name |
| `dataset` | Benchmark dataset |
| `pred_len` | Prediction horizon (96 or 336) |
| `seed` | Random seed |
| `data_mode` | Training data mode (`real`, `mixed`, `anneal`, …) |
| `synth_ratio` | Synthetic fraction (where applicable) |
| `sparsity` | Real data fraction (Group 2 & 4) |
| `difficulty_mode` | Dirichlet difficulty (Group 5) |
| `bundle` | Bundle type (Group 6) |
| `p_latent_factor` | Latent-factor probability (Group 8) |
| `anneal_strategy` | Curriculum strategy (Group 7) |
| `anneal_epoch` | Switch epoch (hard annealing) |
| `mse` | Test MSE (primary metric) |
| `mae` | Test MAE |
| `train_time` | Wall-clock training time (seconds) |

---

## Citation

```bibtex
@article{cazaux2026synthetic,
  title     = {When Does Synthetic Data Help? Empirical Evidence from Deep Learning Time Series Forecasters},
  author    = {Cazaux, Hugo and {\'A}sgeirsson, Eyj{\'o}lfur Ingi and Stef{\'a}nsson, Hlynur},
  journal   = {arXiv preprint},
  year      = {2026}
}
```

---

## Acknowledgements

Model implementations and base dataset loaders are from [TSLib](https://github.com/thuml/Time-Series-Library) (MIT licence). The SegRNN implementation follows [Lin et al. 2024](https://arxiv.org/abs/2308.11200).
