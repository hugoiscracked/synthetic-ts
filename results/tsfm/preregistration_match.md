# Pre-registration: does generator-target match explain the held-out GIFT effect? (Phase 1)

Committed **before any profile of a GIFT dataset or of the synthetic corpus was
computed.** The per-dataset GIFT outcomes are already known (see
`tsfm_gift_preregistration.md`); this test is fixed without reference to them
beyond using the outcome definition registered there.

## Hypothesis
The supervised study shows that a bundle helps to the extent that the structure
it adds matches the target. The pre-training analogue: **a held-out dataset
benefits less from adding the synthetic corpus the farther its statistical
profile is from that corpus.**

Prediction: Spearman rho(distance, log r) > 0, where r = with/without per-dataset
geometric-mean MASE ratio (larger distance -> worse relative outcome).

## Profiles (same definitions as the supervised study, `bundle_matching.py`)
- Features: the four `MATCH_FEATURES` of `scripts/analysis/ts_features.py`
  (seasonality strength, change-point rate, ACF decay, Hill tail index), each
  the **median over windows**.
- **Window length L = 256** for every profile (ACF evaluated to lag 100; the
  supervised study showed window length must be common across profiles).
- **GIFT datasets:** training portion only (`gift_eval` `Dataset(term="short")
  .training_dataset`), each channel standardised with its own training mean/sd.
  400 windows per dataset, split equally across the dataset's held-out
  frequency configs; per window a uniformly random series, channel and start.
  Only series with training length >= L are eligible; a config with none
  contributes no windows; a dataset with none is excluded.
  Consequence, known from lengths alone: `car_parts_with_missing` (27),
  `hospital` (60), `covid_deaths` (152) are excluded -> **n = 16 datasets**.
- **Synthetic corpus:** the 30 `synthetic_*` shards of the "with" arm, 2,000
  windows, shard chosen with probability proportional to (series x window cap)
  -- the share each shard has among the arm's training windows -- then a
  uniformly random series, channel and start; per-channel standardisation over
  the full series.
- Seed 2026 for all sampling.

## Distance and test
- Robust scaling pooled over the 16 dataset profiles and the synthetic profile:
  per feature, centre = median, scale = IQR (as `robust_scales`).
- Distance = Euclidean norm in the scaled 4-D space, dataset to synthetic profile.
- **Primary:** Spearman rho between distance and log r(MASE) over the 16
  datasets; **one-sided permutation p-value** (200,000 permutations, seed 2026),
  alpha = 0.05.
- **Secondary:** the same with r(WQL).
- **Descriptive only (no test):** the 16 profiles, the synthetic profile, and
  each dataset's distance.

## Interpretation, fixed in advance
- rho > 0 with p < 0.05: consistent with the match mechanism at corpus level.
- Otherwise: no support at corpus level from this correlational test (Phase 2,
  the single-family pre-training arms, is the interventional test).
- Limitations stated regardless of outcome: correlational; n = 16; distance to
  the synthetic corpus may be confounded with other dataset properties
  (e.g. distance to the real corpus, series length), which this test does not
  control.
