# Pre-registration: held-out GIFT-Eval scoring of the TSFM ablation checkpoints

Written and committed **before any GIFT-Eval score was computed** for these
checkpoints. The commit timestamp is the record. Nothing below is to be changed
after results exist; any deviation is reported as a deviation.

## Checkpoints
Six final checkpoints, `~/melange_synthabl_v2/checkpoints/synth_abl_v2/51m/`
on Elja: `sa2_{with,without}_s{2021,2022,2023}_51m_0/checkpoint.pth`
(snapshot `f3168b4d6bd7804c` + campaign config; 12,500 optimiser steps).
The ETT results for the same checkpoints are already known and are reported
unchanged, separately.

## Dataset selection rule
Held out = GIFT-Eval benchmark datasets **not in the training corpus of either
arm**. The "with" arm's corpus is the explicit 145 `corpus_dataset_ids`; the
"without" arm's is a subset of it, so held out from "with" implies held out
from both.

1. Name/alias membership screen (`scripts/leakage_audit.py`, gift_eval):
   6/28 in corpus (all six M4 configs, via `m4`), 22/28 not.
2. Manual source check of look-alike ids against the pinned July registry
   descriptions:
   - **`solar` excluded**: our `solar_power` is "NREL Solar Power Data for
     Integration Studies"; GIFT-Eval `solar` derives from the same NREL data.
   - Kept after checking, distinct sources: `jena_weather` (our `weather` is
     Australian daily stations), `sz_taxi` (ours: NYC taxi), `loop_seattle`
     (ours: `los_loop`, Los Angeles), `electricity` (ours: Australian demand
     series), `kdd_cup_2018` (ours: `kdd2022`, wind power).
3. **`ett1`, `ett2` excluded**: already reported under the ETT protocol;
   excluding them keeps this evaluation independent of that table.

Result: **19 datasets, 67 canonical configs** (37 short, 15 medium, 15 long),
taken from the 97-config canonical board (`reports/gift/clepsydra_v12_51m_cells.csv`
config set). List: `tsfm_gift_heldout_configs.txt` (committed alongside).

## Protocol
- Runner: `scripts/gift_eval_run.py --run` from the frozen snapshot, standard
  GIFT-Eval metrics (MASE, WQL/CRPS) via the runner's default `fast` path,
  `--seq-len 2048`.
- **`MELANGE_USE_ARCSINH=1`**: these checkpoints were trained with arcsinh on
  (the runner's default of 0 is for prod checkpoints and would mis-score them).
- Same settings for all six checkpoints.

## Analysis (fixed in advance)
Per config and seed, the ratio r = metric(with) / metric(without), for MASE and
for WQL separately (r < 1 means synthetic data helps).

- **Primary:** geometric mean of r over the 67 configs, per seed; reported as
  mean ± sd over the three seeds, for MASE and WQL.
- **Secondary:**
  - number of configs with r < 1 on the seed-averaged (geometric-mean) ratio,
    with an exact two-sided sign test over the 67 configs;
  - number of configs with r < 1 on all three seeds, and with r > 1 on all three;
  - the same primary statistic per term (short / medium / long) and per dataset
    (all 19 reported).
- Also reported, not primary: each arm's geometric-mean MASE/WQL normalised by
  Seasonal Naive (the leaderboard statistic).
- A config that fails to run for any checkpoint is dropped for all six and
  listed; the count of dropped configs is reported.
- **All results are reported**, whichever way they point, next to the unchanged
  ETT table. Nothing is added to or removed from the dataset list after scoring.
