# Foundation-model pre-training ablation

Results behind Section 5.5 and Appendix "Pre-training Ablation" of the paper.

A Chronos-2-style channel-joint model (39.0 M parameters) is pre-trained on a
corpus built from GIFT-Eval pre-training windows in two arms that differ only in
composition: **with** synthetic data (6.42 M training windows, 31.9% synthetic,
30 shards from six generator families) and **without** (the same corpus minus
those shards). Both arms take 12,500 optimiser steps with identical
hyperparameters; three seeds (2021–2023) per arm, so six checkpoints.

| File | Contents |
|---|---|
| `ett_scores.csv` | Zero-shot MSE/MAE on ETTh1/h2/m1/m2, H ∈ {96, 192}, 800 windows per cell, per arm and seed (48 rows). ETT is excluded from both corpora. |
| `gift_cells/sa2_{with,without}_s{seed}_51m_cells.csv` | Held-out GIFT-Eval scores per configuration (MASE, WQL, and Seasonal-Naive references), 67 configurations per checkpoint. |
| `gift_heldout_configs.txt` | The 67 held-out configurations (19 datasets). |
| `preregistration_gift.md` | Dataset-selection rule, metrics and analysis for the held-out GIFT-Eval scoring, fixed before any score was computed. |
| `preregistration_match.md` | Profile-distance ("match") test, fixed before any profile was computed. |
| `match_profiles.csv` | Profiles of the 16 profiled held-out datasets and of the synthetic corpus. |
| `scripts/` | The analysis scripts named in the pre-registrations (`gift_heldout_analysis.py`, `match_test.py`), the profiling script (`match_profiles.py`, needs the GIFT-Eval data and the corpus shards) and its feature definitions (`ts_features.py`). |

The pre-registration files are verbatim copies of the versions committed in our
working repository before the corresponding results existed.

Reproduce the reported statistics:

```bash
cd results/tsfm
python scripts/gift_heldout_analysis.py --cells-dir gift_cells --configs gift_heldout_configs.txt
python scripts/match_test.py --profiles match_profiles.csv --cells-dir gift_cells
```
