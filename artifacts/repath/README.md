# REPATH calibration inputs

This directory contains the calibration inputs used by `repath-eval calibrate`:

- `repath_query_manifest.csv`: one row per framework query, including its site and split.
- `development_oof_candidate_scores.csv`: pooled four-fold development out-of-fold candidate scores.
- `test_candidate_scores.csv`: held-out test candidate scores.

`REPATH_01_Generate_Features.ipynb` writes the query manifest. `REPATH_02_Train_Evaluate_Ablate.ipynb` writes the two candidate-score files. The calibration command selects the threshold from development OOF scores and applies the frozen threshold to the held-out test scores.

Replace all three files together when regenerating the bundled model artifacts.
