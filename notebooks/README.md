# REPATH research notebooks

The notebooks reproduce the feature-generation, model-training, evaluation, ablation, and confidence-calibration workflow used for the bundled REPATH artifacts.

Install the notebook dependencies from the repository root:

```bash
python -m pip install -e '.[notebooks]'
```

Run the notebooks in order:

1. `REPATH_01_Generate_Features.ipynb`
2. `REPATH_02_Train_Evaluate_Ablate.ipynb`
3. `REPATH_03_Calibrate_Confidence_Margin.ipynb`

## Data flow

```text
data/
  UITestFix site folders
        |
        v
generated/repath/features/
  feature matrices, schema, caches, and retrieval diagnostics
        |
        +--------------------------+
        |                          |
        v                          v
models/                       artifacts/repath/
  lgb_ranker.txt                repath_query_manifest.csv
  config_base.pkl               development_oof_candidate_scores.csv
  config.pkl                    test_candidate_scores.csv
        |
        v
results/notebooks/ and results/calibration/
  training, ablation, and calibration reports
```

The files under `generated/` and `results/` are reproducible outputs and are ignored by Git. The model files and the three calibration inputs are committed because they are required by the benchmark and calibration commands.

The site groups are read from `configs/site_split.yaml`, and the dataset is read from `data/`. The notebooks can be opened from the repository root or from the `notebooks/` directory.
