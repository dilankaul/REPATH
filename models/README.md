# REPATH model artifacts

This directory contains the runtime artifacts produced by the research notebooks:

- `lgb_ranker.txt`: final LightGBM LambdaRank model trained on all development sites.
- `config_base.pkl`: ordered feature schema, semantic-model name, training settings, site-grouped cross-validation metadata, and ranking metrics.
- `config.pkl`: calibrated runtime configuration containing the selected confidence margin `tau = 0.40` and the development/test calibration metrics.

`REPATH_02_Train_Evaluate_Ablate.ipynb` writes `lgb_ranker.txt` and `config_base.pkl`. `REPATH_03_Calibrate_Confidence_Margin.ipynb` writes `config.pkl`.

The model and base configuration must be replaced together. The calibrated configuration can also be regenerated with:

```bash
repath-eval calibrate \
  --base-config models/config_base.pkl \
  --config-output models/config.pkl
```
