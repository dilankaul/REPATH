# REPATH

This repository provides a reproducible evaluation suite for REPATH and five Web-UI element-matching baselines. It combines baseline benchmarking, cold and page-pair amortized runtime measurement, REPATH ranking evaluation, exhaustive feature-group ablation reporting, feature-importance reporting, and confidence-margin calibration.

## Publication

**REPATH: Confidence-Calibrated Learning-to-Rank for Self-Healing Web UI Test Locators**

Accepted for presentation at the 2026 International Conference on Advancements in Computing (ICAC 2026).

## Capabilities

- Benchmark WATER, VISTA, SFTM, WebEvo, Similo, and REPATH under one fixed Top-1 protocol.
- Calculate bootstrap confidence intervals and exact McNemar comparisons with Holm correction.
- Select `test` (default), `development`, `all`, one site, or multiple explicit sites.
- Run accuracy-only benchmarks by default, or select cold end-to-end or page-pair-amortized runtime benchmarking with `--runtime-mode cold` or `--runtime-mode amortized`.
- Evaluate a user-selected REPATH confidence margin with `--tau`.
- Select a margin automatically using pooled development out-of-fold precision and coverage constraints.
- Reproduce feature generation, training, exhaustive ablation, and calibration with the bundled Jupyter notebooks.
- Display the bundled REPATH ranking, calibration, feature-vector, feature-importance, and ablation results directly in this README.

## Installation

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
```

```bash
repath-eval --help
```

`repath-benchmark` remains available as a compatibility alias. The old `run` subcommand remains an alias for `benchmark`.

## Repository layout

```text
REPATH-Web-UI-Locator-Healing/
├── artifacts/
│   ├── benchmark/
│   ├── repath/
│   └── statistics/
├── configs/
├── data/
├── generated/
├── models/
├── notebooks/
├── results/
├── scripts/
├── src/repath_evaluation/
│   ├── algorithms/
│   ├── calibration/
│   ├── data/
│   └── evaluation/
└── tests/
```

## Dataset setup

This suite uses the dataset released with **UITestFix**:

- Dataset: [Web-UI-Dataset](https://anonymous.4open.science/r/Web-UI-Dataset-9645)
- Paper: Yuanzhang Lin, Guoyao Wen, and Xiang Gao, ["Automated Fixing of Web UI Tests via Iterative Element Matching"](https://doi.org/10.1109/ASE56229.2023.00048), ASE 2023.

Download the dataset and open its `MatchingDataset` folder. Copy the folders **inside** `MatchingDataset` directly into this repository's `data/` directory.

```text
REPATH-Web-UI-Locator-Healing/
├── data/
│   ├── amazon-2017-2020/
│   │   ├── old/
│   │   ├── new/
│   │   └── result/
│   ├── beijing-2017-2019/
│   ├── instagram/
│   └── ...
```

Do not create `data/MatchingDataset/...`; each site folder must be an immediate child of `data/`.

```bash
repath-eval inspect
repath-eval doctor
```

A different dataset root is optional:

```bash
repath-eval inspect --data-root /path/to/MatchingDataset
```

## Site groups

The groups are stored in [`configs/site_split.yaml`](configs/site_split.yaml). The feature-generation workflow extracts 782 development and 318 held-out framework queries, for 1,100 total queries.

### Development sites

- `amazon-2017-2020`
- `apple-2018-2020`
- `fandom`
- `Bing`
- `facebook-2020-2022`
- `foodnetwork-2018-2019`
- `google`
- `linkedin-2019-2020`
- `music-2014-2019`
- `paypal`
- `Reddit`
- `twitter-2021-2022`
- `usps-2018-2019`
- `w3school-2016-2019`
- `wikipedia`
- `youtube`

### Held-out test sites

- `book-2016-2019`
- `beijing-2017-2019`
- `instagram`
- `movie-2014-2019`
- `xfinity-2018-2020`
- `yahoo`

The test group is the default:

```bash
repath-eval inspect
repath-eval inspect --sites test
repath-eval inspect --sites development
repath-eval inspect --sites all
repath-eval inspect --sites instagram
repath-eval inspect --sites instagram yahoo xfinity-2018-2020
```

Site and group matching is case-insensitive.

## Benchmarking

Run the complete accuracy benchmark with:

```bash
repath-eval benchmark
```

By default, this evaluates WATER, VISTA, SFTM, WebEvo, Similo, and REPATH on the held-out test sites with one repetition. Runtime measurement is omitted, and results are written to `results/benchmark`. Existing result files in that directory are overwritten.

### Benchmark arguments

| Argument         | Description                                                               | Default                                 |
| ---------------- | ------------------------------------------------------------------------- | --------------------------------------- |
| `--algorithms`   | Algorithms to evaluate                                                    | `water vista sftm webevo similo repath` |
| `--sites`        | Site group, one site, or multiple explicit sites                          | `test`                                  |
| `--repeats`      | Number of complete benchmark repetitions; must be a positive integer      | `1`                                     |
| `--runtime-mode` | Runtime protocol: `cold` or `amortized`; omit for accuracy-only execution | Not set                                 |
| `--output`       | Directory used to save benchmark result files                             | `results/benchmark`                     |

Run only the five baseline algorithms:

```bash
repath-eval benchmark \
  --algorithms water vista sftm webevo similo \
  --output results/baselines_test
```

Run a selected subset of algorithms:

```bash
repath-eval benchmark \
  --algorithms water similo repath \
  --output results/selected_algorithms
```

Run one site:

```bash
repath-eval benchmark \
  --sites instagram \
  --output results/instagram
```

Run multiple explicit sites:

```bash
repath-eval benchmark \
  --sites instagram yahoo xfinity-2018-2020 \
  --output results/selected_sites
```

Repeat the complete benchmark five times:

```bash
repath-eval benchmark \
  --repeats 5 \
  --output results/five_repeats
```

### Runtime protocols

The benchmark supports accuracy-only execution and two timed protocols. Both timed protocols use the same fixed eligible queries and report `runtime_mode`, `runtime_samples`, and `mean_runtime_ms` in the result summaries.

| Runtime configuration      | Execution                                                                                                                                                   | Reported value                          |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| Omitted                    | Prepares each matcher once per page pair and evaluates all queries without recording time                                                                   | Accuracy only                           |
| `--runtime-mode cold`      | Independently loads the page data, constructs and prepares the matcher, and executes the match for every query                                              | Mean cold end-to-end time per query     |
| `--runtime-mode amortized` | Loads and prepares once per page pair, executes all associated queries, and divides the complete timed page-pair duration by the number of eligible queries | Mean page-pair amortized time per query |

For more reproducible runtime measurements, restrict common numerical libraries to one CPU thread before running either timed protocol:

```bash
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

`PYTHONHASHSEED=0` makes Python hash-based ordering deterministic. The thread variables reduce CPU oversubscription and runtime variation between runs.

#### Cold end-to-end runtime

```bash
repath-eval benchmark \
  --runtime-mode cold \
  --output results/runtime_cold
```

Cold timing creates an independent execution for every query. The timer includes page-data loading, matcher construction, local model and configuration loading where applicable, page-pair preparation, matching, and cleanup. REPATH resolves or downloads SBERT before timing and clears its process model caches, so network download time is excluded while local LightGBM and SBERT loading is included in every timed query.

#### Page-pair amortized runtime

```bash
repath-eval benchmark \
  --runtime-mode amortized \
  --output results/runtime_amortized
```

Amortized timing represents a running system with application-level models already loaded. REPATH loads and retains its LightGBM and SBERT models before timed page-pair execution. For each site, the timer includes a fresh page load, page-pair preparation, all eligible query matches, and cleanup. The complete elapsed duration is divided by the number of eligible queries.

In accuracy-only and amortized execution, matchers reuse algorithm-specific page-level state across queries. WATER reuses loaded page data, SFTM reuses its tree mapping, Similo reuses candidate and neighbour metadata, VISTA reuses decoded screenshots and scene descriptors, WebEvo reuses ordered candidate metadata, and REPATH reuses prepared feature context.

For cold results, each `query_results.csv` runtime value is an independently timed query execution. For amortized results, each query row receives the page pair's mean amortized runtime so that site and cross-site weighted means remain consistent with the complete timed duration.

### Benchmark results

The accuracy results use all 318 framework queries from the six held-out test sites. Runtime values are reported as the mean ± sample standard deviation across 10 independent complete benchmark runs. Each runtime run evaluates the same 318-query test set; the standard deviation is calculated across the 10 run-level mean runtimes rather than across individual query rows.

| Algorithm | Top-1 accuracy | Cold end-to-end time (ms/query) | Page-pair amortized time (ms/query) |
| --------- | -------------: | ------------------------------: | ----------------------------------: |
| WATER     |         50.63% |                  155.89 ± 10.48 |                        45.94 ± 2.17 |
| VISTA     |         58.49% |                 3342.89 ± 82.83 |                      813.97 ± 26.61 |
| SFTM      |         81.45% |                  461.21 ± 25.50 |                         6.88 ± 0.57 |
| WebEvo    |         68.87% |                  690.67 ± 33.30 |                       182.89 ± 7.02 |
| Similo    |         78.93% |                  640.81 ± 28.33 |                      403.08 ± 11.69 |
| REPATH    |         94.97% |                 2271.29 ± 86.69 |                      633.32 ± 21.00 |

REPATH achieved the highest Top-1 accuracy. Its mean runtime was lower than VISTA under both timed protocols, although SFTM, WATER, WebEvo, and Similo were faster in the page-pair amortized setting.

#### Runtime environment

Runtime measurements were collected on a GitHub Codespaces virtual machine with the hardware and software configuration shown below.

| Component                | Configuration                                                                          |
| ------------------------ | -------------------------------------------------------------------------------------- |
| Processor                | AMD EPYC 7763, 2 vCPUs                                                                 |
| Memory                   | 7.8 GiB RAM, no swap                                                                   |
| Operating system         | Debian GNU/Linux 12 (bookworm)                                                         |
| Kernel                   | Linux 6.8.0-1052-azure, x86_64                                                         |
| Python                   | 3.11.13 (`/usr/local/bin/python`)                                                      |
| Execution device         | CPU only; CUDA unavailable                                                             |
| NumPy                    | 2.4.6                                                                                  |
| pandas                   | 2.3.3                                                                                  |
| OpenCV                   | 4.14.0                                                                                 |
| LightGBM                 | 4.7.0                                                                                  |
| PyTorch                  | 2.13.0+cu130                                                                           |
| SentenceTransformers     | 5.6.1                                                                                  |
| Reproducibility settings | `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` |

The publication-ready benchmark summaries and environment record are committed under [`artifacts/benchmark/`](artifacts/benchmark/). Raw per-run outputs are written under `results/` and excluded from version control.

## Statistical analysis

The analysis uses paired Top-1 correctness outcomes for the 318 queries from the six held-out test sites. Run the accuracy benchmark before computing the statistics:

```bash
repath-eval benchmark
repath-eval statistics \
  --input results/benchmark/query_results.csv \
  --output-dir results/statistics \
  --bootstrap-count 10000 \
  --seed 42
```

### Accuracy confidence intervals

Top-1 accuracy is reported with nominal 95% percentile bootstrap confidence intervals based on 10,000 query-level resamples.

| Algorithm | Top-1 accuracy | 95% CI |
| --------- | -------------: | -----: |
| WATER | 50.63% | [44.97%, 56.29%] |
| VISTA | 58.49% | [53.14%, 63.84%] |
| SFTM | 81.45% | [77.04%, 85.53%] |
| WebEvo | 68.87% | [63.84%, 73.90%] |
| Similo | 78.93% | [74.21%, 83.33%] |
| REPATH | 94.97% | [92.45%, 97.17%] |

### Pairwise statistical comparisons

REPATH is compared with each baseline using an exact two-sided McNemar test. Holm–Bonferroni correction is applied across the five comparisons. Accuracy improvements and paired-bootstrap confidence intervals are expressed in percentage points (pp).

| Baseline | Improvement (pp) | 95% paired-bootstrap CI (pp) | Exact McNemar p-value | Holm-adjusted p-value |
| -------- | ---------------: | ----------------------------: | --------------------: | --------------------: |
| WATER | +44.34 | [38.68, 50.00] | 5.68e-38 | 2.84e-37 |
| VISTA | +36.48 | [31.13, 41.82] | 7.16e-34 | 2.86e-33 |
| SFTM | +13.52 | [9.43, 17.61] | 2.42e-10 | 2.42e-10 |
| WebEvo | +26.10 | [21.38, 31.13] | 4.45e-24 | 1.33e-23 |
| Similo | +16.04 | [11.64, 20.44] | 1.70e-12 | 3.40e-12 |

All five Holm-adjusted p-values are below 0.001. The bootstrap uses 10,000 paired query-level resamples with replacement (NumPy `default_rng`, seed 42), with shared resampling indices across algorithms. The reported confidence intervals are nominal rather than simultaneous. Neither the query-level bootstrap nor McNemar's test accounts for within-website dependence.

The command writes `accuracy_confidence_intervals.csv` and `statistical_comparison.csv` to `results/statistics/`. The corresponding publication summaries are retained under `artifacts/statistics/`.

## Research notebooks

The repository includes the three notebooks used to produce the bundled REPATH model and calibration inputs:

1. [`REPATH_01_Generate_Features.ipynb`](notebooks/REPATH_01_Generate_Features.ipynb) generates the framework-query manifest, candidate shortlists, feature schema, and 50-feature candidate dataset.

2. [`REPATH_02_Train_Evaluate_Ablate.ipynb`](notebooks/REPATH_02_Train_Evaluate_Ablate.ipynb) performs four-fold site-grouped model selection, trains the final LambdaRank model, evaluates the held-out test sites, reports feature importance, and evaluates all 31 feature-group combinations.

3. [`REPATH_03_Calibrate_Confidence_Margin.ipynb`](notebooks/REPATH_03_Calibrate_Confidence_Margin.ipynb) selects the confidence threshold from development OOF predictions, reports the complete threshold table, and writes the calibrated runtime configuration.

Install the notebook dependencies with:

```bash
python -m pip install -e '.[notebooks]'
```

The notebooks use repository-relative paths and read the site groups from `configs/site_split.yaml`. Large feature matrices, caches, and diagnostic outputs are written under `generated/repath/` and are not committed. Training and calibration reports are written under `results/`. See [`notebooks/README.md`](notebooks/README.md) for the complete data flow.

## REPATH confidence-margin calibration

Run the default calibration:

```bash
repath-eval calibrate
```

By default, the command selects the confidence-margin threshold using pooled development out-of-fold predictions only. It targets precision `0.99`, requires at least `0.80` end-to-end coverage, evaluates tau values from `0.00` to `1.00` in steps of `0.05`, and writes the outputs to `results/calibration`.

With the bundled artifacts, the default calibration selects `tau = 0.40`.

| Argument                    | Description                                                                    | Default                                                 |
| --------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------- |
| `--manifest`                | Query manifest containing query IDs, sites, and splits                         | `artifacts/repath/repath_query_manifest.csv`            |
| `--development-scores`      | Pooled development out-of-fold candidate scores used for threshold selection   | `artifacts/repath/development_oof_candidate_scores.csv` |
| `--test-scores`             | Held-out test candidate scores used only for final evaluation                  | `artifacts/repath/test_candidate_scores.csv`            |
| `--tau`                     | Evaluates a fixed threshold instead of selecting one automatically             | Not set                                                 |
| `--target-precision`        | Minimum development precision required during automatic selection              | `0.99`                                                  |
| `--min-end-to-end-coverage` | Minimum development end-to-end coverage required during automatic selection    | `0.80`                                                  |
| `--tau-min`                 | Smallest threshold considered during automatic selection                       | `0.00`                                                  |
| `--tau-max`                 | Largest threshold considered during automatic selection                        | `1.00`                                                  |
| `--tau-step`                | Threshold-grid increment                                                       | `0.05`                                                  |
| `--output`                  | Directory used to save calibration results                                     | `results/calibration`                                   |
| `--base-config`             | Base REPATH runtime configuration used to generate a calibrated configuration  | Not set                                                 |
| `--config-output`           | Output path for the calibrated runtime configuration; requires `--base-config` | Not set                                                 |

The calibration inputs are included in the repository:

```text
artifacts/repath/
├── repath_query_manifest.csv
├── development_oof_candidate_scores.csv
└── test_candidate_scores.csv
```

Evaluate the bundled fixed operating point without reselecting it:

```bash
repath-eval calibrate \
  --tau 0.40 \
  --output results/calibration_tau_040
```

Use different automatic-selection constraints:

```bash
repath-eval calibrate \
  --target-precision 0.985 \
  --min-end-to-end-coverage 0.80 \
  --output results/calibration_custom
```

The repository also includes `models/config.pkl`, generated from `models/config_base.pkl` using the selected `tau = 0.40`. Regenerate it after replacing the model or score artifacts with:

```bash
repath-eval calibrate \
  --base-config models/config_base.pkl \
  --config-output models/config.pkl \
  --output results/calibration_auto
```

The command reports accepted and correct cases, precision, conditional and end-to-end coverage, conditional and end-to-end recall, and both F1 definitions. Threshold selection never uses held-out test results. Existing calibration result files in the selected output directory are replaced.

## REPATH research results

The bundled notebooks produce these results using the 50-feature representation, 16 development sites, six held-out test sites, four-fold site-grouped development cross-validation, and all 31 non-empty feature-group combinations.

> **Artifact alignment:** `models/lgb_ranker.txt`, `models/config_base.pkl`, the calibration score files, and the query manifest are produced by the bundled notebook workflow and align with the result tables below.

### Candidate retrieval

| Split       | Framework queries | Positive retained | Candidate recall | Mean shortlist | Pair rows |
| ----------- | ----------------: | ----------------: | ---------------: | -------------: | --------: |
| Development |               782 |               781 |           99.87% |          25.05 |     19568 |
| Test        |               318 |               314 |           98.74% |          25.01 |      7852 |

### Ranking performance

| Split                     | Framework | Retrievable | Candidate recall | Conditional Top-1 | Conditional MRR | Conditional Hit@3 | Conditional Hit@5 | End-to-end Top-1 | End-to-end MRR | End-to-end NDCG@5 |
| ------------------------- | --------: | ----------: | ---------------: | ----------------: | --------------: | ----------------: | ----------------: | ---------------: | -------------: | ----------------: |
| Development four-fold OOF |       782 |         781 |           99.87% |            98.08% |          98.77% |            99.62% |            99.74% |           97.95% |         98.64% |            98.88% |
| Held-out test             |       318 |         314 |           98.74% |            96.18% |          97.61% |            99.04% |            99.36% |           94.97% |         96.38% |            96.77% |

### Held-out test performance by site

| Test site         | Framework | Retrievable | Candidate recall | Conditional Top-1 | End-to-end Top-1 | End-to-end MRR | End-to-end NDCG@5 |
| ----------------- | --------: | ----------: | ---------------: | ----------------: | ---------------: | -------------: | ----------------: |
| beijing-2017-2019 |       103 |         103 |          100.00% |           100.00% |          100.00% |        100.00% |           100.00% |
| book-2016-2019    |       107 |         107 |          100.00% |           100.00% |          100.00% |        100.00% |           100.00% |
| yahoo             |        18 |          18 |          100.00% |            94.44% |           94.44% |         97.22% |            97.95% |
| movie-2014-2019   |        34 |          33 |           97.06% |            93.94% |           91.18% |         94.12% |            94.89% |
| instagram         |        12 |          12 |          100.00% |            83.33% |           83.33% |         90.28% |            92.76% |
| xfinity-2018-2020 |        44 |          41 |           93.18% |            82.93% |           77.27% |         82.20% |            83.39% |

### Confidence-margin operating point

The development OOF selector chose `tau = 0.40` while targeting 99% precision and at least 80% end-to-end coverage.

| Split                     |  Tau | Accepted | Correct | Precision | Conditional coverage | End-to-end coverage | Conditional recall | End-to-end recall | Conditional F1 | End-to-end F1 |
| ------------------------- | ---: | -------: | ------: | --------: | -------------------: | ------------------: | -----------------: | ----------------: | -------------: | ------------: |
| Development four-fold OOF | 0.40 |      764 |     757 |    99.08% |               97.82% |              97.70% |             96.93% |            96.80% |         97.99% |        97.93% |
| Held-out test             | 0.40 |      303 |     298 |    98.35% |               96.50% |              95.28% |             94.90% |            93.71% |         96.60% |        95.97% |

<details>

<summary><strong>Complete development OOF tau sweep</strong></summary>

|  Tau | Accepted | Correct | Precision | End-to-end coverage | End-to-end recall | End-to-end F1 |
| ---: | -------: | ------: | --------: | ------------------: | ----------------: | ------------: |
| 0.00 |      781 |     766 |    98.08% |              99.87% |            97.95% |        98.02% |
| 0.05 |      778 |     765 |    98.33% |              99.49% |            97.83% |        98.08% |
| 0.10 |      773 |     764 |    98.84% |              98.85% |            97.70% |        98.26% |
| 0.15 |      772 |     763 |    98.83% |              98.72% |            97.57% |        98.20% |
| 0.20 |      771 |     763 |    98.96% |              98.59% |            97.57% |        98.26% |
| 0.25 |      770 |     762 |    98.96% |              98.47% |            97.44% |        98.20% |
| 0.30 |      767 |     759 |    98.96% |              98.08% |            97.06% |        98.00% |
| 0.35 |      765 |     757 |    98.95% |              97.83% |            96.80% |        97.87% |
| 0.40 |      764 |     757 |    99.08% |              97.70% |            96.80% |        97.93% |
| 0.45 |      764 |     757 |    99.08% |              97.70% |            96.80% |        97.93% |
| 0.50 |      763 |     756 |    99.08% |              97.57% |            96.68% |        97.86% |
| 0.55 |      761 |     754 |    99.08% |              97.31% |            96.42% |        97.73% |
| 0.60 |      760 |     754 |    99.21% |              97.19% |            96.42% |        97.80% |
| 0.65 |      756 |     750 |    99.21% |              96.68% |            95.91% |        97.53% |
| 0.70 |      755 |     749 |    99.21% |              96.55% |            95.78% |        97.46% |
| 0.75 |      754 |     748 |    99.20% |              96.42% |            95.65% |        97.40% |
| 0.80 |      752 |     746 |    99.20% |              96.16% |            95.40% |        97.26% |
| 0.85 |      751 |     746 |    99.33% |              96.04% |            95.40% |        97.33% |
| 0.90 |      749 |     744 |    99.33% |              95.78% |            95.14% |        97.19% |
| 0.95 |      746 |     743 |    99.60% |              95.40% |            95.01% |        97.25% |
| 1.00 |      744 |     741 |    99.60% |              95.14% |            94.76% |        97.12% |

</details>

### Grouped feature importance

| Rank | Feature group | Features |      Gain | Gain share |
| ---: | ------------- | -------: | --------: | ---------: |
|    1 | Semantic      |        5 | 29573.244 |     55.79% |
|    2 | Lexical       |       12 | 18855.804 |     35.57% |
|    3 | Attribute     |       20 |  2092.700 |      3.95% |
|    4 | Structural    |        8 |  1261.093 |      2.38% |
|    5 | Geometry      |        5 |  1227.907 |      2.32% |

### Individual feature importance

The training notebook reports the Top 20 gain-ranked features shown below.

| Rank | Vector index | Feature                 | Group      |      Gain | Gain share |
| ---: | -----------: | ----------------------- | ---------- | --------: | ---------: |
|    1 |           12 | `semantic_sim`          | Semantic   | 22852.957 |     43.11% |
|    2 |           10 | `text_seqsim`           | Lexical    | 12876.604 |     24.29% |
|    3 |           16 | `context_text_sim`      | Semantic   |  3569.410 |      6.73% |
|    4 |            5 | `text_exact`            | Lexical    |  3262.501 |      6.15% |
|    5 |           14 | `sibling_text_sim`      | Semantic   |  1707.919 |      3.22% |
|    6 |           13 | `parent_text_sim`       | Semantic   |  1406.107 |      2.65% |
|    7 |           22 | `sibling_position_diff` | Structural |   988.328 |      1.86% |
|    8 |            6 | `text_substr`           | Lexical    |   971.036 |      1.83% |
|    9 |            0 | `idx_distance_log`      | Lexical    |   946.889 |      1.79% |
|   10 |           37 | `nearby_jaccard`        | Attribute  |   585.561 |      1.10% |
|   11 |           34 | `parent_seqsim`         | Attribute  |   456.066 |      0.86% |
|   12 |            4 | `text_len_ratio`        | Lexical    |   452.803 |      0.85% |
|   13 |           36 | `sibling_after_jaccard` | Attribute  |   421.175 |      0.79% |
|   14 |           46 | `geo_center_dist`       | Geometry   |   374.177 |      0.71% |
|   15 |           32 | `class_seqsim`          | Attribute  |   356.448 |      0.67% |
|   16 |           47 | `geo_size_ratio`        | Geometry   |   284.859 |      0.54% |
|   17 |           48 | `bbox_dx`               | Geometry   |   271.531 |      0.51% |
|   18 |           45 | `geo_iou`               | Geometry   |   202.548 |      0.38% |
|   19 |           18 | `dom_depth_norm_diff`   | Structural |   134.391 |      0.25% |
|   20 |           33 | `ancestor_jaccard`      | Attribute  |   132.591 |      0.25% |

<details>

<summary><strong>Complete ordered 50-feature vector</strong></summary>

| Vector index | Feature                    | Group      | Type      | Semantic model   |
| -----------: | -------------------------- | ---------- | --------- | ---------------- |
|            0 | `idx_distance_log`         | Lexical    | `float32` |                  |
|            1 | `old_text_len`             | Lexical    | `float32` |                  |
|            2 | `new_text_len`             | Lexical    | `float32` |                  |
|            3 | `len_diff`                 | Lexical    | `float32` |                  |
|            4 | `text_len_ratio`           | Lexical    | `float32` |                  |
|            5 | `text_exact`               | Lexical    | `float32` |                  |
|            6 | `text_substr`              | Lexical    | `float32` |                  |
|            7 | `old_text_empty`           | Lexical    | `float32` |                  |
|            8 | `new_text_empty`           | Lexical    | `float32` |                  |
|            9 | `text_token_overlap`       | Lexical    | `float32` |                  |
|           10 | `text_seqsim`              | Lexical    | `float32` |                  |
|           11 | `n_candidates`             | Lexical    | `float32` |                  |
|           12 | `semantic_sim`             | Semantic   | `float32` | all-MiniLM-L6-v2 |
|           13 | `parent_text_sim`          | Semantic   | `float32` | all-MiniLM-L6-v2 |
|           14 | `sibling_text_sim`         | Semantic   | `float32` | all-MiniLM-L6-v2 |
|           15 | `child_text_sim`           | Semantic   | `float32` | all-MiniLM-L6-v2 |
|           16 | `context_text_sim`         | Semantic   | `float32` | all-MiniLM-L6-v2 |
|           17 | `tag_match`                | Structural | `float32` |                  |
|           18 | `dom_depth_norm_diff`      | Structural | `float32` |                  |
|           19 | `parent_chain_similarity`  | Structural | `float32` |                  |
|           20 | `ancestor_path_similarity` | Structural | `float32` |                  |
|           21 | `xpath_token_sim`          | Structural | `float32` |                  |
|           22 | `sibling_position_diff`    | Structural | `float32` |                  |
|           23 | `sibling_context_jaccard`  | Structural | `float32` |                  |
|           24 | `child_tag_jaccard`        | Structural | `float32` |                  |
|           25 | `label_seqsim`             | Attribute  | `float32` |                  |
|           26 | `heading_seqsim`           | Attribute  | `float32` |                  |
|           27 | `aria_seqsim`              | Attribute  | `float32` |                  |
|           28 | `placeholder_seqsim`       | Attribute  | `float32` |                  |
|           29 | `title_seqsim`             | Attribute  | `float32` |                  |
|           30 | `id_seqsim`                | Attribute  | `float32` |                  |
|           31 | `name_seqsim`              | Attribute  | `float32` |                  |
|           32 | `class_seqsim`             | Attribute  | `float32` |                  |
|           33 | `ancestor_jaccard`         | Attribute  | `float32` |                  |
|           34 | `parent_seqsim`            | Attribute  | `float32` |                  |
|           35 | `sibling_before_jaccard`   | Attribute  | `float32` |                  |
|           36 | `sibling_after_jaccard`    | Attribute  | `float32` |                  |
|           37 | `nearby_jaccard`           | Attribute  | `float32` |                  |
|           38 | `role_match`               | Attribute  | `float32` |                  |
|           39 | `type_match`               | Attribute  | `float32` |                  |
|           40 | `visible_match`            | Attribute  | `float32` |                  |
|           41 | `enabled_match`            | Attribute  | `float32` |                  |
|           42 | `id_subset`                | Attribute  | `float32` |                  |
|           43 | `dataTestId_subset`        | Attribute  | `float32` |                  |
|           44 | `attr_subset_any`          | Attribute  | `float32` |                  |
|           45 | `geo_iou`                  | Geometry   | `float32` |                  |
|           46 | `geo_center_dist`          | Geometry   | `float32` |                  |
|           47 | `geo_size_ratio`           | Geometry   | `float32` |                  |
|           48 | `bbox_dx`                  | Geometry   | `float32` |                  |
|           49 | `bbox_dy`                  | Geometry   | `float32` |                  |

</details>

<details>

<summary><strong>Complete 50-feature gain importance for the latest model</strong></summary>

| Rank | Vector index | Feature                    | Group      |      Gain | Gain share |
| ---: | -----------: | -------------------------- | ---------- | --------: | ---------: |
|    1 |           12 | `semantic_sim`             | Semantic   | 22852.957 |     43.11% |
|    2 |           10 | `text_seqsim`              | Lexical    | 12876.604 |     24.29% |
|    3 |           16 | `context_text_sim`         | Semantic   |  3569.410 |      6.73% |
|    4 |            5 | `text_exact`               | Lexical    |  3262.501 |      6.15% |
|    5 |           14 | `sibling_text_sim`         | Semantic   |  1707.919 |      3.22% |
|    6 |           13 | `parent_text_sim`          | Semantic   |  1406.107 |      2.65% |
|    7 |           22 | `sibling_position_diff`    | Structural |   988.328 |      1.86% |
|    8 |            6 | `text_substr`              | Lexical    |   971.036 |      1.83% |
|    9 |            0 | `idx_distance_log`         | Lexical    |   946.889 |      1.79% |
|   10 |           37 | `nearby_jaccard`           | Attribute  |   585.561 |      1.10% |
|   11 |           34 | `parent_seqsim`            | Attribute  |   456.066 |      0.86% |
|   12 |            4 | `text_len_ratio`           | Lexical    |   452.803 |      0.85% |
|   13 |           36 | `sibling_after_jaccard`    | Attribute  |   421.175 |      0.79% |
|   14 |           46 | `geo_center_dist`          | Geometry   |   374.177 |      0.71% |
|   15 |           32 | `class_seqsim`             | Attribute  |   356.448 |      0.67% |
|   16 |           47 | `geo_size_ratio`           | Geometry   |   284.859 |      0.54% |
|   17 |           48 | `bbox_dx`                  | Geometry   |   271.531 |      0.51% |
|   18 |           45 | `geo_iou`                  | Geometry   |   202.548 |      0.38% |
|   19 |           18 | `dom_depth_norm_diff`      | Structural |   134.391 |      0.25% |
|   20 |           33 | `ancestor_jaccard`         | Attribute  |   132.591 |      0.25% |
|   21 |            9 | `text_token_overlap`       | Lexical    |   108.140 |      0.20% |
|   22 |            2 | `new_text_len`             | Lexical    |   105.403 |      0.20% |
|   23 |           49 | `bbox_dy`                  | Geometry   |    94.791 |      0.18% |
|   24 |           24 | `child_tag_jaccard`        | Structural |    63.329 |      0.12% |
|   25 |           44 | `attr_subset_any`          | Attribute  |    58.702 |      0.11% |
|   26 |            3 | `len_diff`                 | Lexical    |    58.241 |      0.11% |
|   27 |            1 | `old_text_len`             | Lexical    |    52.799 |      0.10% |
|   28 |           35 | `sibling_before_jaccard`   | Attribute  |    49.914 |      0.09% |
|   29 |           19 | `parent_chain_similarity`  | Structural |    47.714 |      0.09% |
|   30 |           15 | `child_text_sim`           | Semantic   |    36.851 |      0.07% |
|   31 |            8 | `new_text_empty`           | Lexical    |    21.333 |      0.04% |
|   32 |           23 | `sibling_context_jaccard`  | Structural |    13.579 |      0.03% |
|   33 |           42 | `id_subset`                | Attribute  |     8.519 |      0.02% |
|   34 |           26 | `heading_seqsim`           | Attribute  |     8.115 |      0.02% |
|   35 |           21 | `xpath_token_sim`          | Structural |     7.572 |      0.01% |
|   36 |           20 | `ancestor_path_similarity` | Structural |     6.180 |      0.01% |
|   37 |           29 | `title_seqsim`             | Attribute  |     5.927 |      0.01% |
|   38 |           30 | `id_seqsim`                | Attribute  |     5.305 |      0.01% |
|   39 |           38 | `role_match`               | Attribute  |     3.997 |      0.01% |
|   40 |           27 | `aria_seqsim`              | Attribute  |     0.370 |      0.00% |
|   41 |           11 | `n_candidates`             | Lexical    |     0.056 |      0.00% |
|   42 |           31 | `name_seqsim`              | Attribute  |     0.011 |      0.00% |
|   43 |            7 | `old_text_empty`           | Lexical    |     0.000 |      0.00% |
|   44 |           17 | `tag_match`                | Structural |     0.000 |      0.00% |
|   45 |           25 | `label_seqsim`             | Attribute  |     0.000 |      0.00% |
|   46 |           28 | `placeholder_seqsim`       | Attribute  |     0.000 |      0.00% |
|   47 |           39 | `type_match`               | Attribute  |     0.000 |      0.00% |
|   48 |           40 | `visible_match`            | Attribute  |     0.000 |      0.00% |
|   49 |           41 | `enabled_match`            | Attribute  |     0.000 |      0.00% |
|   50 |           43 | `dataTestId_subset`        | Attribute  |     0.000 |      0.00% |

</details>

### Exhaustive feature-group ablation

<details open>

<summary><strong>Complete 31-combination ablation table</strong></summary>

| Rank | Included feature groups                                | Features | CV rounds | Test end-to-end Top-1 | Test end-to-end MRR | Test end-to-end NDCG@5 | Δ Top-1 vs full |
| ---: | ------------------------------------------------------ | -------: | --------: | --------------------: | ------------------: | ---------------------: | --------------: |
|    1 | lexical + semantic + structural + attribute + geometry |       50 |       120 |                94.97% |              96.38% |                 96.77% |        +0.00 pp |
|    2 | lexical + semantic + structural + geometry             |       30 |        30 |                94.65% |              96.07% |                 96.40% |        -0.31 pp |
|    3 | lexical + semantic + structural + attribute            |       45 |        65 |                94.34% |              95.88% |                 96.37% |        -0.63 pp |
|    4 | lexical + semantic + attribute + geometry              |       42 |        46 |                93.40% |              95.37% |                 96.00% |        -1.57 pp |
|    5 | lexical + semantic + attribute                         |       37 |       103 |                93.40% |              95.39% |                 96.00% |        -1.57 pp |
|    6 | lexical + semantic + structural                        |       25 |        22 |                93.08% |              95.10% |                 95.71% |        -1.89 pp |
|    7 | lexical + semantic + geometry                          |       22 |        80 |                93.08% |              94.99% |                 95.61% |        -1.89 pp |
|    8 | semantic + structural + geometry                       |       18 |        40 |                92.14% |              94.60% |                 95.45% |        -2.83 pp |
|    9 | semantic + structural + attribute + geometry           |       38 |        84 |                92.14% |              94.74% |                 95.44% |        -2.83 pp |
|   10 | lexical + structural + attribute + geometry            |       45 |        49 |                91.51% |              93.80% |                 94.59% |        -3.46 pp |
|   11 | semantic + structural                                  |       13 |        46 |                90.88% |              93.66% |                 94.52% |        -4.09 pp |
|   12 | lexical + structural + attribute                       |       40 |        82 |                90.25% |              92.93% |                 94.06% |        -4.72 pp |
|   13 | lexical + structural + geometry                        |       25 |        52 |                89.94% |              92.64% |                 93.27% |        -5.03 pp |
|   14 | lexical + attribute                                    |       32 |        73 |                89.94% |              92.34% |                 93.27% |        -5.03 pp |
|   15 | lexical + semantic                                     |       17 |        21 |                89.62% |              93.45% |                 94.47% |        -5.35 pp |
|   16 | semantic + structural + attribute                      |       33 |        32 |                89.31% |              92.79% |                 93.90% |        -5.66 pp |
|   17 | lexical + attribute + geometry                         |       37 |        69 |                89.31% |              92.15% |                 92.91% |        -5.66 pp |
|   18 | semantic + geometry                                    |       10 |        89 |                88.68% |              92.70% |                 93.80% |        -6.29 pp |
|   19 | semantic + attribute                                   |       25 |        50 |                88.68% |              92.21% |                 93.23% |        -6.29 pp |
|   20 | semantic + attribute + geometry                        |       30 |        60 |                88.36% |              92.66% |                 93.99% |        -6.60 pp |
|   21 | lexical + structural                                   |       20 |        76 |                88.36% |              91.99% |                 93.47% |        -6.60 pp |
|   22 | lexical + geometry                                     |       17 |        72 |                87.42% |              90.84% |                 91.69% |        -7.55 pp |
|   23 | semantic                                               |        5 |        68 |                84.28% |              90.10% |                 91.68% |       -10.69 pp |
|   24 | lexical                                                |       12 |        36 |                78.62% |              84.82% |                 85.99% |       -16.35 pp |
|   25 | structural + attribute + geometry                      |       33 |       116 |                64.47% |              74.24% |                 76.92% |       -30.50 pp |
|   26 | attribute + geometry                                   |       25 |       109 |                59.12% |              68.46% |                 70.52% |       -35.85 pp |
|   27 | structural + geometry                                  |       13 |        32 |                58.80% |              68.82% |                 71.14% |       -36.16 pp |
|   28 | structural + attribute                                 |       28 |        98 |                57.55% |              70.12% |                 74.06% |       -37.42 pp |
|   29 | structural                                             |        8 |        10 |                49.06% |              63.99% |                 68.20% |       -45.91 pp |
|   30 | geometry                                               |        5 |        88 |                48.11% |              58.96% |                 60.04% |       -46.86 pp |
|   31 | attribute                                              |       20 |        60 |                32.39% |              47.04% |                 48.34% |       -62.58 pp |

</details>

## Output files

A benchmark run writes `query_results.csv`, `site_results.csv`, `summary.csv`, and `run_metadata.json` to the selected output directory. Runtime fields are omitted for accuracy-only execution and included when `--runtime-mode cold` or `--runtime-mode amortized` is selected.

The `results/` directory is excluded from version control because it contains raw and repeated execution outputs. The summarized accuracy, runtime, and environment artifacts used in the reported benchmark are committed under [`artifacts/benchmark/`](artifacts/benchmark/).

A calibration run writes `tau_margin_summary.csv`, `tau_margin_query_decisions.csv`, and `calibration_metadata.json`. Automatic selection also writes `tau_margin_development_oof_curve.csv`.

## Evaluation protocol

The benchmark reads each site's `result/result.json` and uses `matchMap` as the expected old-to-new mapping. Queries whose old grouped node cannot be resolved, or whose old grouped node has `elementType == 1`, are excluded before evaluation. Every remaining query stays in the denominator. A missing prediction, failed match, or algorithm error is incorrect.

Top-1 correctness requires exact equality between the predicted and expected new grouped-node IDs. Accuracy-only and amortized execution reuse page-level preparation across queries; this reuse does not intentionally alter candidates, scores, rankings, predictions, or correctness.

## Baseline attribution and implementation sources

Each implementation is pinned to an authors' repository revision. The revisions are also recorded in every `run_metadata.json` file.

### WATER

- Paper: Shauvik Roy Choudhary, Dan Zhao, Husayn Versee, and Alessandro Orso, ["WATER: Web Application TEst Repair"](https://doi.org/10.1145/2002931.2002935), 2011.
- Authors' repository: [`shauvik/water`](https://github.com/shauvik/water)
- Pinned revision: [`c57311e56afb8e52ccd315e460e1a6f5084a7b95`](https://github.com/shauvik/water/tree/c57311e56afb8e52ccd315e460e1a6f5084a7b95)
- Benchmark boundary: WATER's candidate construction and similarity calculation are retained. The original implementation validates candidate repairs by rerunning a modified browser test; this benchmark instead returns deterministic candidate predictions for the fixed query-level protocol.

### VISTA

- Paper: Andrea Stocco, Rahulkrishna Yandrapally, and Ali Mesbah, ["Visual Web Test Repair"](https://doi.org/10.1145/3236024.3236063), ESEC/FSE 2018.
- Authors' repository: [`saltlab/vista`](https://github.com/saltlab/vista)
- Pinned revision: [`7bf2a588897637e0e1c9a2cc33e4d372a62bcaf0`](https://github.com/saltlab/vista/tree/7bf2a588897637e0e1c9a2cc33e4d372a62bcaf0)
- Benchmark boundary: the crop construction, SIFT/FAST feature checks, normalized template matching, and smallest-containing-element selection follow the pinned implementation. Numerical results can vary from the original Java/OpenCV environment because this package uses current Python OpenCV.

### SFTM

- Paper: Sacha Brisset, Romain Rouvoy, Renaud Pawlak, and Lionel Seinturier, ["SFTM: Fast Comparison of Web Documents using Similarity-based Flexible Tree Matching"](https://arxiv.org/abs/2004.12821), 2020.
- Authors' repository: [`amaris/sftm_tree_matching`](https://github.com/amaris/sftm_tree_matching)
- Pinned revision: [`4f8683b0e5a21050d1679f1c350f558a5a4171e5`](https://github.com/amaris/sftm_tree_matching/tree/4f8683b0e5a21050d1679f1c350f558a5a4171e5)
- Benchmark boundary: the source defaults, tokenization, indexing, similarity propagation, no-match edges, and one-iteration greedy matching behavior are retained. BeautifulSoup replaces Jsoup for HTML parsing; when an archived HTML file is unavailable, captured DOM metadata is used to construct the tree.

### WebEvo

- Paper: Fei Shao, Rui Xu, Wasif Haque, Jingwei Xu, Ying Zhang, Wei Yang, Yanfang Ye, and Xusheng Xiao, ["WebEvo: Taming Web Application Evolution via Detecting Semantic Structure Changes"](https://doi.org/10.1145/3460319.3464800), ISSTA 2021.
- Authors' repository: [`webevoexp/webevo`](https://github.com/webevoexp/webevo)
- Pinned revision: [`d195630b8dd5d85790d5aebb5c81c29bd03f01ca`](https://github.com/webevoexp/webevo/tree/d195630b8dd5d85790d5aebb5c81c29bd03f01ca)
- Benchmark boundary: this package implements WebEvo's semantics-based visual-search module: tag-ordered candidates, text-based shortlist, 16×16 perceptual fingerprint, grayscale/inversion stages, and the 0.60 threshold. The fixed UITestFix queries replace WebEvo's separate history and DOM-change discovery stages, and stored element screenshots replace live Selenium screenshot cropping.

### Similo

- Paper: Michel Nass, Emil Alégroth, Robert Feldt, Maurizio Leotta, and Filippo Ricca, ["Similarity-based Web Element Localization for Robust Test Automation"](https://doi.org/10.1145/3571855), ACM TOSEM, 2023.
- Authors' repository: [`michelnass/Similo2`](https://github.com/michelnass/Similo2)
- Pinned revision: [`743ddede7df92814ddc3fb7f96e09bdb508e285a`](https://github.com/michelnass/Similo2/tree/743ddede7df92814ddc3fb7f96e09bdb508e285a)
- Benchmark boundary: candidate eligibility and the original 14 manually weighted locator properties are retained. The authors' program returns the best candidate; this benchmark preserves all scored candidates so that the common evaluator can derive Top-k rankings while using the same Top-1 ordering.

## Third-party notices

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The SFTM module carries `SPDX-License-Identifier: GPL-3.0-or-later`, and the GPL version 3 text is included in [`LICENSES/GPL-3.0.txt`](LICENSES/GPL-3.0.txt).
