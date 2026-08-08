# CGM glycemic patterns — analysis code

Analysis code for the CGM phenotyping study in the AI-READI cohort: K=3 clustering of
continuous glucose monitoring features in untreated participants, characterization of the
resulting patterns, and tests of whether routine laboratory biomarkers can identify them.

This repository contains the code that produces the numbers. Figure-drawing code is not
included.

## Data

AI-READI v3.0.0. The analytic cohort is the untreated participants (no glucose-lowering
medication) with sufficient CGM coverage. Two treated cohorts, oral medication and insulin,
are used only for the projection analysis. Cohort sizes are reported in the paper.

The dataset is available from the AI-READI programme. No participant data, and no derived
values or summary statistics, are included in this repository; every number is produced by
running the code against the dataset.

## Setup

```bash
pip install -r requirements.txt
export AIREADI_DATA_ROOT=/path/to/derived/data
```

Every script reads `AIREADI_DATA_ROOT` and fails immediately if it is unset. Scripts are
run individually, in the order below.

## Features

The clustering uses 14 per-participant CGM features:

| domain | features |
|---|---|
| level | `mean_glucose` |
| variability | `glucose_sd`, `glucose_cv`, `mage` |
| hyperglycemia burden | `pct_above_140`, `pct_above_180`, `n_spikes_140` |
| hypoglycemia burden | `pct_below_70`, `pct_below_54`, `n_lows_70`, `n_reactive_events` |
| kinetics | `avg_rise_rate`, `avg_fall_rate` |
| circadian | `day_night_diff` |

`day_night_diff` is the mean glucose difference between the day window (08:00–20:00) and
the night window (00:00–06:00), computed on local clock time.

Clusters are named by rule, not by inspection: Hypo-Prone is the cluster with the highest
median `n_lows_70`; Spiker is the cluster with the highest median `n_spikes_140` among the
remaining two; Stable is the third.

## Pipeline

Run in directory order. Later stages depend on the outputs of earlier ones.

### 01_data_preparation

| script | what it does |
|---|---|
| `01_cgm_artifact_filter.py` | Flags consecutive-reading deltas > 25 mg/dL as sensor artifacts |
| `02_build_analysis_table.py` | Builds the per-participant analysis table |
| `03_wearable_cgm_coverage_qc.py` | CGM and wearable coverage per participant-day; eligibility for the behaviour analyses |
| `04_day_night_diff_local_time.py` | Converts CGM timestamps to local clock time and computes `day_night_diff` |

### 02_clustering

| script | what it does |
|---|---|
| `01_kmeans_k3_fit.py` | K=3 fit: median imputation, StandardScaler, KMeans(n_init=20, random_state=42) |
| `02_validation_cascade.py` | Permutation, seed stability, bootstrap, split-half, random-forest recovery, silhouette decomposition |
| `03_gaussian_copula_null.py` | Separation test against a unimodal null matching each feature's marginal and the rank-correlation structure |
| `04_pca_dip_test.py` | Two-component PCA of the clustering space, Hartigan dip test on PC1, diagnostic-stage composition per pattern |

### 03_characterization

| script | what it does |
|---|---|
| `01_biomarker_characterization.py` | 15 analytes across the three patterns, split by whether the marker requires a fasted draw |
| `02_circadian_excursion_decomposition.py` | Splits the evening glucose difference into baseline and excursion components |
| `03_circadian_ancova_betas.py` | Nocturnal, dawn and amplitude contrasts adjusted for mean glucose |
| `04_hypo_prone_robustness_battery.py` | Four pre-specified thresholds testing whether Hypo-Prone is a stable population |
| `05_hypo_prone_low_event_timing.py` | Local-time distribution of low readings, separating compression artifact from waking physiology |
| `06_behaviour_anthropometry_tests.py` | Ten wearable activity, sleep and stress features plus BMI and waist-to-hip ratio across the patterns |

### 04_biomarker_orthogonality

| script | what it does |
|---|---|
| `01_threeway_classification.py` | CGM-only vs biomarker-only vs combined, same rows, 3-class membership |
| `02_subtype_classification.py` | The same question at finer resolution (mild vs extreme Spiker, four label sets) |
| `03_hba1c_decomposition.py` | Whether the biomarker-panel signal reduces to HbA1c alone |
| `04_class_imbalance_decomposition.py` | Whether low minority-class recall is a threshold effect or absent signal |
| `05_detection_gap_extreme_spiker.py` | The Hypo-Prone detection design applied to the opposite extreme |
| `06_single_biomarker_classification.py` | Each marker alone, and the five-index panel, against the CGM positive control; also marker correlation with spike burden and high-IR prevalence by pattern |
| `07_hba1c_mean_decomposition.py` | Variance in each CGM feature explained by HbA1c, biomarker correlation with the hyper and hypo axes, and the Spiker-Stable HbA1c gap before and after adjustment for mean glucose |
| `08_hp_detection_recall.py` | Hypo-Prone recall from CGM features versus HbA1c alone and the non-fasting panel, plus the fasted paired arm on identical rows |

### 05_external_validation

| script | what it does |
|---|---|
| `01_kselection_cgmacros.py` | K-selection sweep in an independent CGM cohort (CGMacros) |
| `02_cgmacros_matched_cohort.py` | Projection onto the HbA1c-matched subset using the same features and windows |

### 06_medicated_projection

| script | what it does |
|---|---|
| `01_project_treated_cohorts.py` | Assigns the oral-medication and insulin cohorts to the three patterns |

## Which analysis supports which figure

| figure | analysis code |
|---|---|
| 1 — three patterns on a glycemic continuum | `02_clustering/01`, `02_clustering/02`, `02_clustering/03`, `02_clustering/04`, `05_external_validation/01` |
| 2 — patterns differ in glucose physiology, not measured behaviour | `01_data_preparation/03`, `03_characterization/02`, `03_characterization/03`, `03_characterization/06` |
| 3 — biomarkers cannot classify pattern membership | `04_biomarker_orthogonality/01`; panels are recomputed inside the figure script from the analysis table |
| 4 — biomarkers capture mostly the mean | panels a–c are computed inside the figure script from the analysis table; `03_characterization/01` provides the biomarker characterization |
| 5 — a reproducible, lab-invisible hypoglycemia-prone group | `03_characterization/04`, `03_characterization/05`, `04_biomarker_orthogonality/05`, `04_biomarker_orthogonality/08`, `05_external_validation/02`, `06_medicated_projection/01` |

## Statistical conventions

- Random seed 42 throughout; `KMeans(n_init=20)`.
- Cluster separation is assessed against the Gaussian-copula null in `02_clustering/03`, which
  preserves each feature's marginal and the rank-correlation structure. The label-shuffle
  permutation in the validation cascade destroys all grouping, so it tests only that the
  partition is not random; it is a sanity check, not evidence of separation.
- The CGM feature set is a positive control, not an independent predictor: the labels are
  K-means assignments on those same features, so its classification accuracy is a ceiling.
- QUICKI is a deterministic monotone function of the same insulin and glucose values that
  define HOMA-IR. The two carry identical rank information and are not independent tests.
- Classification: StandardScaler fit within each training fold, pooled out-of-fold
  predictions, balanced accuracy over `RepeatedStratifiedKFold(5x10)`, 2000-resample
  bootstrap 95% percentile confidence intervals, per-class recall reported alongside.
- Classifiers are unweighted; class imbalance is handled on the evaluation side through
  balanced accuracy and AUC.
- Gradient boosting: where a per-class recall is computed on fewer than about 25 members per
  training fold, the scikit-learn default `min_samples_leaf=20` exceeds the whole minority
  class, so no leaf can isolate it and recall collapses regardless of signal. Those cells are
  re-run at `min_samples_leaf=2` and both values are reported.
- Group comparisons: Kruskal-Wallis across the three patterns with Benjamini-Hochberg FDR,
  eta-squared effect sizes, pairwise Mann-Whitney contrasts.

## Environment

Python 3.10.11, numpy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3,
statsmodels 0.13.5.
