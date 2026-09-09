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

The dataset is available from the AI-READI programme.

**No participant-level data is included in this repository.** The only data committed here
is `demo/demo_data/`, which is entirely simulated (see Demo below). Every result reported in
the manuscript is produced by running this code against the AI-READI dataset.

A small number of published summary values do appear in the source, as labelled reference
constants and in explanatory text. They are there so a run against the AI-READI cohort can
say whether it reproduced what was reported, and none of them is a cohort record or a
participant-level value:

- `04_biomarker_orthogonality/07` holds `PUBLISHED_REFERENCE`, the variance in `glucose_sd`
  and `glucose_cv` explained by HbA1c. The comparison is printed, never enforced, so the
  script runs on any dataset.
- `04_biomarker_orthogonality/07` also uses a published correlation ceiling as a live
  threshold when counting non-glycemic markers on the hypoglycemia axis.
- `03_characterization/04`, `03_characterization/05` and `06_medicated_projection/01`
  restate published effect sizes, a silhouette value, a day/night split and treated-cohort
  prevalences in the narrative text they emit alongside the recomputed values.
- Several filenames encode cohort sizes (`..._n1306.csv`, `..._n2237.csv`). These are path
  identifiers the scripts read and write, not analysis inputs.

Everything else, including every number in the emitted tables and logs, is computed at run
time from the data.

### Provenance of the feature-extraction step

One disclosure belongs here rather than only in the code. **The script that originally
computed the per-participant CGM features was not preserved.** What is published as
`00_feature_extraction/01_extract_cgm_features.py` is a reimplementation, written against
the stored feature table and calibrated against it rather than recovered.

It does not reproduce that table perfectly. Twelve of the fifteen features come back
exactly. `n_spikes_140` differs for a minority of participants, and `mage` and
`day_night_diff` are approximate; the per-feature fidelity of the reimplementation is
recorded in `00_feature_extraction/README.md`. Its `day_night_diff` additionally uses a
different window from the analysis and is superseded by
`01_data_preparation/04_day_night_diff_local_time.py`.

The consequence for a reader: the published results were produced from the stored feature
table, and re-deriving that table from raw readings with this script will reproduce most of
it exactly and three features only closely. The other two extractors in that directory,
covering the reactive-event and nocturnal features, are the preserved originals and are
exact.

The raw-CGM cleaning step that produces `cgm_all_readings_clean_final.csv` has no surviving
implementation either. Its rule is specified in full in `00_feature_extraction/README.md`,
so it can be reimplemented, but the cleaned CGM tables are inputs to this repository rather
than outputs of it.

## System requirements

**No non-standard hardware is required.** Every script runs on an ordinary desktop or
laptop CPU. There is no GPU, accelerator, cluster, or minimum-core requirement, and nothing
is parallelised beyond what NumPy and scikit-learn do by default.

| | |
|---|---|
| OS tested | Ubuntu 22.04.5 LTS (Jammy), Linux 6.8.0 x86_64 |
| Python | 3.10.11 |
| Hardware used for the timings below | 4 CPU cores, 31 GiB RAM |
| Other OSes | Not tested. The code is pure Python with no OS-specific calls or paths, so macOS and Windows are expected to work; we have not verified this. |

Package versions are pinned in `requirements.txt`: numpy 1.26.4, pandas 2.3.3,
scikit-learn 1.7.2, scipy 1.15.3, statsmodels 0.13.5, matplotlib 3.7.1, diptest 0.10.0,
joblib 1.2.0, pyarrow.

Memory is dominated by the raw CGM table, which is read whole. On the full AI-READI cohort
that file is roughly 450 MB on disk; a machine with 8 GB of RAM is comfortable. The demo
needs far less.

## Installation

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Measured install time: about 30 seconds** on the machine above — 3.2 s to create the
virtual environment and 24-26 s for `pip install` (24.1 s with a cold package cache,
26.3 s warm; the two are within noise of each other because the wheels are prebuilt). The
resulting environment is about 650 MB. There is nothing to compile.

## Demo

The demo data is entirely simulated and contains no participant information. It is
constructed with three latent archetypes built in, so the pipeline has structure to recover
and can be verified to run end to end. Its cluster statistics, including the silhouette
score, are artifacts of the generator and bear no relation to the results reported in the
manuscript, where the patterns lie on a continuum rather than forming separated groups. The
demo demonstrates that the code executes; it is not evidence for any scientific claim.

`demo/demo_data/` is the generated tree and is committed, so step 1 is optional.

**The demo tree is complete — you do not need to run `00_feature_extraction/`.** Every
per-participant feature the analysis scripts read is already in it. That directory documents
how the features are derived from raw readings; it is not a prerequisite, and running it
against the demo root is unnecessary.

### Commands

```bash
python demo/make_demo_data.py --out demo/demo_data     # optional: regenerate the data
export AIREADI_DATA_ROOT=$PWD/demo/demo_data
python 01_data_preparation/04_day_night_diff_local_time.py
python 02_clustering/01_kmeans_k3_fit.py
```

### Expected output

The last command prints, among its diagnostics:

```
FEATURES: 14 = canonical-15 minus `n_lows_54` (near-duplicate of pct_below_54)

SIZES  Spiker 20 / Stable 30 / Hypo-Prone 10

=== VALIDATION ===
silhouette 0.7581 | Calinski-Harabasz 186.2 | Davies-Bouldin 0.514

K-selection sweep (same pipeline, K=2..6):
   K   silhouette      CH      DB   sizes
   2   0.6126      70.0  0.686   30;30
   3   0.7581     186.2  0.514   30;20;10
   4   0.7519     254.3  0.423   30;20;7;3
seed stability (20 seeds): ARI mean 1.0000 +/- 0.0000
```

Three non-empty clusters, K=3 winning the sweep, and identical assignments across 20 seeds.
The numbers are reproducible to the digit: the generator is seeded and the fit uses
`random_state=42`. It writes labels, an analysis table, the fitted KMeans and scaler, and a
log under `demo/demo_data/canonical_n14_rerun/`.

The same log also prints `ARI vs canonical-15 (UTC) = 1.0000` and `ARI vs pruned-10
(tz-fixed) = 1.0000`. Both agreements are generator artifacts: the demo gives every label
set the same memberships. On the AI-READI cohort those lines read 0.9367 and 0.5835.

### Measured run time

| step | time |
|---|---|
| `make_demo_data.py` (76 participants, 10 days each) | 23.6 s |
| `01_data_preparation/04_day_night_diff_local_time.py` | 2.5 s |
| `02_clustering/01_kmeans_k3_fit.py` | 2.1 s |
| **the three commands above** | **under 30 s** |

The demo is a quick check that the code installs and runs, not a full pipeline run. The
three commands above are the whole of it. Other scripts can be pointed at the same data
root and most of them work, but that is not what the demo is for and is not something we
verify here; the heavier classification scripts take several minutes each because of the
repeated cross-validation and bootstrapping, and two cannot run at all (below).

### What the demo does not cover

Two scripts need inputs that no script in this repository produces, so no synthetic dataset
can stand them up: `03_characterization/04_hypo_prone_robustness_battery.py` reads an
`excursion_morphology/` tree, and `05_external_validation/02_cgmacros_matched_cohort.py`
needs the real CGMacros release.

One column, `reactive_rate`, is present but empty. Its definition is recoverable only as a
ratio against a superseded spike count, so the generator declines to fabricate a value; see
the comment in `demo/make_demo_data.py`.

If you do run `00_feature_extraction/01` against the demo root out of curiosity, it rewrites
`demo/demo_data/processed/cgm_features_clean_n1306.csv` in place. The column set matches, but
its `mage`, `n_spikes_140` and `day_night_diff` differ slightly, so the file shows as
modified; re-run `make_demo_data.py` to restore the committed copy.

## Instructions for use

To run the analyses on the real dataset rather than the demo:

```bash
export AIREADI_DATA_ROOT=/path/to/derived/data
python 00_feature_extraction/01_extract_cgm_features.py     # only if the feature table is
python 00_feature_extraction/03_extract_nocturnal_features.py  # not already built
python 01_data_preparation/01_cgm_artifact_filter.py
python 01_data_preparation/02_build_analysis_table.py
python 01_data_preparation/03_wearable_cgm_coverage_qc.py
python 01_data_preparation/04_day_night_diff_local_time.py
python 02_clustering/01_kmeans_k3_fit.py
... and so on, in the directory order given under Pipeline below
```

Every script reads `AIREADI_DATA_ROOT` and fails immediately if it is unset. Scripts are
run individually, in directory order; later stages consume the outputs of earlier ones.
The one ordering constraint inside a directory is that `01_data_preparation/04` must run
before `02_clustering/01`, because `04` writes the local-time `day_night_diff` that the fit
reads, as `local_time_refit/processed/day_night_diff_local_all.csv`. `04` also writes a
10-feature refit of its own into the same directory (`labels/`, `processed/`, `models/`,
`logs/`); nothing downstream reads that, and it is provenance for the timezone fix rather
than a pipeline input.
`01_data_preparation/03` is independent of the other three and can run at any point.

Nothing is written outside `AIREADI_DATA_ROOT`. The pipeline creates `local_time_refit/`,
`canonical_n14_rerun/`, `logs/`, `plots/`, and `processed/analysis_table.csv` inside it.

Reproducibility: seed 42 throughout, `KMeans(n_init=20)`. Given the same input tree, every
script is deterministic.

Expected run time on the full cohort is longer than the demo, dominated by the same
repeated-CV and bootstrap scripts; the raw-CGM passes in `01_data_preparation/` and
`06_medicated_projection/` scale with the size of the CGM tables.

## Input data manifest

These paths must exist under `AIREADI_DATA_ROOT` before the pipeline is run. They are the
derived data root, not the raw AI-READI download. `00_feature_extraction/` covers the step
that turns cleaned CGM readings into per-participant features, with the caveats given in
that directory's README; the raw-CGM cleaning that produces `cgm_all_readings_clean_final.csv`
is specified there but has no surviving implementation, so the cleaned CGM tables are inputs
to this repository rather than outputs of it.

| path | columns used | read by |
|---|---|---|
| `participants/participants.tsv` | `person_id`, `clinical_site` (TSV, tab-separated) | `01/04`, `06/01` |
| `cgm_all_readings_deduped.csv` | `participant_id`, `start_datetime`, `glucose_value_mg_dL` | `01/01` |
| `cgm_all_readings_clean_final.csv` | same three | `01/04`, `03/02`, `03/03`, `03/05`, `06/01` |
| `cgm_all_readings_clean_final_medicated.csv` | same three | `01/04`, `06/01` |
| `cgm_json/<pid>/<pid>_DEX.json` | Open mHealth; `body.cgm[].effective_time_frame` | `01/03` |
| `wearable/stress/garmin_vivosmart5/<pid>/<pid>_stress.json` | Open mHealth; `body.stress[]` | `01/03` |
| `wearable/activity/garmin_vivosmart5/<pid>/<pid>_activity.json` | Open mHealth; `body.activity[]`, `activity_name`, `base_movement_quantity` | `01/03` |
| `clinical_data/observation.csv` | `person_id`, `observation_concept_id`, `value_as_number` | `03/01`, `04/02`, `04/06`, `04/08` |
| `CSV/lifestyle_all.csv` | per-person wearable and survey aggregates | `01/02` |
| `CSV/clinical_merged_clean.csv` | `person_id`, `study_group` | `01/02` |
| `output/hypo_clustering_unmedicated.csv` | `person_id`, `hypo_k3` | `01/01`, `01/02` |
| `output/hypo_clustering_all_n2237.csv` | `person_id`, `study_group` | `06/01` |
| `output/five_group_labels_n1306.csv` | `person_id`, `five_group_name` | `01/03` |
| `processed/analysis_table_n1306_clean.csv` | CGM features + labs + wearables + `timezone`, `study_group`, `hypo_k3` | `01/03`, `01/04`, `02/01`, `06/01` |
| `processed/cgm_features_clean_n1306.csv` | `person_id` + the 15 CGM features | `03/04` |
| `archive_tzfix_backup_20260801/analysis_table_n1306_pruned.csv` | pre-timezone-fix snapshot of the pruned-10 table; its `day_night_diff` must be the UTC variant, which `01/04` asserts (r > 0.999) before applying the fix | `01/04` |
| `pruned_n10_rerun/labels/hypo_clustering_pruned_n1306.csv` | `person_id`, `hypo_k3_pruned` (pruned-10 labels, **not** the analysis labels — see below) | `02/01` |
| `external_validation/results/cgmacros_features_labels.csv` | `participant_id` + the 15 features (independent CGMacros cohort) | `05/01`, `05/02` |
| `excursion_morphology/` | `lows.parquet`, `excursions.parquet`, `wear_qc.csv`, and two log CSVs | `03/04` |

The last entry is produced by an upstream analysis that is not published here, so
`03_characterization/04` cannot be run from this repository alone. `05_external_validation/02`
additionally needs the raw CGMacros release (per-subject CSVs and `bio.csv`).

### The `pruned_n10_*` trees are provenance, not results

Nothing reported in the manuscript is computed from these. The analysis feature set is the
14 features in `02_clustering/01` (`F14`), and its labels are `hypo_k3_n14` under
`canonical_n14_rerun/`. The two `pruned_n10_*` trees are an earlier 10-feature branch, kept
because two steps still read them:

- `01_data_preparation/04` loads the pre-timezone-fix snapshot of the pruned-10 table
  (`archive_tzfix_backup_20260801/`) only to assert that its stored `day_night_diff` is the
  UTC-buggy variant, which is what establishes the timezone bug the script then fixes. It must
  be the snapshot: a working copy that has already had the fix applied stores the local variant
  and will not pass the assertion. Its own outputs land in `local_time_refit/`.
- `02_clustering/01` reads the pruned-10 labels solely to report `ARI vs pruned-10
  (tz-fixed)` in its fit log. On the AI-READI cohort that ARI is **0.5835** - the two
  partitions genuinely differ, which is why the label column was renamed (below).

`local_time_refit/processed/day_night_diff_local_all.csv` is a pipeline handoff rather than
provenance: `01_data_preparation/04` writes it and `02_clustering/01` reads it for the
local-time day/night means it substitutes before fitting. It is not part of the 10-feature
branch — earlier revisions wrote it under a directory named for that branch, which is why
the directory was renamed.

The 10-feature set also appears legitimately inside
`03_characterization/04_hypo_prone_robustness_battery.py` as `P10`, one perturbation arm of
the HP robustness battery alongside `C15`. That is a sensitivity analysis, not the headline
feature set.

### Label column naming

The analysis label column is **`hypo_k3_n14`**. Earlier revisions of this code wrote the
same values under a second column named `hypo_k3_pruned`, a leftover from the 10-feature
branch, and downstream scripts read that alias. The alias has been removed: `02_clustering/01`
now writes `hypo_k3_n14` only, and every downstream script reads it.

The name mattered because `hypo_k3_pruned` also exists, with different values, in
`pruned_n10_rerun/labels/hypo_clustering_pruned_n1306.csv`. A derived tree written by an
older revision may still carry the alias; re-run from `02_clustering/01` to regenerate it.

### Identifier convention

The raw CGM tables key on **`participant_id`**; every other file keys on **`person_id`**.
They hold the same integer. Scripts that touch both cast explicitly.

### Laboratory analytes

Fifteen analytes are carried in the analysis table, split by whether the marker is
interpretable on a non-fasted draw. The cohort has no fasting flag, so the split is enforced
by self-report (next section).

| | analytes |
|---|---|
| fasting-independent (full cohort) | `hba1c`, `hdl`, `ldl`, `total_cholesterol`, `crp`, `alt`, `ast`, `creatinine`, `urine_albumin`, `wbc` |
| fasting-dependent (>= 12 h subset) | `fasting_glucose`, `insulin`, `c_peptide`, `homa_ir_corrected`, `triglycerides` |

`insulin` and `fasting_insulin` are separate columns and both are required. Insulin and
C-peptide are stored in **ng/mL**; `03_characterization/01` and
`04_biomarker_orthogonality/06` convert to uU/mL by x28.70 and rescale `homa_ir_corrected`,
which is stored with a x6 constant, by x4.783. Four derived columns are also read:
`tg_hdl`, `hepatic_ir`, `beta_hyper`, and `reactive_rate` (see the note under Demo).
Anthropometry (`bmi`, `whr`, `waist`, `age`) is read from the same table.

### Fasting gate: OMOP concept 2005200151

`clinical_data/observation.csv` rows with `observation_concept_id == 2005200151` carry
**hours since last ate**, self-reported. `value_as_number` becomes `fast_h`, and the
`fast_h >= 12` subset is the only cohort on which fasting-dependent markers are evaluated.

This one integer therefore gates roughly half the biomarker analyses. It appears in the code
only as a bare literal, in four scripts: `03_characterization/01:89`,
`04_biomarker_orthogonality/02:84`, `04_biomarker_orthogonality/06:74`, and
`04_biomarker_orthogonality/08:64`. The 12-hour threshold is `FAST_MIN` in each.

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
the night window (00:00–06:00), computed on local clock time. Two windows named "night"
coexist in this project and are not interchangeable: this one, and the local 23:00–06:00
window used by the nocturnal metrics in `00_feature_extraction/03`. Both are live and each is
correct for what it measures.

The fifteenth candidate feature, `n_lows_54`, is not in the table above: it is dropped before
fitting as the count form of `pct_below_54`, the same construct on a different scale. This is
what `02_clustering/01` reports as `14 = canonical-15 minus n_lows_54`.

Clusters are named by rule, not by inspection: Hypo-Prone is the cluster with the highest
median `n_lows_70`; Spiker is the cluster with the highest median `n_spikes_140` among the
remaining two; Stable is the third.

## Pipeline

Run in directory order. Later stages depend on the outputs of earlier ones.

### 00_feature_extraction

| script | what it does |
|---|---|
| `01_extract_cgm_features.py` | The 15 CGM features from cleaned 5-minute readings. A reimplementation, not the original producer — see the directory README |
| `02_extract_reactive_event_features.py` | Reactive-event descriptors: spike peak, nadir, drop, time to nadir |
| `03_extract_nocturnal_features.py` | Nocturnal metrics on local 23:00-06:00, and directional MAGE, from the Open mHealth export |

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
