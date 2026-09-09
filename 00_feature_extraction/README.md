# 00_feature_extraction

The step upstream of `01_data_preparation/`: turning cleaned CGM readings into the
per-participant features the rest of the pipeline consumes.

This directory was added late, and it is not a clean origin story. Read the provenance
notes below before treating anything here as the code that produced the published numbers.

| script | produces |
|---|---|
| `01_extract_cgm_features.py` | the 15 CGM features, into `processed/cgm_features_clean_n1306.csv` |
| `02_extract_reactive_event_features.py` | reactive-event descriptors: `avg_spike_peak`, `avg_nadir_value`, `avg_drop`, `avg_time_to_nadir`, `has_reactive_hypo` |
| `03_extract_nocturnal_features.py` | `noct_nadir`, `noct_tbr70`, `noct_mean`, `noct_tbr54`, `noct_hypo_events`, `tar_250`, directional MAGE |

## Provenance

**The original feature-extraction script was not preserved.** `01_extract_cgm_features.py`
is a reimplementation, written against the stored feature table and calibrated until it
reproduced it. Twelve of the fifteen features come back exactly; `n_spikes_140` differs for
a minority of participants, and `mage` and `day_night_diff` are approximate. Its
`day_night_diff` uses a different window from the analysis and is superseded by
`01_data_preparation/04_day_night_diff_local_time.py` — use that one.

Scripts `02` and `03` are ports of the extractors that were preserved, and they are exact.

Two windows named "night" coexist in this project and are not interchangeable. The
nocturnal metrics in `03` use local 23:00–06:00. `day_night_diff` uses local 00:00–06:00
against a day window of 08:00–20:00. Both are live and both are correct for what they
measure.

## The raw-CGM cleaning step is not here

The features are computed from `cgm_all_readings_clean_final.csv`, which is the raw stream
after a two-stage QC. **No implementation of that QC survives.** The rule is fully
specified and was applied uniformly to every study group, so it can be reimplemented, but
the code that ran is gone and nothing in this repository writes that file.

The specification, for anyone reconstructing it:

**Stage 1 — duplicate records.** Drop rows sharing `(participant_id, start_datetime)` with
an earlier row, keeping the first occurrence. The cause was upstream batch ingest, which
gave a subset of participants a block of exact duplicate extras.

**Stage 2 — V/Λ sign-reversal filter.** Flag reading `g[t]` if either

- Λ-shape, peak-then-return: `Δ_up > +25` and `Δ_after < −15`
- V-shape, trough-then-return: `Δ_up < −25` and `Δ_after > +15`

with both directions additionally requiring a time gap **before and after of ≤ 10 minutes**.
Changes sustained across longer gaps are physiologically possible and are not flagged.

The rule targets sign-flip reversals only, never sustained rises or falls, so meal, insulin
and exercise responses survive it. It is shape-based rather than magnitude-based, which is
why the same thresholds apply to volatile and quiet traces alike. It is a custom rule with
no published citation. Removal rates per cohort are cohort statistics and are reported in
Methods.

A separate rate-of-change check — consecutive-reading deltas above 25 mg/dL — is published
as `01_data_preparation/01_cgm_artifact_filter.py`. That one flags and does not remove, and
is downstream of the cleaning described here.

## reactive_rate

The stored analysis table carries a `reactive_rate` column that none of these scripts
produces, and it should not be used without care. It equals

    100 * n_reactive_events / n_spikes_140

computed against a **superseded** per-reading definition of `n_spikes_140`. The column was
carried forward unchanged when `n_spikes_140` was redefined as a count of excursions, so in
the current table its denominator no longer matches the `n_spikes_140` shipped beside it,
and the values are systematically smaller than the same ratio under current definitions.
Prefer `n_reactive_events` directly, or recompute the ratio against the current spike count.
