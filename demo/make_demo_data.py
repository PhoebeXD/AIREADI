"""Generate a synthetic demo dataset for this pipeline.

The demo data is entirely simulated and contains no participant information. It is
constructed with three latent archetypes built in, so the pipeline has structure to
recover and can be verified to run end to end. Its cluster statistics, including the
silhouette score, are artifacts of the generator and bear no relation to the results
reported in the manuscript, where the patterns lie on a continuum rather than forming
separated groups. The demo demonstrates that the code executes; it is not evidence for
any scientific claim.

Every value is drawn from the seeded random number generator below. The glucose traces
come from a toy circadian-plus-meal model, not a physiological simulator, and the
laboratory values are drawn from textbook population ranges.

Purpose: give someone who does not have AI-READI access a small tree they can point
AIREADI_DATA_ROOT at, so the pipeline runs end to end and its outputs can be inspected.

The tree is complete: every per-participant feature the analysis scripts read is already in
it, so `00_feature_extraction/` does NOT need to be run first. That directory is there to
document how the features are derived from raw readings, not as a prerequisite.

Layout written (mirrors the AI-READI v3.0.0 derived-data root the scripts read):

    README.md                      <- the notice above, written into the data root
    participants/participants.tsv
    cgm_all_readings_deduped.csv
    cgm_all_readings_clean_final.csv
    cgm_all_readings_clean_final_medicated.csv
    cgm_json/<pid>/<pid>_DEX.json
    wearable/stress/garmin_vivosmart5/<pid>/<pid>_stress.json
    wearable/activity/garmin_vivosmart5/<pid>/<pid>_activity.json
    clinical_data/observation.csv
    CSV/lifestyle_all.csv
    CSV/clinical_merged_clean.csv
    output/hypo_clustering_unmedicated.csv
    output/hypo_clustering_all_n2237.csv
    output/five_group_labels_n1306.csv
    processed/analysis_table_n1306_clean.csv
    processed/cgm_features_clean_n1306.csv
    pruned_n10_rerun/labels/hypo_clustering_pruned_n1306.csv
    archive_tzfix_backup_20260801/analysis_table_n1306_pruned.csv
    external_validation/results/cgmacros_features_labels.csv

Not covered. Two scripts need inputs that no script in this repository produces, so the
demo cannot stand them up:
    03_characterization/04_hypo_prone_robustness_battery.py -- needs an excursion_morphology/
        tree (lows.parquet, excursions.parquet, wear_qc.csv, and two log CSVs) built by an
        upstream analysis that is not published here.
    05_external_validation/02_cgmacros_matched_cohort.py -- needs the real CGMacros release
        (per-subject CSVs and bio.csv), which is a different cohort with its own schema.

Usage
    python demo/make_demo_data.py --out demo/demo_data
    export AIREADI_DATA_ROOT=$PWD/demo/demo_data
    python 01_data_preparation/04_day_night_diff_local_time.py
    python 02_clustering/01_kmeans_k3_fit.py
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

NOTICE = """\
The demo data is entirely simulated and contains no participant information. It is
constructed with three latent archetypes built in, so the pipeline has structure to
recover and can be verified to run end to end. Its cluster statistics, including the
silhouette score, are artifacts of the generator and bear no relation to the results
reported in the manuscript, where the patterns lie on a continuum rather than forming
separated groups. The demo demonstrates that the code executes; it is not evidence for
any scientific claim.

One demo artifact is worth naming, because it is easy to misread. The generator gives every
label set the same memberships, so `02_clustering/01` prints `ARI vs canonical-15 = 1.0000`
and `ARI vs pruned-10 (tz-fixed) = 1.0000` here. Those agreements are built in, not found.
On the AI-READI cohort the same two lines read 0.9367 and 0.5835: the analysis labels
(`hypo_k3_n14`) and the older pruned-10 labels are genuinely different partitions."""

SEED = 20260903
N_UNMED = 60          # untreated participants -- the clustering cohort
N_MED = 16            # treated participants -- used only by 06_medicated_projection
N_DAYS = 10
STEP_MIN = 5
PER_DAY = 24 * 60 // STEP_MIN            # 288 readings/day at 5-minute cadence
START_DATE = "2023-05-01"                # arbitrary; no relation to any real collection window

# Clinical sites and their non-DST UTC offsets, matching the SITE_TZ / UTC_OFFSET maps in
# 01_data_preparation/04_day_night_diff_local_time.py and 06_medicated_projection/01.
SITES = {"UAB": ("cst", -6), "UW": ("pst", -8), "UCSD": ("pst", -8)}
FAST_CONCEPT_ID = 2005200151             # hours since last ate, self-report (OMOP observation)

# Three latent archetypes. The clustering is not told about these; they exist only to make the
# synthetic cohort heterogeneous enough that KMeans(k=3) lands on three non-empty clusters.
ARCHETYPES = {
    "Stable": dict(
        n=30, base=95.0, base_sd=6.0, meal_peak=45.0, meal_sd=10.0,
        noise=3.5, dawn=6.0, undershoot=0.0, night_drift=-3.0,
    ),
    "Spiker": dict(
        n=20, base=108.0, base_sd=9.0, meal_peak=105.0, meal_sd=22.0,
        noise=5.0, dawn=12.0, undershoot=0.0, night_drift=-4.0,
    ),
    "Hypo-Prone": dict(
        n=10, base=92.0, base_sd=7.0, meal_peak=70.0, meal_sd=18.0,
        noise=4.5, dawn=5.0, undershoot=42.0, night_drift=-9.0,
    ),
}

CGM_FEATURES_15 = [
    "mean_glucose", "glucose_sd", "glucose_cv", "mage", "pct_above_140", "pct_above_180",
    "pct_below_70", "pct_below_54", "n_lows_70", "n_lows_54", "n_spikes_140",
    "avg_rise_rate", "avg_fall_rate", "day_night_diff", "n_reactive_events",
]


# ---------------------------------------------------------------- glucose trace simulator
def simulate_trace(rng, params, offset_hours):
    """One participant's 5-minute CGM series over N_DAYS.

    Built in LOCAL clock time (meals sit at local mealtimes), then handed back with the
    matching UTC timestamps, because the export stores UTC and the day/night windows are
    local-clock concepts.
    """
    n = N_DAYS * PER_DAY
    local_idx = pd.date_range(f"{START_DATE}T00:00:00", periods=n, freq=f"{STEP_MIN}min")
    hour = local_idx.hour.values + local_idx.minute.values / 60.0

    base = params["base"] + rng.normal(0, params["base_sd"])
    g = np.full(n, base, dtype=float)

    # Circadian shape: an overnight decline plus a dawn rise.
    g += params["night_drift"] * np.exp(-((hour - 3.0) ** 2) / 8.0)
    g += params["dawn"] * np.exp(-((hour - 7.0) ** 2) / 3.0)

    # Meals at roughly 07:30 / 12:30 / 18:30 local, with per-day jitter, plus occasional snacks.
    for day in range(N_DAYS):
        day0 = day * PER_DAY
        for meal_h, scale in ((7.5, 0.85), (12.5, 1.0), (18.5, 1.1)):
            centre = meal_h + rng.normal(0, 0.6)
            amp = max(0.0, rng.normal(params["meal_peak"], params["meal_sd"])) * scale
            rise = rng.uniform(0.55, 0.95)          # hours to peak
            decay = rng.uniform(1.4, 2.6)           # hours of return
            t = hour[day0:day0 + PER_DAY] - centre
            shape = np.where(t < 0, np.exp(-(t ** 2) / (2 * rise ** 2)),
                                    np.exp(-t / decay))
            shape = np.where(np.abs(t) > 8, 0.0, shape)
            g[day0:day0 + PER_DAY] += amp * shape

            # Reactive undershoot: a post-meal dip below baseline, which is what produces
            # low readings and n_reactive_events for the Hypo-Prone archetype.
            if params["undershoot"] > 0:
                dip_c = centre + rise + rng.uniform(1.6, 2.8)
                dip_a = max(0.0, rng.normal(params["undershoot"], params["undershoot"] * 0.3))
                td = hour[day0:day0 + PER_DAY] - dip_c
                g[day0:day0 + PER_DAY] -= dip_a * np.exp(-(td ** 2) / 1.2)

    # AR(1) sensor/physiological noise so consecutive readings are correlated.
    noise = np.zeros(n)
    e = rng.normal(0, params["noise"], n)
    for i in range(1, n):
        noise[i] = 0.75 * noise[i - 1] + e[i]
    g += noise

    g = np.clip(g, 40.0, 400.0)
    utc_idx = local_idx + pd.Timedelta(hours=-offset_hours)   # local = utc + offset
    return utc_idx, np.round(g, 1)


# ---------------------------------------------------------------- feature extractor
def compute_dynamics(g, utc_hours, offset_hours):
    """The 15 clustering features.

    Mirrors the canonical extractor in 06_medicated_projection/01_project_treated_cohorts.py
    (compute_dynamics). Kept as a copy rather than an import because that module runs its
    analysis at import time. If the extractor there changes, change it here too.
    """
    if len(g) < 100:
        return None
    r = {
        "mean_glucose": np.mean(g), "glucose_sd": np.std(g),
        "glucose_cv": 100 * np.std(g) / (np.mean(g) + 1e-10),
        "pct_above_140": 100 * np.mean(g > 140), "pct_above_180": 100 * np.mean(g > 180),
        "pct_below_70": 100 * np.mean(g < 70), "pct_below_54": 100 * np.mean(g < 54),
        "n_lows_70": int((g < 70).sum()), "n_lows_54": int((g < 54).sum()),
    }
    n_spikes = 0
    for i in range(len(g)):
        if g[i] > 140 and (i == 0 or g[i - 1] <= 140):
            n_spikes += 1
    r["n_spikes_140"] = n_spikes

    diffs = np.diff(g) / 5
    r["avg_rise_rate"] = np.mean(diffs[diffs > 0]) if np.any(diffs > 0) else 0
    r["avg_fall_rate"] = np.mean(np.abs(diffs[diffs < 0])) if np.any(diffs < 0) else 0

    if len(g) > 20:
        peaks = argrelextrema(g, np.greater, order=6)[0]
        troughs = argrelextrema(g, np.less, order=6)[0]
        if len(peaks) > 0 and len(troughs) > 0:
            extrema = sorted(list(peaks) + list(troughs))
            exc = [abs(g[extrema[i + 1]] - g[extrema[i]]) for i in range(len(extrema) - 1)]
            large = [e for e in exc if e > np.std(g)]
            r["mage"] = np.mean(large) if large else 0
        else:
            r["mage"] = 0
    else:
        r["mage"] = 0

    def dnd(hours):
        day = (hours >= 8) & (hours < 20)
        night = (hours >= 0) & (hours < 6)
        if day.sum() > 10 and night.sum() > 10:
            return float(np.mean(g[day]) - np.mean(g[night]))
        return 0.0

    local_hours = (utc_hours + offset_hours) % 24
    r["day_night_diff"] = dnd(local_hours)
    r["day_night_diff_utc"] = dnd(utc_hours)      # helper only; not a pipeline feature

    window = 36
    n_reactive = 0
    i = 0
    while i < len(g):
        if g[i] > 140:
            while i < len(g) - 1 and g[i + 1] >= g[i]:
                i += 1
            end = min(i + window, len(g))
            for j in range(i, end):
                if g[j] < 70:
                    n_reactive += 1
                    i = j
                    break
        i += 1
    r["n_reactive_events"] = n_reactive

    # ---- panel-b features (04_biomarker_orthogonality/07) --------------------------
    # avg_spike_peak / avg_nadir_value: the reactive-event extractor in
    # 00_feature_extraction/02_extract_reactive_event_features.py. Same >140 spike, same
    # 36-reading window; it additionally records the excursion peak and the first
    # reading below 70, and averages each over a participant's events.
    events = []
    i = 0
    while i < len(g):
        if g[i] > 140:
            peak_val = g[i]
            while i < len(g) - 1 and g[i + 1] >= g[i]:
                i += 1
                if g[i] > peak_val:
                    peak_val = g[i]
            end = min(i + window, len(g))
            for j in range(i, end):
                if g[j] < 70:
                    events.append((peak_val, g[j]))
                    i = j
                    break
        i += 1
    r["avg_spike_peak"] = np.mean([e[0] for e in events]) if events else np.nan
    r["avg_nadir_value"] = np.mean([e[1] for e in events]) if events else np.nan

    # noct_nadir / noct_tbr70: 00_feature_extraction/03_extract_nocturnal_features.py.
    # Note the nocturnal window is local 23:00-06:00, which is NOT the 00:00-06:00
    # night window that day_night_diff uses.
    noct = (local_hours >= 23) | (local_hours < 6)
    g_noct = g[noct]
    r["noct_nadir"] = float(np.min(g_noct)) if len(g_noct) else np.nan
    r["noct_tbr70"] = float(np.mean(g_noct < 70) * 100) if len(g_noct) else np.nan

    # reactive_rate is left empty on purpose. Its definition WAS recovered from the
    # earliest table (processed/analysis_n1084.csv), where it is exactly
    # 100 * n_reactive_events / n_spikes_140 -- but against the SUPERSEDED per-reading
    # n_spikes_140 (range 0-1355), not the excursion-based count shipped today (0-126).
    # The column was carried forward byte-identical while its denominator was redefined,
    # so in the current analysis table the implied denominator matches the shipped
    # n_spikes_140 on 1 of 552 rows. Reproducing either the stale ratio or a
    # current-denominator substitute would put a misleading value in the demo, so the
    # column is present and empty.
    r["reactive_rate"] = np.nan
    return r


# ---------------------------------------------------------------- labs, wearables, surveys
def make_labs(rng, arch):
    """Plausible textbook-range laboratory values, nudged by archetype. Simulated."""
    ir = {"Stable": 0.0, "Spiker": 1.0, "Hypo-Prone": 0.35}[arch]
    hba1c = np.clip(rng.normal(5.4 + 0.45 * ir, 0.35), 4.2, 8.5)
    fpg = np.clip(rng.normal(94 + 12 * ir, 10), 65, 190)
    # insulin and c_peptide are stored in ng/mL, per the unit convention documented in
    # 03_characterization/01_biomarker_characterization.py
    insulin = np.clip(rng.lognormal(np.log(0.35 + 0.25 * ir), 0.45), 0.05, 4.0)
    c_pep = np.clip(rng.lognormal(np.log(1.7 + 0.9 * ir), 0.4), 0.2, 9.0)
    # homa_ir_corrected is the stored x6-constant variant the scripts rescale by 4.783
    homa = insulin * fpg / (6.0 * 405.0) * 28.70
    tg = np.clip(rng.lognormal(np.log(95 + 45 * ir), 0.42), 30, 600)
    hdl = np.clip(rng.normal(56 - 9 * ir, 12), 20, 110)
    ldl = np.clip(rng.normal(108 + 9 * ir, 28), 30, 240)
    return {
        "hba1c": round(float(hba1c), 2),
        "fasting_glucose": round(float(fpg), 1),
        "insulin": round(float(insulin), 4),
        "fasting_insulin": round(float(insulin), 4),
        "c_peptide": round(float(c_pep), 3),
        "homa_ir_corrected": round(float(homa), 4),
        "triglycerides": round(float(tg), 1),
        "hdl": round(float(hdl), 1),
        "ldl": round(float(ldl), 1),
        "total_cholesterol": round(float(ldl + hdl + tg / 5.0), 1),
        "tg_hdl": round(float(tg / hdl), 3),
        "hepatic_ir": round(float(insulin * 28.70 * tg / 100.0), 3),
        "crp": round(float(np.clip(rng.lognormal(np.log(1.4 + 0.9 * ir), 0.8), 0.1, 40)), 2),
        "alt": round(float(np.clip(rng.lognormal(np.log(20 + 7 * ir), 0.4), 5, 200)), 1),
        "ast": round(float(np.clip(rng.lognormal(np.log(21 + 4 * ir), 0.35), 5, 200)), 1),
        "creatinine": round(float(np.clip(rng.normal(0.88, 0.19), 0.35, 2.5)), 2),
        "urine_albumin": round(float(np.clip(rng.lognormal(np.log(7.0), 0.9), 0.5, 400)), 2),
        "wbc": round(float(np.clip(rng.normal(6.6 + 0.5 * ir, 1.7), 2.0, 18.0)), 2),
    }


def make_wearable(rng, arch):
    """Per-participant Garmin aggregates and one survey field. Simulated."""
    steps = float(np.clip(rng.normal(7100, 2600), 500, 22000))
    active = float(np.clip(rng.normal(41, 19), 0, 220))
    deep = float(np.clip(rng.normal(17, 5), 2, 40))
    rem = float(np.clip(rng.normal(21, 6), 3, 45))
    return {
        "avg_daily_steps": round(steps, 1),
        "avg_active_min": round(active, 1),
        "avg_sedentary_min": round(float(np.clip(rng.normal(690, 130), 200, 1200)), 1),
        "avg_total_sleep_hr": round(float(np.clip(rng.normal(6.9, 1.0), 3.0, 11.0)), 2),
        "sleep_efficiency_pct": round(float(np.clip(rng.normal(87, 6), 50, 100)), 1),
        "deep_sleep_pct": round(deep, 1),
        "rem_sleep_pct": round(rem, 1),
        "light_sleep_pct": round(100.0 - deep - rem, 1),
        "avg_mean_stress": round(float(np.clip(rng.normal(33, 9), 5, 90)), 1),
        "avg_max_stress": round(float(np.clip(rng.normal(88, 9), 40, 100)), 1),
        "alcohol_ever": int(rng.random() < 0.62),
    }


def make_anthro(rng, arch):
    ir = {"Stable": 0.0, "Spiker": 1.0, "Hypo-Prone": 0.3}[arch]
    bmi = float(np.clip(rng.normal(26.5 + 3.4 * ir, 4.6), 16.0, 55.0))
    whr = float(np.clip(rng.normal(0.88 + 0.05 * ir, 0.07), 0.65, 1.25))
    return {
        "bmi": round(bmi, 2),
        "whr": round(whr, 3),
        "waist": round(float(np.clip(rng.normal(94 + 11 * ir, 14), 55, 165)), 1),
        "age": int(np.clip(rng.normal(58, 13), 18, 92)),
    }


# ---------------------------------------------------------------- Open mHealth JSON writers
def write_cgm_json(path, utc_idx, g):
    body = [
        {"effective_time_frame": {"time_interval": {
            "start_date_time": ts.strftime("%Y-%m-%dT%H:%M:%SZ")}},
         "blood_glucose": {"unit": "mg/dL", "value": float(v)}}
        for ts, v in zip(utc_idx, g)
    ]
    with open(path, "w") as f:
        json.dump({"header": {"simulated": True}, "body": {"cgm": body}}, f)


def write_stress_json(path, rng, utc_idx):
    # 3-minute cadence, the Garmin vivosmart5 stress sampling the QC script walks.
    body = []
    for ts in utc_idx[::1]:
        body.append({
            "effective_time_frame": {"date_time": ts.strftime("%Y-%m-%dT%H:%M:%SZ")},
            "stress": {"unit": "score", "value": int(np.clip(rng.normal(33, 18), 0, 100))},
        })
    with open(path, "w") as f:
        json.dump({"header": {"simulated": True}, "body": {"stress": body}}, f)


def write_activity_json(path, rng, first_local_day):
    body = []
    for day in range(N_DAYS):
        d = pd.Timestamp(first_local_day) + pd.Timedelta(days=day)
        for hour in range(6, 23):
            name = "sedentary" if rng.random() < 0.55 else rng.choice(
                ["walking", "running", "generic"], p=[0.7, 0.1, 0.2])
            st = d + pd.Timedelta(hours=hour)
            en = st + pd.Timedelta(minutes=int(rng.integers(10, 55)))
            steps = 0 if name == "sedentary" else int(rng.integers(200, 1400))
            body.append({
                "effective_time_frame": {"time_interval": {
                    "start_date_time": st.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "end_date_time": en.strftime("%Y-%m-%dT%H:%M:%SZ")}},
                "activity_name": str(name),
                "base_movement_quantity": {"unit": "steps", "value": steps},
            })
    with open(path, "w") as f:
        json.dump({"header": {"simulated": True}, "body": {"activity": body}}, f)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="demo/demo_data", help="output data root")
    ap.add_argument("--no-json", action="store_true",
                    help="skip the per-participant JSON exports (faster; "
                         "01_data_preparation/03 then has nothing to read)")
    args = ap.parse_args()

    root = os.path.abspath(args.out)
    for sub in ["participants", "processed", "output", "logs", "CSV", "clinical_data",
                "cgm_json", "pruned_n10_rerun/labels", "archive_tzfix_backup_20260801",
                "external_validation/results",
                "wearable/stress/garmin_vivosmart5", "wearable/activity/garmin_vivosmart5"]:
        os.makedirs(f"{root}/{sub}", exist_ok=True)

    # The notice travels with the data tree: anyone who copies demo_data/ elsewhere, or
    # lands in it without the generator, still reads what this data is and is not.
    with open(f"{root}/README.md", "w") as fh:
        fh.write(f"# Synthetic demo data\n\n{NOTICE}\n\nGenerated by "
                 f"`demo/make_demo_data.py` (seed {SEED}). Do not edit by hand; re-run\n"
                 f"the generator instead.\n")

    rng = np.random.default_rng(SEED)
    site_names = list(SITES)

    # Assign archetypes and treatment status.
    plan = []
    pid = 10001
    for arch, p in ARCHETYPES.items():
        for _ in range(p["n"]):
            plan.append((pid, arch, "unmedicated"))
            pid += 1
    for i in range(N_MED):
        plan.append((pid, "Spiker" if i % 2 else "Stable",
                     "oral" if i % 2 else "insulin"))
        pid += 1

    part_rows, unmed_rows, med_rows, feat_rows, obs_rows = [], [], [], [], []

    for pid, arch, treat in plan:
        site = site_names[rng.integers(0, len(site_names))]
        tzname, offset = SITES[site]
        params = ARCHETYPES[arch]
        utc_idx, g = simulate_trace(rng, params, offset)

        part_rows.append({"person_id": pid, "clinical_site": site,
                          "recommended_split": "train"})

        readings = pd.DataFrame({
            "participant_id": pid,
            "start_datetime": utc_idx.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "glucose_value_mg_dL": g,
        })
        (med_rows if treat != "unmedicated" else unmed_rows).append(readings)

        f = compute_dynamics(g, utc_idx.hour.values.astype(float), offset)
        f.update(person_id=pid, archetype=arch, treatment=treat,
                 clinical_site=site, timezone=tzname)
        f.update(make_labs(rng, arch))
        f.update(make_wearable(rng, arch))
        f.update(make_anthro(rng, arch))
        feat_rows.append(f)

        # Self-reported hours since last ate; about half the cohort clears the 12 h bar so
        # the fasting-dependent arm of 03_characterization/01 has rows in every pattern.
        obs_rows.append({"person_id": pid, "observation_concept_id": FAST_CONCEPT_ID,
                         "value_as_number": round(float(rng.uniform(1, 4) if rng.random() < 0.5
                                                        else rng.uniform(12, 15)), 1)})

        if not args.no_json:
            for d, suffix, writer in (
                ("cgm_json", "_DEX.json", lambda p: write_cgm_json(p, utc_idx, g)),
                ("wearable/stress/garmin_vivosmart5", "_stress.json",
                 lambda p: write_stress_json(p, rng, utc_idx)),
                ("wearable/activity/garmin_vivosmart5", "_activity.json",
                 lambda p: write_activity_json(p, rng, START_DATE)),
            ):
                os.makedirs(f"{root}/{d}/{pid}", exist_ok=True)
                writer(f"{root}/{d}/{pid}/{pid}{suffix}")

    feats = pd.DataFrame(feat_rows)
    unmed = feats[feats.treatment == "unmedicated"].copy()

    # ---- raw CGM tables -------------------------------------------------------------
    pd.concat(unmed_rows, ignore_index=True).to_csv(
        f"{root}/cgm_all_readings_clean_final.csv", index=False)
    pd.concat(unmed_rows + med_rows, ignore_index=True).to_csv(
        f"{root}/cgm_all_readings_deduped.csv", index=False)
    pd.concat(med_rows, ignore_index=True).to_csv(
        f"{root}/cgm_all_readings_clean_final_medicated.csv", index=False)

    pd.DataFrame(part_rows).to_csv(f"{root}/participants/participants.tsv",
                                   sep="\t", index=False)
    pd.DataFrame(obs_rows).to_csv(f"{root}/clinical_data/observation.csv", index=False)

    # ---- labels ---------------------------------------------------------------------
    # The demo's starting labels come from the archetype. The pipeline re-fits its own
    # clustering in 02_clustering/01; these only bootstrap the scripts that expect a
    # pre-existing label column.
    code = {"Spiker": 0, "Stable": 1, "Hypo-Prone": 2}
    unmed["hypo_k3"] = unmed.archetype.map(code)
    unmed["cluster_name"] = unmed.archetype

    unmed[["person_id", "hypo_k3"]].to_csv(
        f"{root}/output/hypo_clustering_unmedicated.csv", index=False)

    sg = {"unmedicated": "healthy", "oral":
          "oral_medication_and_or_non_insulin_injectable_medication_controlled",
          "insulin": "insulin_dependent"}
    feats["study_group"] = feats.treatment.map(sg)
    # give the untreated arm both non-treated study groups
    mask = (feats.treatment == "unmedicated") & (np.arange(len(feats)) % 3 == 0)
    feats.loc[mask, "study_group"] = "pre_diabetes_lifestyle_controlled"
    feats[["person_id", "study_group"]].to_csv(
        f"{root}/output/hypo_clustering_all_n2237.csv", index=False)
    feats.loc[feats.treatment == "unmedicated", ["person_id", "study_group"]].to_csv(
        f"{root}/CSV/clinical_merged_clean.csv", index=False)

    unmed = unmed.merge(feats[["person_id", "study_group"]], on="person_id", how="left")
    unmed["five_group_name"] = np.where(
        unmed.hypo_k3 == 2, "Hypo-Prone",
        np.where(unmed.study_group == "healthy", "Healthy-" + unmed.archetype,
                 "PreDM-" + unmed.archetype))
    unmed[["person_id", "five_group_name"]].to_csv(
        f"{root}/output/five_group_labels_n1306.csv", index=False)

    # ---- lifestyle aggregates -------------------------------------------------------
    life_cols = ["person_id", "avg_daily_steps", "avg_active_min", "avg_sedentary_min",
                 "avg_total_sleep_hr", "sleep_efficiency_pct", "deep_sleep_pct",
                 "rem_sleep_pct", "light_sleep_pct", "avg_mean_stress", "avg_max_stress",
                 "alcohol_ever"]
    feats[life_cols].to_csv(f"{root}/CSV/lifestyle_all.csv", index=False)

    # ---- the analysis table ---------------------------------------------------------
    # NOTE: no script in this repository derives the 15 CGM features from raw CGM into this
    # table -- the pipeline starts one step downstream and reads it. The demo therefore has to
    # build it, which it does with the same extractor 06_medicated_projection/01 uses.
    # day_night_diff here is the LOCAL-time version, matching that extractor.
    drop = ["archetype", "treatment", "day_night_diff_utc"]
    at = unmed.drop(columns=[c for c in drop if c in unmed.columns])
    at.to_csv(f"{root}/processed/analysis_table_n1306_clean.csv", index=False)

    # 03_characterization/04 reads a bare per-participant feature table alongside the
    # analysis table; same features, no labels or labs. Column set matches what
    # 00_feature_extraction/01 emits, so running that step against the demo root is
    # optional rather than a prerequisite.
    feat_tbl = at[["person_id"] + CGM_FEATURES_15].copy()
    feat_tbl.insert(1, "n_readings", N_DAYS * PER_DAY)
    feat_tbl.to_csv(f"{root}/processed/cgm_features_clean_n1306.csv", index=False)

    # ---- the frozen pruned branch ---------------------------------------------------
    # 01_data_preparation/04 asserts that this table's stored day_night_diff is the UTC
    # variant (r > 0.999 against its own recomputation), so it gets the UTC column.
    pruned = unmed.drop(columns=[c for c in drop if c in unmed.columns]).copy()
    pruned["day_night_diff"] = unmed["day_night_diff_utc"].values
    pruned["hypo_k3_pruned"] = unmed["hypo_k3"].values
    pruned.to_csv(f"{root}/archive_tzfix_backup_20260801/analysis_table_n1306_pruned.csv",
                  index=False)
    pruned[["person_id", "hypo_k3_pruned"]].to_csv(
        f"{root}/pruned_n10_rerun/labels/hypo_clustering_pruned_n1306.csv", index=False)

    # ---- CGMacros stand-in ----------------------------------------------------------
    # 05_external_validation/01 reads a pre-computed feature table for the independent
    # CGMacros cohort. This is a synthetic stand-in with the same columns so the script
    # runs; 05_external_validation/02 additionally needs the real CGMacros release and is
    # not covered by this demo.
    ext = unmed[CGM_FEATURES_15].sample(n=45, random_state=SEED, replace=True).reset_index(drop=True)
    ext.insert(0, "participant_id", [f"CGMacros-{i:03d}" for i in range(1, len(ext) + 1)])
    ext["dnd_820"] = ext["day_night_diff"]
    ext.to_csv(f"{root}/external_validation/results/cgmacros_features_labels.csv", index=False)

    # ---- report ---------------------------------------------------------------------
    n_read = N_DAYS * PER_DAY
    print("SYNTHETIC DEMO DATA -- no participant information.")
    print("Cluster statistics from this data are artifacts of the generator and are not")
    print("evidence for any scientific claim. See the README in the data root.")
    print(f"  root                 {root}")
    print(f"  seed                 {SEED}")
    print(f"  participants         {len(feats)}  ({len(unmed)} untreated, {N_MED} treated)")
    print(f"  CGM                  {N_DAYS} days x {PER_DAY}/day = {n_read:,} readings each")
    print(f"  archetype counts     {unmed.archetype.value_counts().to_dict()}"
          if "archetype" in unmed else "")
    print("\n  feature ranges (untreated):")
    for c in ["mean_glucose", "glucose_cv", "n_lows_70", "n_spikes_140",
              "pct_above_140", "day_night_diff", "n_reactive_events"]:
        s = unmed[c]
        print(f"    {c:20s} {s.min():8.2f} .. {s.max():8.2f}   median {s.median():8.2f}")


if __name__ == "__main__":
    main()
