"""The 15 per-participant CGM features, from cleaned 5-minute readings.

PROVENANCE — read this before using the output for anything.

The script that originally produced the stored feature table was not preserved. This file
is the surviving extractor: a reimplementation written against the stored table and
calibrated on a sample of participants until it reproduced it. How closely it does so
varies by feature, and that matters when reading its output:

    reproduced exactly
        mean_glucose, glucose_sd, glucose_cv, pct_above_140, pct_above_180,
        pct_below_70, pct_below_54, n_lows_70, n_lows_54, n_reactive_events,
        avg_rise_rate, avg_fall_rate
    near-exact, differing for a minority of participants
        n_spikes_140
    approximate
        mage, day_night_diff

So twelve of the fifteen are reproduced exactly and three are not. Treat `mage` and
`n_spikes_140` from this script as very close but not identical to the values behind the
published numbers, and do NOT use its `day_night_diff`: that feature is computed on a
day[06-24] / night[00-06] split here, whereas the analysis uses local 08-20 vs 00-06 and
derives it in 01_data_preparation/04_day_night_diff_local_time.py, which supersedes this.
The per-feature fidelity of this reimplementation is recorded in the README in this
directory.

`n_lows_70` and `pct_below_54` are point counts and percentages of readings below the
threshold. There is no duration, persistence or compression filter at the feature stage;
artifact handling happens in the QC that produces the cleaned input (see the README in
this directory, and 01_data_preparation/01 for the rate-of-change flagging).

Assumes 5-minute sampling (Dexcom G6/G4). The reactive and spike window is 36 readings,
i.e. 3 hours.

in   cgm_all_readings_clean_final.csv
out  processed/cgm_features_clean_n1306.csv
"""
import os

import numpy as np
import pandas as pd

FEATURES = ['mean_glucose', 'glucose_sd', 'glucose_cv', 'mage', 'pct_above_140',
            'pct_above_180', 'pct_below_70', 'pct_below_54', 'n_lows_70', 'n_lows_54',
            'n_spikes_140', 'avg_rise_rate', 'avg_fall_rate', 'day_night_diff',
            'n_reactive_events']

WINDOW = 36            # readings; 3 h at 5-minute sampling


def _spikes_reactive(g):
    """n_reactive_events and n_spikes_140.

    The reactive-event rule: on crossing 140, walk to the top of the excursion, then look
    forward WINDOW readings for a reading below 70. One event per excursion, and the scan
    resumes at the low. n_spikes_140 counts upward crossings of 140.
    """
    n = len(g)
    events = 0
    i = 0
    while i < n:
        if g[i] > 140:
            peak = g[i]
            while i < n - 1 and g[i + 1] >= g[i]:
                i += 1
                if g[i] > peak:
                    peak = g[i]
            end = min(i + WINDOW, n)
            for j in range(i, end):
                if g[j] < 70:
                    events += 1
                    i = j
                    break
        i += 1
    n_spikes = int(((g[:-1] <= 140) & (g[1:] > 140)).sum())
    return n_spikes, events


def _mage(g):
    """Mean amplitude of the excursions between turning points that exceed 1 SD."""
    sd = g.std()
    if sd == 0 or len(g) < 3:
        return np.nan
    d = np.diff(g)
    s = np.sign(d)
    s[s == 0] = 1
    tp = [0] + [k for k in range(1, len(s)) if s[k] != s[k - 1]] + [len(g) - 1]
    exc = [abs(g[tp[k + 1]] - g[tp[k]]) for k in range(len(tp) - 1)]
    vals = [e for e in exc if e > sd]
    return float(np.mean(vals)) if vals else np.nan


def extract_one(g, hr):
    """g: glucose in mg/dL, time-ordered, 5-minute. hr: hour-of-day per reading, or None.

    Returns None for a participant with fewer than 50 readings.
    """
    g = np.asarray(g, float)
    g = g[~np.isnan(g)]
    if len(g) < 50:
        return None
    d = np.diff(g)
    ns, nr = _spikes_reactive(g)

    dnd = np.nan
    if hr is not None:
        hr = np.asarray(hr)[:len(g)]
        day, night = g[hr >= 6], g[hr < 6]
        if len(day) and len(night):
            dnd = float(day.mean() - night.mean())

    return {
        'mean_glucose': float(g.mean()),
        'glucose_sd': float(g.std()),
        'glucose_cv': float(g.std() / g.mean() * 100) if g.mean() else np.nan,
        'mage': _mage(g),
        'pct_above_140': float((g > 140).mean() * 100),
        'pct_above_180': float((g > 180).mean() * 100),
        'pct_below_70': float((g < 70).mean() * 100),
        'pct_below_54': float((g < 54).mean() * 100),
        'n_lows_70': int((g < 70).sum()),
        'n_lows_54': int((g < 54).sum()),
        'n_spikes_140': ns,
        'avg_rise_rate': float(d[d > 0].mean() / 5) if (d > 0).any() else 0.0,
        'avg_fall_rate': float((-d[d < 0]).mean() / 5) if (d < 0).any() else 0.0,
        'day_night_diff': dnd,
        'n_reactive_events': nr,
    }


def extract_cohort(df, pid_col, time_col, glu_col):
    """Per-participant feature frame from a long CGM table."""
    df = df[[pid_col, time_col, glu_col]].copy()
    df[glu_col] = pd.to_numeric(df[glu_col], errors='coerce')
    t = pd.to_datetime(df[time_col], errors='coerce', utc=True, format='ISO8601')
    df['_hr'] = t.dt.hour
    df['_ord'] = t

    rows = []
    for pid, grp in df.groupby(pid_col):
        grp = grp.sort_values('_ord')
        g = grp[glu_col].values
        hr = grp['_hr'].values if grp['_hr'].notna().any() else None
        feat = extract_one(g, hr)
        if feat is None:
            continue
        feat['person_id'] = pid
        feat['n_readings'] = int(np.isfinite(g).sum())
        rows.append(feat)

    out = pd.DataFrame(rows)
    return out[['person_id', 'n_readings'] + FEATURES] if len(out) else out


def main():
    base = os.environ.get("AIREADI_DATA_ROOT", "")
    assert base, "set AIREADI_DATA_ROOT to the data root"
    src = f"{base}/cgm_all_readings_clean_final.csv"
    out = f"{base}/processed/cgm_features_clean_n1306.csv"
    os.makedirs(f"{base}/processed", exist_ok=True)

    print(f"reading {src} ...")
    cgm = pd.read_csv(src, usecols=['participant_id', 'start_datetime',
                                    'glucose_value_mg_dL'])
    print(f"  {len(cgm):,} readings, {cgm.participant_id.nunique():,} participants")

    feats = extract_cohort(cgm, 'participant_id', 'start_datetime', 'glucose_value_mg_dL')
    feats.to_csv(out, index=False)
    print(f"wrote {out}  ({len(feats)} participants x {len(FEATURES)} features)")
    print("\nNOTE: day_night_diff here is the day[06-24]/night[00-06] variant. The analysis "
          "uses the local 08-20 vs 00-06 feature from 01_data_preparation/04.")


if __name__ == "__main__":
    main()
