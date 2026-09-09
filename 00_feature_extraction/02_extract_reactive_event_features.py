"""Reactive-hypoglycemia event descriptors, per participant.

These are the post-hoc characterization features, not clustering inputs. Only
`n_reactive_events` is in the 14-feature clustering set; the rest describe the shape of
the events it counts.

    n_reactive_events    excursions above 140 followed by a reading below 70 within 3 h
    has_reactive_hypo    1 if the participant has at least one such event
    avg_spike_peak       mean peak of those excursions
    avg_nadir_value      mean of the first reading below 70 in each event
    avg_drop             mean peak-minus-nadir
    avg_time_to_nadir    mean minutes from peak to that reading

`avg_spike_peak` and `avg_nadir_value` are read by 04_biomarker_orthogonality/07 as part of
the hyper and hypo axes. All six are NaN for a participant with no events, which is the
majority of the cohort.

A seventh column, `reactive_rate`, appears in the stored analysis table and is NOT produced
here. It equals 100 * n_reactive_events / n_spikes_140 against a superseded per-reading
spike count, and was carried forward unchanged after n_spikes_140 was redefined as an
excursion count, so it no longer agrees with the denominator shipped beside it. See the
README in this directory.

in   cgm_all_readings_clean_final.csv
out  processed/reactive_event_features.csv
"""
import os

import numpy as np
import pandas as pd

WINDOW = 36            # readings; 3 h at 5-minute sampling


def compute_events(g):
    """Reactive events for one participant's time-ordered glucose array.

    On crossing 140, walk to the top of the excursion, then scan forward WINDOW readings
    for the first reading below 70. That reading closes the event and the scan resumes
    from it, so one excursion contributes at most one event.
    """
    events = []
    n_spikes = 0
    i = 0
    while i < len(g):
        if g[i] > 140:
            n_spikes += 1
            peak_val, peak_idx = g[i], i
            while i < len(g) - 1 and g[i + 1] >= g[i]:
                i += 1
                if g[i] > peak_val:
                    peak_val, peak_idx = g[i], i
            end = min(i + WINDOW, len(g))
            for j in range(i, end):
                if g[j] < 70:
                    events.append({
                        'peak_val': peak_val,
                        'nadir_val': g[j],
                        'drop': peak_val - g[j],
                        'readings_to_nadir': j - peak_idx,
                        'time_to_nadir_min': (j - peak_idx) * 5,
                    })
                    i = j
                    break
        i += 1
    return events, n_spikes


def features_one(g):
    g = np.asarray(g, float)
    g = g[~np.isnan(g)]
    if len(g) < 50:
        return None
    events, _ = compute_events(g)
    return {
        'n_reactive_events': len(events),
        'has_reactive_hypo': 1 if events else 0,
        'avg_spike_peak': np.mean([e['peak_val'] for e in events]) if events else np.nan,
        'avg_nadir_value': np.mean([e['nadir_val'] for e in events]) if events else np.nan,
        'avg_drop': np.mean([e['drop'] for e in events]) if events else np.nan,
        'avg_time_to_nadir': (np.mean([e['time_to_nadir_min'] for e in events])
                              if events else np.nan),
    }


def main():
    base = os.environ.get("AIREADI_DATA_ROOT", "")
    assert base, "set AIREADI_DATA_ROOT to the data root"
    src = f"{base}/cgm_all_readings_clean_final.csv"
    out = f"{base}/processed/reactive_event_features.csv"
    os.makedirs(f"{base}/processed", exist_ok=True)

    cgm = pd.read_csv(src, usecols=['participant_id', 'start_datetime',
                                    'glucose_value_mg_dL'])
    cgm['_t'] = pd.to_datetime(cgm.start_datetime, utc=True, format='ISO8601',
                               errors='coerce')
    print(f"{len(cgm):,} readings, {cgm.participant_id.nunique():,} participants")

    rows = []
    for pid, grp in cgm.sort_values('_t').groupby('participant_id'):
        f = features_one(grp.glucose_value_mg_dL.values)
        if f is None:
            continue
        f['person_id'] = pid
        rows.append(f)

    df = pd.DataFrame(rows)
    df = df[['person_id', 'n_reactive_events', 'has_reactive_hypo', 'avg_spike_peak',
             'avg_nadir_value', 'avg_drop', 'avg_time_to_nadir']]
    df.to_csv(out, index=False)
    n_any = int(df.has_reactive_hypo.sum())
    print(f"wrote {out}  ({len(df)} participants, {n_any} with at least one event)")


if __name__ == "__main__":
    main()
