"""Nocturnal and directional-variability CGM metrics, per participant.

Computed from the Open mHealth CGM export rather than the flat table, because the export
header carries the per-participant timezone and these are local-clock quantities.

    noct_mean, noct_nadir       mean and minimum glucose in the nocturnal window
    noct_tbr70, noct_tbr54      percent of nocturnal readings below 70 / below 54
    noct_hypo_events            nocturnal episodes below 70 lasting >= 15 consecutive min
    tar_250                     percent of all readings above 250
    mage_plus, mage_minus       mean upward / downward excursion above 1 SD
    mage_ratio                  mage_minus / mage_plus

`noct_nadir` and `noct_tbr70` are read by 04_biomarker_orthogonality/07 as part of the
hypo axis.

The nocturnal window is local 23:00-06:00. This is NOT the night window used by
`day_night_diff`, which is local 00:00-06:00 against a day window of 08:00-20:00. The two
are separate definitions and both are live; see 01_data_preparation/04.

Glucose values arrive as numbers or as the sentinel strings "low" and "high", which the
device writes at the ends of its range. They are mapped to 39 and 401 mg/dL, matching the
reporting limits of the Dexcom sensors used.

in   cgm_json/<person_id>/<person_id>_DEX.json
out  processed/nocturnal_features.csv
"""
import json
import os

import numpy as np
import pandas as pd

UTC_OFFSET = {"pst": -8, "cst": -6}      # non-DST, matching the OMH header labels
NOCT_START, NOCT_END = 23, 6             # local hours; window wraps midnight
MIN_EPISODE_READINGS = 3                 # >= 15 min at 5-minute sampling


def load_participant_cgm(cgm_dir, pid):
    """Return (timezone, timestamps, glucose) or None when the export is missing/empty."""
    fp = f"{cgm_dir}/{pid}/{pid}_DEX.json"
    if not os.path.exists(fp):
        return None
    with open(fp) as fh:
        doc = json.load(fh)
    tz = doc["header"].get("timezone", "pst").lower()
    recs = doc["body"]["cgm"]
    if not recs:
        return None

    starts = [r["effective_time_frame"]["time_interval"]["start_date_time"] for r in recs]
    vals = []
    for r in recs:
        v = r["blood_glucose"]["value"]
        if isinstance(v, (int, float)):
            vals.append(float(v))
        elif isinstance(v, str):
            s = v.strip().lower()
            if s == "low":
                vals.append(39.0)
            elif s == "high":
                vals.append(401.0)
            else:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(np.nan)
        else:
            vals.append(np.nan)

    t = pd.to_datetime(pd.Series(starts), utc=True, format="ISO8601")
    return tz, t, np.array(vals, dtype=float)


def compute_metrics(cgm_dir, pid):
    loaded = load_participant_cgm(cgm_dir, pid)
    if loaded is None:
        return None
    tz, t, g = loaded

    order = np.argsort(t.values)
    t = t.iloc[order].reset_index(drop=True)
    g = g[order]
    mask = ~np.isnan(g)
    t = t[mask].reset_index(drop=True)
    g = g[mask]
    if len(g) < 5:
        return None

    local_hour = (t.dt.hour + UTC_OFFSET.get(tz, -8)) % 24
    noct_mask = (local_hour >= NOCT_START) | (local_hour < NOCT_END)
    g_noct = g[noct_mask.values]

    if len(g_noct):
        noct_mean = float(np.mean(g_noct))
        noct_nadir = float(np.min(g_noct))
        noct_tbr70 = float(np.mean(g_noct < 70) * 100)
        noct_tbr54 = float(np.mean(g_noct < 54) * 100)
    else:
        noct_mean = noct_nadir = noct_tbr70 = noct_tbr54 = np.nan

    # Nocturnal episodes: runs of consecutive nocturnal readings below 70.
    below_noct = ((g < 70) & noct_mask.values).astype(np.int8)
    events, run = 0, 0
    for v in below_noct:
        if v:
            run += 1
        else:
            if run >= MIN_EPISODE_READINGS:
                events += 1
            run = 0
    if run >= MIN_EPISODE_READINGS:
        events += 1

    tar_250 = float(np.mean(g > 250) * 100)

    # Directional MAGE on a 3-point median-smoothed series: mean up- and down-excursion
    # between turning points, counting only excursions larger than 1 SD.
    gs = pd.Series(g).rolling(3, center=True).median().bfill().ffill().to_numpy()
    sd = float(np.std(gs))
    d = np.sign(np.diff(gs))
    d[d == 0] = 1
    turns = np.where(np.diff(d) != 0)[0] + 1
    if len(turns) < 2:
        mage_plus = mage_minus = mage_ratio = np.nan
    else:
        ext = gs[turns]
        amplitudes = np.abs(np.diff(ext))
        direction = np.sign(np.diff(ext))
        qualifying = amplitudes > sd
        up = amplitudes[qualifying & (direction > 0)]
        down = amplitudes[qualifying & (direction < 0)]
        mage_plus = float(np.mean(up)) if len(up) else np.nan
        mage_minus = float(np.mean(down)) if len(down) else np.nan
        mage_ratio = (float(mage_minus / mage_plus)
                      if (mage_plus and mage_plus > 0 and not np.isnan(mage_minus))
                      else np.nan)

    return dict(person_id=int(pid), timezone=tz,
                noct_mean=noct_mean, noct_nadir=noct_nadir,
                noct_tbr70=noct_tbr70, noct_tbr54=noct_tbr54,
                noct_hypo_events=int(events), tar_250=tar_250,
                mage_plus=mage_plus, mage_minus=mage_minus, mage_ratio=mage_ratio,
                n_readings=int(len(g)))


def main():
    base = os.environ.get("AIREADI_DATA_ROOT", "")
    assert base, "set AIREADI_DATA_ROOT to the data root"
    cgm_dir = f"{base}/cgm_json"
    out = f"{base}/processed/nocturnal_features.csv"
    os.makedirs(f"{base}/processed", exist_ok=True)

    pids = sorted(d for d in os.listdir(cgm_dir)
                  if os.path.isdir(os.path.join(cgm_dir, d)))
    print(f"{len(pids)} participant folders under {cgm_dir}")

    rows, missing = [], 0
    for i, pid in enumerate(pids, 1):
        if i % 250 == 0:
            print(f"  [{i}/{len(pids)}]", flush=True)
        m = compute_metrics(cgm_dir, pid)
        if m is None:
            missing += 1
            continue
        rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} participants, {missing} skipped for missing/empty export)")


if __name__ == "__main__":
    main()
