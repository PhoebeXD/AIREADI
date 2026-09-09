"""Per-participant wearable and CGM coverage QC.

Computes CGM and Garmin coverage per participant-day, the overlap between them, and
sensor-error flags, then applies the eligibility rule used by the behaviour analyses.

Coverage rules
    CGM covered day     >= 18 h with any reading (>= 216 readings at 5-min cadence)
    Garmin covered day  >= 12 h
    eligible            >= 7 overlapping covered days

in   cgm_json/<pid>/, wearable/stress/, wearable/activity/
out  output/phase3_qc_eligible_participants.csv, logs/phase3_qc_summary.md,
     logs/phase3_qc_per_day_detail.csv
"""
import os
from pathlib import Path
import json
import csv
import sys
import time
import pandas as pd
import numpy as np

_root = os.environ.get("AIREADI_DATA_ROOT", "")
assert _root, "set AIREADI_DATA_ROOT to the data root"
ROOT = Path(_root)
CGM_DIR  = ROOT / 'cgm_json'
STR_DIR  = ROOT / 'wearable' / 'stress'    / 'garmin_vivosmart5'
ACT_DIR  = ROOT / 'wearable' / 'activity'  / 'garmin_vivosmart5'
ANA_TBL  = ROOT / 'processed' / 'analysis_table_n1306_clean.csv'
LABELS   = ROOT / 'output' / 'five_group_labels_n1306.csv'

OUT_DIR  = ROOT / 'output'
LOG_DIR  = ROOT / 'logs'
PARTIC_OUT = OUT_DIR / 'phase3_qc_eligible_participants.csv'
SUMMARY    = LOG_DIR / 'phase3_qc_summary.md'
PER_DAY    = LOG_DIR / 'phase3_qc_per_day_detail.csv'

THR_GARMIN_DAYS   = 7
THR_OVERLAP_DAYS  = 7
THR_STRESS_DAYS   = 4
THR_CGM_HOURS_DAY = 18
THR_GAR_HOURS_DAY = 12


def parse_cgm(pid):
    """Returns {date_str: n_readings, ...}."""
    p = CGM_DIR / str(pid) / f'{pid}_DEX.json'
    if not p.is_file(): return {}
    try:
        with open(p) as f: d = json.load(f)
    except Exception:
        return {}
    out = {}
    for x in d.get('body', {}).get('cgm', []):
        tf = x.get('effective_time_frame', {})
        if 'time_interval' in tf:
            dt = tf['time_interval'].get('start_date_time', '')
        else:
            dt = tf.get('date_time', '')
        if not dt: continue
        date = dt[:10]
        out[date] = out.get(date, 0) + 1
    return out


def parse_stress(pid):
    """Returns dict per date:
       {date: {'n_total': int, 'n_usable': int, 'n_zero': int, 'n_100': int,
               'hours_worn': int, 'max_run_100': int, 'max_run_0': int}}
    """
    p = STR_DIR / str(pid) / f'{pid}_stress.json'
    if not p.is_file(): return {}, {}
    try:
        with open(p) as f: d = json.load(f)
    except Exception:
        return {}, {}

    body = d.get('body', {}).get('stress', [])
    if not body: return {}, {}

    # Build per-date rolling stats. We need consecutive-run lengths for 0 and 100.
    by_date = {}
    last_date = None
    cur_run_val = None
    cur_run_len = 0
    cur_run_date = None
    max_run_100 = {}; max_run_0 = {}
    hours_seen = {}

    for x in body:
        tf = x.get('effective_time_frame', {})
        dt = tf.get('date_time') or tf.get('time_interval', {}).get('start_date_time')
        if not dt: continue
        date = dt[:10]
        hh = dt[11:13]
        try:
            val = int(x.get('stress', {}).get('value', -99))
        except Exception:
            val = -99

        rec = by_date.setdefault(date, {'n_total': 0, 'n_usable': 0, 'n_zero': 0, 'n_100': 0})
        rec['n_total'] += 1
        if 0 <= val <= 100:
            rec['n_usable'] += 1
            if val == 0:   rec['n_zero'] += 1
            if val == 100: rec['n_100'] += 1

        # hours worn (any non-NaN sample)
        hs = hours_seen.setdefault(date, set())
        hs.add(hh)

        # consecutive-run tracking (across minutes for sentinel-extreme runs)
        run_val = val if val in (0, 100) else None
        if run_val is not None and run_val == cur_run_val and date == cur_run_date:
            cur_run_len += 1
        else:
            # close previous run, possibly attribute to date
            if cur_run_val == 100 and cur_run_date is not None:
                max_run_100[cur_run_date] = max(max_run_100.get(cur_run_date, 0), cur_run_len)
            if cur_run_val == 0 and cur_run_date is not None:
                max_run_0[cur_run_date] = max(max_run_0.get(cur_run_date, 0), cur_run_len)
            cur_run_val = run_val
            cur_run_len = 1 if run_val is not None else 0
            cur_run_date = date
    # close trailing run
    if cur_run_val == 100 and cur_run_date is not None:
        max_run_100[cur_run_date] = max(max_run_100.get(cur_run_date, 0), cur_run_len)
    if cur_run_val == 0 and cur_run_date is not None:
        max_run_0[cur_run_date] = max(max_run_0.get(cur_run_date, 0), cur_run_len)

    for date, rec in by_date.items():
        rec['hours_worn']  = len(hours_seen.get(date, set()))
        rec['max_run_100'] = max_run_100.get(date, 0)
        rec['max_run_0']   = max_run_0.get(date, 0)
    return by_date, {}


def parse_activity(pid):
    """Returns per-date totals: {date: {'steps': int, 'active_min': int, 'sedentary_min': int}}."""
    p = ACT_DIR / str(pid) / f'{pid}_activity.json'
    if not p.is_file(): return {}
    try:
        with open(p) as f: d = json.load(f)
    except Exception:
        return {}
    out = {}
    for x in d.get('body', {}).get('activity', []):
        tf = x.get('effective_time_frame', {}).get('time_interval', {})
        st = tf.get('start_date_time')
        en = tf.get('end_date_time')
        if not st: continue
        date = st[:10]
        # duration in minutes
        try:
            dt_st = pd.Timestamp(st)
            dt_en = pd.Timestamp(en) if en else dt_st
            dur_min = max(0.0, (dt_en - dt_st).total_seconds() / 60.0)
        except Exception:
            dur_min = 0.0
        name = x.get('activity_name', '') or ''
        try:
            steps = float(x.get('base_movement_quantity', {}).get('value', 0) or 0)
        except Exception:
            steps = 0.0
        rec = out.setdefault(date, {'steps': 0.0, 'active_min': 0.0, 'sedentary_min': 0.0})
        rec['steps'] += steps
        if name in ('walking', 'running', 'generic'):
            rec['active_min'] += dur_min
        elif name == 'sedentary':
            rec['sedentary_min'] += dur_min
    return out


def main():
    ana = pd.read_csv(ANA_TBL, usecols=['person_id', 'cluster_name'])
    lab = pd.read_csv(LABELS, usecols=['person_id', 'five_group_name'])
    df  = ana.merge(lab, on='person_id', how='left')
    pids = df['person_id'].astype(int).tolist()
    print(f'Cohort n={len(pids)}', flush=True)

    rows_pid = []
    rows_day = []

    t0 = time.time()
    for i, pid in enumerate(pids, 1):
        if i % 50 == 0 or i == len(pids):
            elapsed = time.time() - t0
            eta = elapsed / i * (len(pids) - i)
            print(f'  [{i:>4}/{len(pids)}]  elapsed={elapsed:5.0f}s  ETA={eta:5.0f}s', flush=True)

        cgm = parse_cgm(pid)                              # {date: n_readings}
        stress_by_d, _ = parse_stress(pid)                # {date: dict}
        act_by_d = parse_activity(pid)                    # {date: dict}

        cgm_days = set()
        for date, n in cgm.items():
            if n >= 216:                                  # ≥18h × 12 readings/hr (5-min)
                cgm_days.add(date)

        stress_days_any = set()
        stress_days_usable = set()
        gar_days = set()
        sensor_high_days = 0
        sensor_low_days  = 0
        n_stress_obs_total = 0
        n_stress_usable_total = 0

        for date, rec in stress_by_d.items():
            n_stress_obs_total    += rec['n_total']
            n_stress_usable_total += rec['n_usable']
            stress_days_any.add(date)
            if rec['n_usable'] >= 1:
                stress_days_usable.add(date)
            if rec['hours_worn'] >= THR_GAR_HOURS_DAY:
                gar_days.add(date)
            if rec['max_run_100'] >= 720:                 # 12h stuck at 100
                sensor_high_days += 1
            if rec['max_run_0']   >= 720:                 # 12h stuck at 0
                sensor_low_days  += 1

        # activity days (steps>0 OR active_min>0)
        act_days = set()
        steps_extreme_days = 0
        steps_low_active_anomaly = 0
        n_steps_total = 0.0
        for date, rec in act_by_d.items():
            if rec['steps'] > 0 or rec['active_min'] > 0:
                act_days.add(date)
            n_steps_total += rec['steps']
            if rec['steps'] > 50000:
                steps_extreme_days += 1
            if rec['steps'] < 100 and rec['active_min'] > 60:
                steps_low_active_anomaly += 1

        overlap_days = cgm_days & gar_days
        garmin_all_days = stress_days_any | act_days

        # Inclusion
        reasons = []
        if len(garmin_all_days) < THR_GARMIN_DAYS:  reasons.append('garmin_days<7')
        if len(overlap_days)    < THR_OVERLAP_DAYS: reasons.append('overlap_days<7')
        if len(stress_days_usable) < THR_STRESS_DAYS: reasons.append('stress_days<4')
        included = (len(reasons) == 0)

        rows_pid.append({
            'person_id': pid,
            'cluster_name': df.loc[df['person_id']==pid, 'cluster_name'].iloc[0],
            'five_group_name': df.loc[df['person_id']==pid, 'five_group_name'].iloc[0],
            'included_phase3': included,
            'exclusion_reason': ';'.join(reasons),
            'cgm_total_days': len(cgm),
            'cgm_covered_days_18h': len(cgm_days),
            'garmin_total_days': len(garmin_all_days),
            'garmin_covered_days_12h': len(gar_days),
            'overlap_days': len(overlap_days),
            'stress_days_any': len(stress_days_any),
            'stress_days_usable': len(stress_days_usable),
            'activity_days': len(act_days),
            'n_stress_obs_total': n_stress_obs_total,
            'n_stress_obs_usable': n_stress_usable_total,
            'n_steps_total': n_steps_total,
            'sensor_high_days': sensor_high_days,
            'sensor_low_days': sensor_low_days,
            'steps_extreme_days': steps_extreme_days,
            'steps_anomaly_days': steps_low_active_anomaly,
        })

        # per-day detail (only for included participants — saves space)
        if included:
            for date in (cgm_days | stress_days_any | act_days):
                cgm_n = cgm.get(date, 0)
                srec  = stress_by_d.get(date, {})
                arec  = act_by_d.get(date, {})
                rows_day.append({
                    'person_id': pid,
                    'date': date,
                    'cgm_readings': cgm_n,
                    'cgm_hours_est': min(24, cgm_n / 12.0),
                    'cgm_covered_18h': cgm_n >= 216,
                    'garmin_hours_worn': srec.get('hours_worn', 0),
                    'garmin_covered_12h': srec.get('hours_worn', 0) >= THR_GAR_HOURS_DAY,
                    'stress_n_total': srec.get('n_total', 0),
                    'stress_n_usable': srec.get('n_usable', 0),
                    'stress_max_run_100_min': srec.get('max_run_100', 0),
                    'stress_max_run_0_min':   srec.get('max_run_0', 0),
                    'steps': arec.get('steps', 0.0),
                    'active_min': arec.get('active_min', 0.0),
                    'sedentary_min': arec.get('sedentary_min', 0.0),
                    'overlap_18h_12h': (cgm_n >= 216) and (srec.get('hours_worn', 0) >= THR_GAR_HOURS_DAY),
                })

    dpid = pd.DataFrame(rows_pid)
    ddat = pd.DataFrame(rows_day)
    PARTIC_OUT.parent.mkdir(exist_ok=True)
    PER_DAY.parent.mkdir(exist_ok=True)
    dpid.to_csv(PARTIC_OUT, index=False)
    ddat.to_csv(PER_DAY, index=False)
    print(f'\nSaved {PARTIC_OUT}  ({len(dpid)} rows)', flush=True)
    print(f'Saved {PER_DAY}  ({len(ddat)} rows)', flush=True)

    # Summary report
    n = len(dpid); ni = int(dpid['included_phase3'].sum())
    by5 = dpid.groupby('five_group_name')['included_phase3'].agg(['sum','count'])
    by3 = dpid.groupby('cluster_name')['included_phase3'].agg(['sum','count'])

    rsn = (dpid.loc[~dpid['included_phase3'], 'exclusion_reason']
              .str.split(';').explode().value_counts())

    lines = []
    lines.append('# Phase 3 QC summary')
    lines.append(f'**Cohort:** n={n} (analysis_table_n1306_clean.csv ∩ five_group_labels_n1306.csv)')
    lines.append('\n## Inclusion thresholds applied')
    lines.append(f'- garmin_total_days ≥ {THR_GARMIN_DAYS} (any stress or activity sample on that date)')
    lines.append(f'- overlap_days ≥ {THR_OVERLAP_DAYS} (CGM ≥{THR_CGM_HOURS_DAY}h AND Garmin ≥{THR_GAR_HOURS_DAY}h same date)')
    lines.append(f'- stress_days_usable ≥ {THR_STRESS_DAYS} (≥1 stress value in [0,100] on that day)')
    lines.append(f'- CGM "covered day" = ≥{THR_CGM_HOURS_DAY} hours (≥216 readings at 5-min cadence)')
    lines.append(f'- Garmin "covered day" = ≥{THR_GAR_HOURS_DAY} hours with any stress timestamp')
    lines.append('\n## Totals')
    lines.append(f'- Included: **{ni} / {n}** ({100*ni/n:.1f}%)')
    lines.append(f'- Excluded: {n - ni}')
    lines.append('\n## Exclusion reasons (multi-reason participants counted once per reason)')
    for r, c in rsn.items():
        lines.append(f'- {r}: {c}')

    lines.append('\n## Per 5-group eligibility')
    lines.append('\n| 5 Group | n in cohort | n included | % | flag |')
    lines.append('|---|---:|---:|---:|---|')
    for g in ['Spiker-Lean','Spiker-HighIR','Stable-Lean','Stable-HighIR','HP']:
        if g in by5.index:
            tot = int(by5.loc[g, 'count']); inc = int(by5.loc[g, 'sum'])
            flag = '<30' if inc < 30 else ''
            lines.append(f'| {g} | {tot} | {inc} | {100*inc/tot:.1f}% | {flag} |')

    lines.append('\n## Per K=3 phenotype eligibility')
    lines.append('\n| K=3 | n | n included | % |')
    lines.append('|---|---:|---:|---:|')
    for g in ['Spiker','Stable','Hypo-Prone']:
        if g in by3.index:
            tot = int(by3.loc[g, 'count']); inc = int(by3.loc[g, 'sum'])
            lines.append(f'| {g} | {tot} | {inc} | {100*inc/tot:.1f}% |')

    # Coverage distributions among included
    inc = dpid[dpid['included_phase3']]
    lines.append('\n## Coverage distributions (included participants only)')
    for col in ['garmin_total_days', 'overlap_days', 'stress_days_usable',
                'cgm_covered_days_18h', 'garmin_covered_days_12h',
                'n_stress_obs_usable', 'activity_days']:
        s = inc[col].describe()[['min','25%','50%','75%','max']]
        lines.append(f'- {col}: median={s["50%"]:.0f}, IQR=[{s["25%"]:.0f}, {s["75%"]:.0f}], range=[{s["min"]:.0f}, {s["max"]:.0f}]')

    # Sensor-error totals (all participants, day counts)
    lines.append('\n## Sensor-error participant-days (across full cohort)')
    lines.append(f'- stress stuck at 100 for ≥12h consecutive: {int(dpid["sensor_high_days"].sum())} participant-days, {int((dpid["sensor_high_days"]>0).sum())} participants affected')
    lines.append(f'- stress stuck at 0 for ≥12h consecutive: {int(dpid["sensor_low_days"].sum())} participant-days, {int((dpid["sensor_low_days"]>0).sum())} participants affected')
    lines.append(f'- steps >50,000/day: {int(dpid["steps_extreme_days"].sum())} participant-days, {int((dpid["steps_extreme_days"]>0).sum())} participants affected')
    lines.append(f'- steps <100 with >60 active min (likely device-on/no-counting): {int(dpid["steps_anomaly_days"].sum())} participant-days, {int((dpid["steps_anomaly_days"]>0).sum())} participants affected')
    lines.append('- HR-based outlier checks: **N/A — no heart-rate files ship in this dataset** (only sleep, activity, stress under wearable/garmin_vivosmart5)')

    lines.append('\n## Output files')
    lines.append(f'- `{PARTIC_OUT.relative_to(ROOT)}` — one row per cohort participant (n={n})')
    lines.append(f'- `{PER_DAY.relative_to(ROOT)}` — one row per included-participant × date (n={len(ddat)})')
    lines.append(f'- `{SUMMARY.relative_to(ROOT)}` — this file')

    SUMMARY.write_text('\n'.join(lines))
    print(f'Saved {SUMMARY}', flush=True)


if __name__ == '__main__':
    main()
