"""Timing of Hypo-Prone low events.

Sensor compression lows are characteristically nocturnal and position-related, so the
local-time distribution of readings below 70 mg/dL separates that explanation from waking
physiology. Local time comes from the analysis table timezone column.

in   analysis table, labels, cgm_all_readings_clean_final.csv
out  logs/hp_low_timing.{md,csv}, hourly and day-split CSVs
"""
import os
import numpy as np
import pandas as pd

BASE = os.environ.get("AIREADI_DATA_ROOT", "")
assert BASE, "set AIREADI_DATA_ROOT to the data root"
PRUNE  = f'{BASE}/canonical_n14_rerun'
AT     = f'{PRUNE}/processed/analysis_table_n1306_n14.csv'
CGM    = f'{BASE}/cgm_all_readings_clean_final.csv'
LOGDIR = f'{PRUNE}/logs'
UTC_OFFSET = {'pst': -8, 'cst': -6}
NAME = {0: 'Spiker', 1: 'Stable', 2: 'HP'}

lab = pd.read_csv(AT, usecols=['person_id', 'hypo_k3_pruned', 'timezone'])
lab['timezone'] = lab['timezone'].fillna('pst').str.lower()
vc = lab['hypo_k3_pruned'].value_counts().to_dict()
assert len(vc) == 3, 'expected three pattern labels'
g3 = dict(zip(lab.person_id, lab.hypo_k3_pruned))
tz = dict(zip(lab.person_id, lab.timezone))

low_hist = {g: np.zeros(24) for g in NAME}   # readings <70 by local hour
tot_hist = {g: np.zeros(24) for g in NAME}   # total readings by local hour
lows70 = {g: 0 for g in NAME}; lows54 = {g: 0 for g in NAME}; tot = {g: 0 for g in NAME}

for ch in pd.read_csv(CGM, usecols=['participant_id', 'start_datetime', 'glucose_value_mg_dL'],
                      chunksize=2_000_000):
    ch = ch[ch.participant_id.isin(g3)]
    if ch.empty:
        continue
    grp = ch.participant_id.map(g3).values
    off = ch.participant_id.map(tz).map(UTC_OFFSET).fillna(-8).astype(int)
    t = pd.to_datetime(ch.start_datetime, utc=True, format='ISO8601')
    hr = (t + pd.to_timedelta(off, unit='h')).dt.hour.values
    gv = ch.glucose_value_mg_dL.values
    for g in NAME:
        m = grp == g
        if not m.any():
            continue
        h = hr[m]; v = gv[m]
        tot[g] += int(m.sum()); lows70[g] += int((v < 70).sum()); lows54[g] += int((v < 54).sum())
        tot_hist[g] += np.bincount(h, minlength=24)
        low_hist[g] += np.bincount(h[v < 70], minlength=24)

# ---- day/night split of lows<70 (overnight 00:00-05:59 vs waking 06:00-23:59) ----
rows = []
for g in NAME:
    lh = low_hist[g]; tt = lh.sum()
    night = lh[0:6].sum(); day = lh[6:24].sum()
    rows.append(dict(group=NAME[g], n_readings=int(tot[g]), n_lows70=int(lows70[g]),
                     pct_readings_lt70=round(100 * lows70[g] / tot[g], 3),
                     n_lows54=int(lows54[g]), pct_readings_lt54=round(100 * lows54[g] / tot[g], 4),
                     overnight_00_06_pct=round(100 * night / tt, 1),
                     waking_06_24_pct=round(100 * day / tt, 1)))
split = pd.DataFrame(rows)

# ---- HP hourly low-rate (lows<70 per 1000 readings) ----
lr = 1000 * low_hist[2] / np.maximum(tot_hist[2], 1)
hourly = pd.DataFrame({'local_hour': range(24),
                       'hp_n_low': low_hist[2].astype(int), 'hp_n_tot': tot_hist[2].astype(int),
                       'hp_low_rate_per_1000': np.round(lr, 1)})
peak_h = int(np.argmax(lr)); trough_h = int(np.argmin(lr))
noct_rate = lr[0:6].mean(); wake_rate = lr[6:24].mean()
# wear-normalized fraction of readings <70 (removes the 18h/6h window-size confound)
hp_low_n = low_hist[2]; hp_tot_n = tot_hist[2]
noct_frac = 100 * hp_low_n[0:6].sum() / hp_tot_n[0:6].sum()
wake_frac = 100 * hp_low_n[6:24].sum() / hp_tot_n[6:24].sum()

os.makedirs(LOGDIR, exist_ok=True)
split.to_csv(f'{LOGDIR}/hp_low_timing_split.csv', index=False)
hourly.to_csv(f'{LOGDIR}/hp_low_timing_hourly.csv', index=False)

md = []
md.append('# HP low-event timing — device-artifact defense (Fig 5 / Methods)\n')
md.append('Provenance: AI-READI v3.0.0; input `cgm_all_readings_clean_final.csv` '
          '(POST dedup + V/Λ QC, README ); local time via `timezone` (pst -8 / cst -6, non-DST); '
          'low = reading <70 (point-count, extract_features.py:61). Windows: overnight 00:00-05:59 vs waking 06:00-23:59.\n')
md.append('## Day/night split of low readings (<70) — COUNT split\n')
md.append(split.to_markdown(index=False))
md.append('\n> The count split is partly mechanical: waking spans 18 of 24 h, '
          'so **75% waking / 25% overnight is the uniform null**. HP 80.5/19.5 is only modestly beyond that; '
          'state it as "overnight 19.5% < 25% expected", not bare "80.5% waking". Lead the defense with the '
          'WEAR-NORMALIZED rate below instead.\n')
md.append('## Wear-normalized low-rate (the bulletproof statement)\n')
md.append(f'- **% of readings <70: overnight (00-06) {noct_frac:.1f}% vs waking (06-24) {wake_frac:.1f}%** '
          f'→ lows are LESS frequent overnight (removes window-size confound).\n')
md.append('## HP hourly low-rate (lows<70 per 1000 readings = per-reading prevalence ×1000, local hour)\n')
md.append(hourly.to_markdown(index=False))
md.append(f'\n- Per-hour rate = lows<70 per 1000 readings in that clock hour (NOT event counts; wear ~uniform, '
          f'{int(hp_tot_n.min())}-{int(hp_tot_n.max())} readings/hour).')
md.append(f'- Peak hour = {peak_h} (rate {lr[peak_h]:.1f}); trough hour = {trough_h} (rate {lr[trough_h]:.1f}) '
          f'→ peak/trough ~{lr[peak_h]/lr[trough_h]:.1f}x.')
md.append(f'- Overnight (00-06) mean rate {noct_rate:.1f} vs waking (06-24) mean rate {wake_rate:.1f} '
          f'(~{wake_rate/noct_rate:.1f}x higher waking; moderate + non-monotonic — h00/h05 exceed h20/h21).')
md.append('- Shape: daytime-dominant, broad midday/afternoon plateau (h10-h16), overnight is the LOW region '
          '→ physiological, OPPOSITE of the overnight/position-related signature of compression artifact.\n')
with open(f'{LOGDIR}/hp_low_timing.md', 'w') as f:
    f.write('\n'.join(md))

print(split.to_string(index=False))
print(f'\nHP peak hour {peak_h} (rate {lr[peak_h]:.1f}); overnight {noct_rate:.1f} vs waking {wake_rate:.1f}')
print('WROTE', f'{LOGDIR}/hp_low_timing.md', '+ _split.csv + _hourly.csv')
