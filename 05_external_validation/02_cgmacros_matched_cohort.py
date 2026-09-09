"""External validation on an A1c-matched CGMacros subset.

    features  the same 14 clustering features
    window    day_night_diff on day 08-20 / night 00-06 local, matching the primary cohort
    cohort    HbA1c < 6.5 only, n = 31 of 45

CGMacros includes participants in the diabetes range and records no medication status, so
the subset is matched on HbA1c rather than on treatment.

in   CGMacros CGM and bio tables, processed/analysis_table_n1306_n14.csv
out  processed/cgmacros_n14_matched31.csv, logs/external_cgmacros_n14.md
"""
import glob, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

SEED = 42
B = os.environ.get("AIREADI_DATA_ROOT", "")
assert B, "set AIREADI_DATA_ROOT to the data root"
N = f'{B}/canonical_n14_rerun'
F14 = ['mean_glucose', 'glucose_sd', 'glucose_cv', 'mage', 'pct_above_140', 'pct_above_180',
       'pct_below_70', 'pct_below_54', 'n_lows_70', 'n_spikes_140',
       'avg_rise_rate', 'avg_fall_rate', 'day_night_diff', 'n_reactive_events']
log = []
def emit(m):
    print(m, flush=True); log.append(m)

# ---- day_night_diff on the AI-READI window, from raw (5-min resampled) ----
rows = []
for f in sorted(glob.glob(f'{B}/external_validation/cgmacros/extracted/CGMacros/CGMacros-*/CGMacros-*.csv')):
    pid = 'CGMacros-' + os.path.basename(f).split('-')[1].split('.')[0]
    d = pd.read_csv(f, usecols=lambda c: c in ['Timestamp', 'Dexcom GL', 'Libre GL'])
    gc = 'Dexcom GL' if 'Dexcom GL' in d and d['Dexcom GL'].notna().sum() > 50 else 'Libre GL'
    t = pd.to_datetime(d.Timestamp, errors='coerce')
    v = pd.to_numeric(d[gc], errors='coerce')
    s = pd.DataFrame({'t': t, 'v': v}).dropna().set_index('t')['v'].resample('5min').mean().dropna()
    h = s.index.hour
    day, night = s[(h >= 8) & (h < 20)], s[(h >= 0) & (h < 6)]
    rows.append({'participant_id': pid,
                 'dnd_820': (day.mean() - night.mean()) if len(day) > 10 and len(night) > 10 else np.nan})
cg = pd.read_csv(f'{B}/external_validation/results/cgmacros_features_labels.csv') \
       .merge(pd.DataFrame(rows), on='participant_id')
emit(f'day_night_diff: extractor day[6-24] mean {cg.day_night_diff.mean():+.2f} -> '
     f'AI-READI day[8-20] mean {cg.dnd_820.mean():+.2f} (r={np.corrcoef(cg.day_night_diff, cg.dnd_820)[0,1]:.3f})')
cg['day_night_diff'] = cg['dnd_820']

# ---- A1c and the matched cohort ----
bio = pd.read_csv(f'{B}/external_validation/cgmacros/_bio_extracted.csv') \
    if os.path.exists(f'{B}/external_validation/cgmacros/_bio_extracted.csv') else None
if bio is None:
    import zipfile
    z = zipfile.ZipFile(f'{B}/external_validation/cgmacros/CGMacros_dateshifted365.zip')
    bio = pd.read_csv(z.open('CGMacros/bio.csv'))
    bio.to_csv(f'{B}/external_validation/cgmacros/_bio_extracted.csv', index=False)
bio.columns = [c.strip() for c in bio.columns]
bio['pid'] = bio.subject.astype(int)
cg['pid'] = cg.participant_id.str.extract(r'(\d+)').astype(int)
cg = cg.merge(bio[['pid', 'A1c PDL (Lab)', 'BMI', 'Age']].rename(columns={'A1c PDL (Lab)': 'A1c'}), on='pid')
emit(f'CGMacros A1c: normo<5.7 {(cg.A1c<5.7).sum()} | preDM 5.7-6.4 {((cg.A1c>=5.7)&(cg.A1c<6.5)).sum()} '
     f'| diabetes>=6.5 {(cg.A1c>=6.5).sum()}  (treatment status NOT recorded)')
sub = cg[cg.A1c < 6.5].copy()
emit(f'MATCHED cohort A1c<6.5: n={len(sub)} of {len(cg)}')

# ---- de-novo K-selection on the 14 features ----
X = sub[F14]; X = X.fillna(X.median()); Xs = StandardScaler().fit_transform(X)
emit('\nde-novo K-selection (14 features, matched n=%d):' % len(sub))
emit('   K   silhouette      CH      DB   sizes')
for K in range(2, 7):
    l = KMeans(K, n_init=20, random_state=SEED).fit_predict(Xs)
    sz = ';'.join(str(v) for v in pd.Series(l).value_counts().sort_values(ascending=False))
    emit(f'   {K}   {silhouette_score(Xs,l):.4f}   {calinski_harabasz_score(Xs,l):6.1f}  '
         f'{davies_bouldin_score(Xs,l):.3f}   {sz}')

sub['k3'] = KMeans(3, n_init=20, random_state=SEED).fit_predict(Xs)
hp = sub.groupby('k3').n_lows_70.median().idxmax()
med = sub.groupby('k3').mean_glucose.median()
other = [k for k in sub.k3.unique() if k != hp]
hi = max(other, key=lambda k: med[k])
sub['group'] = np.where(sub.k3 == hp, 'HP-analogue',
                        np.where(sub.k3 == hi, 'higher-glucose', 'moderate'))
st = sub.groupby('group').agg(n=('pid', 'size'), med_lows70=('n_lows_70', 'median'),
                              pct_below_54=('pct_below_54', 'median'),
                              reactive=('n_reactive_events', 'median'),
                              mean_glucose=('mean_glucose', 'median'), mage=('mage', 'median'),
                              A1c=('A1c', 'median'))
emit('\nK=3 clusters:\n' + st.round(2).to_string())
hpm = sub[sub.k3 == hp]
emit(f'\nHP-analogue members: {sorted(hpm.pid)}  A1c={sorted(hpm.A1c)}  diabetic={int((hpm.A1c>=6.5).sum())}')

ai = pd.read_csv(f'{N}/processed/analysis_table_n1306_n14.csv')
aihp = ai[ai.hypo_k3_n14 == 2]
emit(f'\ncorrespondence vs AI-READI n14 HP (n={len(aihp)}):')
for c, l in [('n_lows_70', 'lows<70'), ('pct_below_54', '%<54'), ('n_reactive_events', 'reactive'),
             ('mean_glucose', 'mean glucose'), ('mage', 'MAGE')]:
    emit(f'   {l:14s} CGMac {hpm[c].median():8.2f}   AI-READI {aihp[c].median():8.2f}')
emit(f'   {"HbA1c":14s} CGMac {hpm.A1c.median():8.2f}   AI-READI {aihp.hba1c.median():8.2f}')

sub.to_csv(f'{N}/processed/cgmacros_n14_matched31.csv', index=False)
with open(f'{N}/logs/external_cgmacros_n14.md', 'w') as f:
    f.write('# External validation — CGMacros matched n=31, 14 features, day 08-20\n\n```\n'
            + '\n'.join(log) + '\n```\n')
emit(f'\nWROTE {N}/processed/cgmacros_n14_matched31.csv')
