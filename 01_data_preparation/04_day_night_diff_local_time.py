"""Local-time day/night glucose difference, and the K=3 refit that consumes it.

CGM timestamps are stored in UTC. The day (08-20) and night (00-06) windows that define
day_night_diff are local-clock concepts, so hours are converted per participant using the
clinical-site offset from participants.tsv (UAB -6, UW and UCSD -8, no DST) before the
windows are applied.

in   participants.tsv, cgm_all_readings_clean_final.csv,
     cgm_all_readings_clean_final_medicated.csv, processed/analysis_table_n1306_clean.csv
out  processed/day_night_diff_local_all.csv, labels + refit model artifacts
"""
import os, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

SEED = 42
np.random.seed(SEED)

BASE = os.environ.get("AIREADI_DATA_ROOT", "")
assert BASE, "set AIREADI_DATA_ROOT to the data root"
NEW  = f'{BASE}/pruned_n10_tzfix'
OLD_TABLE = f'{BASE}/pruned_n10_rerun/processed/analysis_table_n1306_pruned.csv'
CGM_UNMED = f'{BASE}/cgm_all_readings_clean_final.csv'
CGM_MED   = f'{BASE}/cgm_all_readings_clean_final_medicated.csv'
PART      = f'{BASE}/participants/participants.tsv'
CLEAN_AT  = f'{BASE}/processed/analysis_table_n1306_clean.csv'

for sub in ['labels', 'processed', 'models', 'logs']:
    os.makedirs(f'{NEW}/{sub}', exist_ok=True)

PRUNED = ['mean_glucose', 'pct_above_180', 'n_spikes_140', 'n_lows_70', 'pct_below_54',
          'mage', 'avg_rise_rate', 'avg_fall_rate', 'n_reactive_events', 'day_night_diff']
NAME = {0: 'Spiker', 1: 'Stable', 2: 'Hypo-Prone'}
SITE_TZ = {'UAB': 'cst', 'UW': 'pst', 'UCSD': 'pst'}
UTC_OFFSET = {'pst': -8, 'cst': -6}          # non-DST, matches scripts/63
DAY_LO, DAY_HI, NIGHT_LO, NIGHT_HI = 8, 20, 0, 6    # AI-READI window convention, unchanged

log = []
def emit(m):
    print(m, flush=True); log.append(m)


# ---------------------------------------------------------------- timezone map
part = pd.read_csv(PART, sep='\t', usecols=['person_id', 'clinical_site'])
part['tz'] = part['clinical_site'].map(SITE_TZ)
assert part['tz'].notna().all(), 'unmapped clinical_site'
TZ = dict(zip(part.person_id.astype(int), part.tz))

_chk = pd.read_csv(CLEAN_AT, usecols=['person_id', 'timezone'])
_chk['derived'] = _chk.person_id.astype(int).map(TZ)
_mis = int((_chk.derived != _chk.timezone.str.lower()).sum())
emit(f'timezone map: site-derived vs stored `timezone` mismatches = {_mis}/{len(_chk)} '
     f'(must be 0); coverage {part.tz.value_counts().to_dict()}')
assert _mis == 0, 'site->tz mapping does not reproduce the stored timezone column'


# ---------------------------------------------------------------- recompute day_night_diff (LOCAL)
def dnd_local(path, tag):
    """Per-person day_night_diff on LOCAL hours. Returns df[person_id, dnd_local, dnd_utc]."""
    cgm = pd.read_csv(path, usecols=['participant_id', 'start_datetime', 'glucose_value_mg_dL'])
    cgm['pid'] = cgm.participant_id.astype(int)
    t = pd.to_datetime(cgm.start_datetime, utc=True, format='ISO8601', errors='coerce')
    cgm['uh'] = t.dt.hour
    cgm['off'] = cgm.pid.map(TZ).map(UTC_OFFSET)
    n_miss = int(cgm['off'].isna().sum())
    cgm['off'] = cgm['off'].fillna(-8).astype(int)       # pst default, matches script 63
    cgm['lh'] = (cgm.uh + cgm.off) % 24
    emit(f'  [{tag}] {len(cgm):,} readings, {cgm.pid.nunique():,} participants, '
         f'{n_miss:,} readings with no tz (defaulted pst)')

    def _one(hcol):
        h = cgm[hcol].values
        day = (h >= DAY_LO) & (h < DAY_HI)
        night = (h >= NIGHT_LO) & (h < NIGHT_HI)
        A = cgm.loc[day].groupby('pid').glucose_value_mg_dL.agg(['mean', 'size'])
        B = cgm.loc[night].groupby('pid').glucose_value_mg_dL.agg(['mean', 'size'])
        j = A.join(B, lsuffix='_d', rsuffix='_n', how='inner')
        j = j[(j['size_d'] > 10) & (j['size_n'] > 10)]   # same guard as the original code
        return (j['mean_d'] - j['mean_n'])

    out = pd.concat([_one('lh').rename('dnd_local'), _one('uh').rename('dnd_utc')], axis=1)
    out.index.name = 'person_id'
    return out.reset_index()


emit('\n=== recomputing day_night_diff on LOCAL hours ===')
d_un = dnd_local(CGM_UNMED, 'unmedicated')
d_me = dnd_local(CGM_MED, 'medicated')
alld = pd.concat([d_un.assign(cohort='unmedicated'), d_me.assign(cohort='medicated')],
                 ignore_index=True)
alld.to_csv(f'{NEW}/processed/day_night_diff_local_all.csv', index=False)
emit(f'  wrote processed/day_night_diff_local_all.csv  (n={len(alld):,})')
_c = alld.dropna(subset=['dnd_local', 'dnd_utc'])
emit(f'  UTC  : mean {_c.dnd_utc.mean():+.2f}  median {_c.dnd_utc.median():+.2f}  '
     f'{100*(_c.dnd_utc<0).mean():.0f}% negative')
emit(f'  LOCAL: mean {_c.dnd_local.mean():+.2f}  median {_c.dnd_local.median():+.2f}  '
     f'{100*(_c.dnd_local<0).mean():.0f}% negative')
emit(f'  pearson r(UTC, LOCAL) = {np.corrcoef(_c.dnd_utc, _c.dnd_local)[0,1]:+.3f}')


# ---------------------------------------------------------------- corrected feature table
df = pd.read_csv(OLD_TABLE)
df['person_id'] = df.person_id.astype(int)
assert df.person_id.is_unique, 'duplicate person_id'
old_lab = df['hypo_k3_pruned'].values.copy()
emit(f'\nloaded old frozen table n={len(df)}  labels '
     f'{pd.Series(old_lab).value_counts().sort_index().to_dict()}')

df = df.merge(d_un[['person_id', 'dnd_local', 'dnd_utc']], on='person_id', how='left')
_r = np.corrcoef(df.day_night_diff.fillna(0), df.dnd_utc.fillna(0))[0, 1]
emit(f'provenance check: stored day_night_diff vs recomputed UTC r = {_r:.4f} (must be ~1.000)')
assert _r > 0.999, 'stored feature is not the UTC version — investigate before proceeding'

n_na = int(df.dnd_local.isna().sum())
df['day_night_diff_utc_OLD'] = df['day_night_diff']
df['day_night_diff'] = df['dnd_local'].fillna(df['dnd_local'].median())
emit(f'replaced day_night_diff with LOCAL version ({n_na} missing -> median-filled)')


# ---------------------------------------------------------------- refit
X = df[PRUNED].copy()
X = X.fillna(X.median())
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)
km = KMeans(n_clusters=3, n_init=20, random_state=SEED).fit(Xs)
raw = km.labels_

prof = pd.DataFrame({'raw': raw, 'n_lows_70': df.n_lows_70.values,
                     'n_spikes_140': df.n_spikes_140.values}).groupby('raw').median()
hp_c = int(prof.n_lows_70.idxmax())
sp_c = int(prof.drop(index=hp_c).n_spikes_140.idxmax())
st_c = int([c for c in [0, 1, 2] if c not in (hp_c, sp_c)][0])
remap = {sp_c: 0, st_c: 1, hp_c: 2}
new_lab = np.array([remap[c] for c in raw])
emit(f'\nlabel rule -> raw{sp_c}=Spiker, raw{st_c}=Stable, raw{hp_c}=Hypo-Prone')
emit('cluster medians used for the rule:\n' + prof.round(2).to_string())


# ---------------------------------------------------------------- validation
sil_new = silhouette_score(Xs, new_lab)
Xold = df[[c for c in PRUNED if c != 'day_night_diff'] + ['day_night_diff_utc_OLD']].copy()
Xold = Xold.fillna(Xold.median())
sil_old = silhouette_score(StandardScaler().fit_transform(Xold), old_lab)
ari = adjusted_rand_score(old_lab, new_lab)
sizes = pd.Series(new_lab).value_counts().sort_index().to_dict()
emit(f'\n=== VALIDATION ===')
emit(f'new sizes {sizes}  (Spiker/Stable/HP)')
emit(f'silhouette: new {sil_new:.4f}   old {sil_old:.4f}')
emit(f'ARI(new, old frozen) = {ari:.4f}')
ct = pd.crosstab(pd.Series(old_lab, name='OLD'), pd.Series(new_lab, name='NEW'))
ct.index = [NAME[i] for i in ct.index]; ct.columns = [NAME[i] for i in ct.columns]
emit('cross-tab:\n' + ct.to_string())
moved = int((old_lab != new_lab).sum())
emit(f'participants changing group: {moved}/{len(df)} ({100*moved/len(df):.1f}%)')
hp_keep = int(((old_lab == 2) & (new_lab == 2)).sum())
emit(f'HP retained: {hp_keep}/{int((old_lab==2).sum())}')

# permutation null: shuffle EACH feature column independently (destroys the joint structure,
# preserves every marginal), refit, and score the shuffled data with its own labels.
rng = np.random.RandomState(SEED)
perm = []
for _ in range(100):
    Xp = np.column_stack([rng.permutation(Xs[:, j]) for j in range(Xs.shape[1])])
    lp = KMeans(3, n_init=5, random_state=SEED).fit_predict(Xp)
    perm.append(silhouette_score(Xp, lp))
perm = np.array(perm)
p_emp = (perm >= sil_new).sum() / (len(perm) + 1)
emit(f'permutation null silhouette (100x, per-feature shuffle): mean {perm.mean():.4f}  '
     f'max {perm.max():.4f}  -> observed {sil_new:.4f}, empirical p <= {max(p_emp, 1/(len(perm)+1)):.3f}')


# ---------------------------------------------------------------- outputs
lab_out = pd.DataFrame({'person_id': df.person_id, 'hypo_k3_tzfix': new_lab,
                        'hypo_k3_pruned_old': old_lab, 'raw_cluster': raw,
                        'day_night_diff_local': df.day_night_diff,
                        'day_night_diff_utc_OLD': df.day_night_diff_utc_OLD})
lab_out.to_csv(f'{NEW}/labels/hypo_clustering_tzfix_n1306.csv', index=False)
df['hypo_k3_tzfix'] = new_lab
df.drop(columns=['dnd_local', 'dnd_utc']).to_csv(
    f'{NEW}/processed/analysis_table_n1306_tzfix.csv', index=False)
with open(f'{NEW}/models/kmeans_tzfix_n10.pkl', 'wb') as f: pickle.dump(km, f)
with open(f'{NEW}/models/scaler_tzfix_n10.pkl', 'wb') as f: pickle.dump(scaler, f)
with open(f'{NEW}/logs/refit_tzfix_log.md', 'w') as f:
    f.write('# Local-time day/night difference — K=3 refit\n\n')
    f.write('```\n' + '\n'.join(log) + '\n```\n')
emit('\nWROTE labels/, processed/, models/, logs/ under pruned_n10_tzfix/')
