"""K=3 clustering of the untreated cohort on 14 CGM features.

Feature set: the 15 candidate CGM features less n_lows_54, which is the same construct as
pct_below_54 (count vs percent of readings below 54 mg/dL), a near-perfect duplicate. The
feature set was selected post hoc; see Methods.

Pipeline
    median imputation -> StandardScaler -> KMeans(k=3, n_init=20, random_state=42)

Cluster naming rule
    Hypo-Prone = highest median n_lows_70
    Spiker     = highest median n_spikes_140 among the remaining two
    Stable     = the last cluster

day_night_diff is the local-time feature from 01_data_preparation/04.

in   processed/analysis_table_n1306_clean.csv, processed/day_night_diff_local_all.csv
out  labels, analysis table, KMeans and scaler artifacts, fit log
"""
import os, json, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score)

SEED = 42
np.random.seed(SEED)
BASE = os.environ.get("AIREADI_DATA_ROOT", "")
assert BASE, "set AIREADI_DATA_ROOT to the data root"
NEW  = f'{BASE}/canonical_n14_rerun'
for d in ['labels', 'processed', 'models', 'logs', 'plots']:
    os.makedirs(f'{NEW}/{d}', exist_ok=True)

F14 = ['mean_glucose', 'glucose_sd', 'glucose_cv', 'mage', 'pct_above_140', 'pct_above_180',
       'pct_below_70', 'pct_below_54', 'n_lows_70', 'n_spikes_140',
       'avg_rise_rate', 'avg_fall_rate', 'day_night_diff', 'n_reactive_events']
DROPPED = 'n_lows_54'
NAME = {0: 'Spiker', 1: 'Stable', 2: 'Hypo-Prone'}

log = []
def emit(m):
    print(m, flush=True); log.append(m)


# ---------------------------------------------------------------- load + tz fix
at = pd.read_csv(f'{BASE}/processed/analysis_table_n1306_clean.csv')
at['person_id'] = at.person_id.astype(int)
assert at.person_id.is_unique, 'duplicate person_id in the analysis table'

fix = pd.read_csv(f'{BASE}/local_time_refit/processed/day_night_diff_local_all.csv')
dnd = dict(zip(fix.person_id.astype(int), fix.dnd_local))
at['day_night_diff_utc_OLD'] = at['day_night_diff']
newv = at.person_id.map(dnd)
n_miss = int(newv.isna().sum())
at['day_night_diff'] = newv.fillna(newv.median())
emit(f'TZ FIX: day_night_diff UTC mean {at.day_night_diff_utc_OLD.mean():+.2f} '
     f'-> LOCAL mean {at.day_night_diff.mean():+.2f}  ({n_miss} unmatched -> median)')
emit(f'FEATURES: {len(F14)} = canonical-15 minus `{DROPPED}` '
     f'(near-duplicate of pct_below_54)')

# collinearity report for the record
C = at[F14].fillna(at[F14].median()).corr().abs()
np.fill_diagonal(C.values, 0)
worst = C.stack().sort_values(ascending=False).head(5)
emit('top remaining collinear pairs:')
for (a, b), r in worst.items():
    if a < b: emit(f'    {r:.3f}  {a} ~ {b}')

# ---------------------------------------------------------------- fit
X = at[F14].copy()
X = X.fillna(X.median())
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)
km = KMeans(n_clusters=3, n_init=20, random_state=SEED).fit(Xs)
raw = km.labels_

prof = pd.DataFrame({'raw': raw, 'n_lows_70': at.n_lows_70.values,
                     'n_spikes_140': at.n_spikes_140.values}).groupby('raw').median()
hp = int(prof.n_lows_70.idxmax())
sp = int(prof.drop(index=hp).n_spikes_140.idxmax())
st = [c for c in (0, 1, 2) if c not in (hp, sp)][0]
remap = {sp: 0, st: 1, hp: 2}
lab = np.array([remap[r] for r in raw])
at['hypo_k3_n14'] = lab
emit(f'\nlabel rule -> raw{sp}=Spiker, raw{st}=Stable, raw{hp}=Hypo-Prone')
emit('cluster medians used by the rule:\n' + prof.round(2).to_string())
sizes = pd.Series(lab).value_counts().sort_index().to_dict()
emit(f'\nSIZES  Spiker {sizes[0]} / Stable {sizes[1]} / Hypo-Prone {sizes[2]}')

# ---------------------------------------------------------------- validation
sil = silhouette_score(Xs, lab)
emit(f'\n=== VALIDATION ===')
emit(f'silhouette {sil:.4f} | Calinski-Harabasz {calinski_harabasz_score(Xs, lab):.1f} '
     f'| Davies-Bouldin {davies_bouldin_score(Xs, lab):.3f}')

emit('\nK-selection sweep (same pipeline, K=2..6):')
emit('   K   silhouette      CH      DB   sizes')
for K in range(2, 7):
    l = KMeans(K, n_init=20, random_state=SEED).fit_predict(Xs)
    sz = ';'.join(str(v) for v in pd.Series(l).value_counts().sort_values(ascending=False))
    emit(f'   {K}   {silhouette_score(Xs,l):.4f}   {calinski_harabasz_score(Xs,l):7.1f} '
         f' {davies_bouldin_score(Xs,l):.3f}   {sz}')

# reference label sets
ref = {}
c15 = pd.read_csv(f'{BASE}/processed/analysis_table_n1306_clean.csv',
                  usecols=['person_id', 'hypo_k3'])
ref['canonical-15 (UTC)'] = at.person_id.map(dict(zip(c15.person_id.astype(int), c15.hypo_k3))).values
p10 = pd.read_csv(f'{BASE}/pruned_n10_rerun/labels/hypo_clustering_pruned_n1306.csv')
ref['pruned-10 (tz-fixed)'] = at.person_id.map(dict(zip(p10.person_id.astype(int),
                                                        p10.hypo_k3_pruned))).values
emit('')
for k, v in ref.items():
    emit(f'ARI vs {k:24s} = {adjusted_rand_score(v, lab):.4f}')

# seed stability
aris = []
for s in range(20):
    l2 = KMeans(3, n_init=20, random_state=s).fit_predict(Xs)
    aris.append(adjusted_rand_score(lab, l2))
emit(f'seed stability (20 seeds): ARI mean {np.mean(aris):.4f} ± {np.std(aris):.4f} '
     f'[{np.min(aris):.4f}, {np.max(aris):.4f}]')

# Two shuffle-based nulls were briefly used and BOTH are inappropriate for this question:
#   A  label shuffle          -> near-tautological; silhouette of random labels is ~0 by
#                                construction (null mean -0.035). Cannot fail.
#   B  per-feature shuffle    -> assumes the 14 CGM features are mutually INDEPENDENT, which is
#                                biologically impossible (mean_glucose~pct_above_180 are linked;
#                                n_lows_70 and pct_below_70 are near-duplicates). Rejecting it shows only that
#                                glucose features are correlated.
# Both reject for a single correlated UNIMODAL cloud, i.e. they cannot distinguish clusters from a
# continuum — and the dip test here says p=1.000 (no separate clusters). Test B was also NEVER run
# on the n10 branch, so running it here would create a branch asymmetry.
# The separation test is the null comparison in 02_clustering/03_gaussian_copula_null.py
# + 42b_null_copula.py): a Gaussian-COPULA null preserving BOTH the covariance and the empirical
# marginals. On pruned-10 the observed silhouette sat at the MEDIAN of that null (p=0.476, z=+0.01)

# ---------------------------------------------------------------- outputs
pd.DataFrame({'person_id': at.person_id, 'hypo_k3_n14': lab,
              'hypo_k3_canonical': ref['canonical-15 (UTC)'], 'raw_cluster': raw}
             ).to_csv(f'{NEW}/labels/hypo_clustering_n14_n1306.csv', index=False)
out = at.copy()          # already carries hypo_k3_n14 from the labelling step above
out.to_csv(f'{NEW}/processed/analysis_table_n1306_n14.csv', index=False)
pickle.dump(km, open(f'{NEW}/models/kmeans_n14.pkl', 'wb'))
pickle.dump(scaler, open(f'{NEW}/models/scaler_n14.pkl', 'wb'))
json.dump({'features': F14, 'dropped_from_canonical15': DROPPED, 'seed': SEED,
           'raw_to_canonical': {str(k): int(v) for k, v in remap.items()},
           'sizes': {NAME[i]: int((lab == i).sum()) for i in (0, 1, 2)},
           'silhouette': round(float(sil), 4),
           'tz_fixed_day_night_diff': True,
           'feature_set_provenance': 'selected post hoc; see Methods'},
          open(f'{NEW}/models/meta.json', 'w'), indent=1)
with open(f'{NEW}/logs/refit_n14_log.md', 'w') as f:
    f.write('# K=3 fit on 14 CGM features\n\n```\n' + '\n'.join(log) + '\n```\n')
emit(f'\nWROTE {NEW}/ (labels, processed, models, logs)')
