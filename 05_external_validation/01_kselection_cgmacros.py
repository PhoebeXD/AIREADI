"""K-selection sweep in an independent CGM cohort (CGMacros, n = 45).

Mirrors the primary K-selection sweep: z-scored CGM features, KMeans(n_init=20,
random_state=42) for K = 2..8, scored by silhouette, Calinski-Harabasz and Davies-Bouldin.
The question is whether the low-K pattern reproduces outside the primary cohort. The cohort
is small and the Hypo-Prone analogue is near-absent, so this does not test recovery of the
rare cluster.

in   external CGM feature table
out  K-selection sweep CSV
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

_root = os.environ.get("AIREADI_DATA_ROOT", "")
assert _root, "set AIREADI_DATA_ROOT to the data root"
BASE = Path(_root)
FEAT = BASE / 'external_validation' / 'results' / 'cgmacros_features_labels.csv'
OUT  = BASE / 'external_validation' / 'results' / 'kselection_cgmacros.csv'
SEED = 42

# Exact canonical 15-feature order
FEATURES = ['n_reactive_events', 'n_spikes_140', 'n_lows_70', 'n_lows_54',
            'pct_below_70', 'pct_below_54', 'pct_above_140', 'pct_above_180',
            'avg_rise_rate', 'avg_fall_rate', 'mage', 'day_night_diff',
            'mean_glucose', 'glucose_sd', 'glucose_cv']

df = pd.read_csv(FEAT)
print(f'CGMacros participants: {len(df)}')
assert df[FEATURES].isna().sum().sum() == 0, 'unexpected NaNs in features'

X = StandardScaler().fit_transform(df[FEATURES].values)

rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
    lab = km.labels_
    sizes = np.bincount(lab)
    # per-cluster signature on raw (unscaled) clinical features
    sig = []
    for c in np.argsort(-sizes):
        m = df[lab == c]
        sig.append({'c': int(c), 'n': int(sizes[c]),
                    'med_lows': float(m['n_lows_70'].median()),
                    'med_spikes': float(m['n_spikes_140'].median()),
                    'med_mean_glu': float(m['mean_glucose'].median())})
    rows.append(dict(
        K=k,
        silhouette=silhouette_score(X, lab),
        calinski_harabasz=calinski_harabasz_score(X, lab),
        davies_bouldin=davies_bouldin_score(X, lab),
        inertia=float(km.inertia_),
        sizes=';'.join(map(str, sorted(sizes, reverse=True))),
        n_small_lt5=int((sizes < 5).sum()),
        signatures=str(sig),
    ))

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)
print(f'Saved {OUT.relative_to(BASE)}\n')

# Report table
print(f'{"K":>2} {"silhouette":>11} {"CalHar":>8} {"DaviesB":>8} {"sizes":<22} {"tiny<5":>6}')
for _, r in res.iterrows():
    print(f'{int(r.K):>2} {r.silhouette:>11.4f} {r.calinski_harabasz:>8.1f} '
          f'{r.davies_bouldin:>8.4f} {r.sizes:<22} {int(r.n_small_lt5):>6}')

best_sil = res.loc[res.silhouette.idxmax(), 'K']
best_ch  = res.loc[res.calinski_harabasz.idxmax(), 'K']
best_db  = res.loc[res.davies_bouldin.idxmin(), 'K']
print(f'\nArgmax silhouette: K={int(best_sil)}   Argmax Calinski-Harabasz: K={int(best_ch)}   '
      f'Argmin Davies-Bouldin: K={int(best_db)}')
k2 = res[res.K == 2].iloc[0]; k3 = res[res.K == 3].iloc[0]
print(f'\nK=2 vs K=3 head-to-head:')
print(f'  silhouette       K2={k2.silhouette:.4f}  K3={k3.silhouette:.4f}  Δ(K3−K2)={k3.silhouette-k2.silhouette:+.4f}')
print(f'  Calinski-Harabasz K2={k2.calinski_harabasz:.1f}  K3={k3.calinski_harabasz:.1f}')
print(f'  Davies-Bouldin   K2={k2.davies_bouldin:.4f}  K3={k3.davies_bouldin:.4f} (lower better)')
