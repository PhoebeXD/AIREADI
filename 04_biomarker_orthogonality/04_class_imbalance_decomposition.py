#!/usr/bin/env python3
"""Is low minority-class recall a rarity artifact or absent signal?

AUC is rarity-invariant, so high AUC with low recall indicates a decision-threshold effect
while low AUC indicates no signal. Each minority class is run unweighted and class-balanced
across the full panel, HbA1c alone, and a CGM reference; balanced-weight precision shows
whether rebalancing buys assignment or only mass-flagging.

in   processed/analysis_table_n1306_n14.csv
out  imbalance_decomposition.csv
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, roc_auc_score, precision_score

SEED = 42; np.random.seed(SEED)
BASE = os.environ.get("AIREADI_DATA_ROOT", "")
assert BASE, "set AIREADI_DATA_ROOT to the data root"
P = f'{BASE}/canonical_n14_rerun'; OUT = f'{P}/subtype_classification'
os.makedirs(OUT, exist_ok=True)
DF = pd.read_csv(f'{P}/processed/analysis_table_n1306_n14.csv')
NAME = {0: 'Spiker', 1: 'Stable', 2: 'HP'}; DF['grp'] = DF['hypo_k3_n14'].map(NAME)
HYPER = ['mean_glucose', 'pct_above_180', 'n_spikes_140']
z = StandardScaler().fit_transform(DF[HYPER].values); DF['hs'] = z.mean(1)
n_ext = max(int((DF.grp == 'HP').sum()), int(round(len(DF) * 0.03)))
ext = set(DF.sort_values('hs', ascending=False).head(n_ext)['person_id'])
DF['is_ext'] = DF['person_id'].isin(ext) & (DF.grp != 'HP')
PANEL = ['hba1c', 'fasting_glucose', 'fasting_insulin', 'homa_ir_corrected', 'triglycerides', 'hdl']

def run(sub, y, cols, cw):
    cc = sub[cols].notna().all(1).values; X = sub.loc[cc, cols].values.astype(float); yy = y[cc]
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pred = np.empty(len(yy), int); proba = np.zeros(len(yy))
    for tr, te in skf.split(X, yy):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=5000, class_weight=cw, random_state=SEED).fit(sc.transform(X[tr]), yy[tr])
        pred[te] = clf.predict(sc.transform(X[te])); proba[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return (roc_auc_score(yy, proba), recall_score(yy, pred, pos_label=1, zero_division=0),
            precision_score(yy, pred, pos_label=1, zero_division=0))

CASES = [
    ('extreme-Spiker vs mild-Spiker', DF[DF.grp == 'Spiker'].copy(),
     lambda s: s['is_ext'].astype(int).values, ['mean_glucose']),
    ('Spiker vs Stable',              DF[DF.grp != 'HP'].copy(),
     lambda s: (s.grp == 'Spiker').astype(int).values, ['mean_glucose']),
    ('HP vs rest',                    DF.copy(),
     lambda s: (s.grp == 'HP').astype(int).values, ['n_lows_70']),
]
rows = []
for title, sub, labfn, cgm in CASES:
    y = labfn(sub)
    print(f"### {title}  (minority n={int(y.sum())}, prevalence {y.mean():.1%})")
    for cols, name in [(PANEL, 'Full panel'), (['hba1c'], 'HbA1c-only'), (cgm, f'{cgm[0]} (CGM ref)')]:
        au, ru, pu = run(sub, y, cols, None); ab, rb, pb = run(sub, y, cols, 'balanced')
        print(f"  {name:22s} AUC={au:.3f} | unwtd rec={ru:.2f} prec={pu:.2f} -> "
              f"balanced rec={rb:.2f} prec={pb:.2f}")
        rows.append(dict(case=title, minority_n=int(y.sum()), prevalence=float(y.mean()),
                         feature_set=name, auc=au, recall_unwtd=ru, prec_unwtd=pu,
                         recall_balanced=rb, prec_balanced=pb))
    print()
pd.DataFrame(rows).to_csv(f'{OUT}/imbalance_decomposition.csv', index=False)
print(f"WROTE {OUT}/imbalance_decomposition.csv\nDONE")
