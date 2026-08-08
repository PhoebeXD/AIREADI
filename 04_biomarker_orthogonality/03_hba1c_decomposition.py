#!/usr/bin/env python3
"""Is the biomarker-panel signal just HbA1c?

For every subgroup test, compares HbA1c alone against the full 6-marker panel, with mean
glucose as a CGM reference. Same within-fold-scaled pipeline as the main classification
script; balanced accuracy (10x5 CV), ROC-AUC and bootstrap CI.

in   processed/analysis_table_n1306_n14.csv
out  hba1c_decomposition.csv
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score

SEED = 42; np.random.seed(SEED)
BASE = os.environ.get("AIREADI_DATA_ROOT", "")
assert BASE, "set AIREADI_DATA_ROOT to the data root"
PRUNED = f'{BASE}/canonical_n14_rerun'; OUT = f'{PRUNED}/subtype_classification'
os.makedirs(OUT, exist_ok=True)

DF = pd.read_csv(f'{PRUNED}/processed/analysis_table_n1306_n14.csv')
NAME = {0: 'Spiker', 1: 'Stable', 2: 'HP'}; DF['grp'] = DF['hypo_k3_pruned'].map(NAME)
HYPER = ['mean_glucose', 'pct_above_180', 'n_spikes_140']
z = StandardScaler().fit_transform(DF[HYPER].values); DF['hyper_sev'] = z.mean(axis=1)
n_ext = max(int((DF.grp == 'HP').sum()), int(round(len(DF) * 0.03)))
ext_ids = set(DF.sort_values('hyper_sev', ascending=False).head(n_ext)['person_id'])
DF['is_extreme'] = DF['person_id'].isin(ext_ids) & (DF.grp != 'HP')
DF['grp4'] = np.where(DF.grp == 'HP', 'HP', np.where(DF.grp == 'Stable', 'Stable',
             np.where(DF['is_extreme'], 'extreme-Spiker', 'mild-Spiker')))

PANEL = ['hba1c', 'fasting_glucose', 'fasting_insulin', 'homa_ir_corrected', 'triglycerides', 'hdl']
SETS = {'HbA1c-only': ['hba1c'], 'Full panel': PANEL, 'mean_glucose (CGM ref)': ['mean_glucose']}

TESTS = [
    ('T1 Spiker vs Stable',          DF.grp != 'HP',       lambda s: (s.grp == 'Spiker').astype(int), ['Stable', 'Spiker'], 0.5),
    ('T2 mild vs extreme Spiker',    DF.grp == 'Spiker',   lambda s: s['is_extreme'].astype(int),     ['mild', 'extreme'], 0.5),
    ('T3 HP/Spiker/Stable',          pd.Series(True, DF.index), lambda s: s['grp'].map({'Stable':0,'Spiker':1,'HP':2}), ['Stable','Spiker','HP'], 1/3),
    ('T4 HP/Stable/mild/extreme',    pd.Series(True, DF.index), lambda s: s['grp4'].map({'Stable':0,'mild-Spiker':1,'extreme-Spiker':2,'HP':3}), ['Stable','mild','extreme','HP'], 0.25),
]

def oof(X, y, seed, nc):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = np.empty(len(y), int); proba = np.zeros((len(y), nc))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=5000, random_state=SEED).fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
        p = clf.predict_proba(sc.transform(X[te]))
        for j, c in enumerate(clf.classes_): proba[te, c] = p[:, j]
    return pred, proba

rows = []
print(f"{'test':28s} {'set':24s} {'n':>5s} {'bacc':>6s} {'AUC':>6s}  recalls")
for tname, mask, labfn, cls, base in TESTS:
    sub = DF[mask].copy(); nc = len(cls)
    for sname, cols in SETS.items():
        cc = sub[cols].notna().all(axis=1).values
        X = sub.loc[cc, cols].values.astype(float); y = labfn(sub).values[cc].astype(int)
        baccs = []
        for rep in range(10):
            p, _ = oof(X, y, SEED + rep, nc); baccs.append(balanced_accuracy_score(y, p))
        p0, pr0 = oof(X, y, SEED, nc)
        auc = roc_auc_score(y, pr0[:, 1]) if nc == 2 else roc_auc_score(y, pr0, multi_class='ovr', average='macro')
        rec = recall_score(y, p0, labels=list(range(nc)), average=None, zero_division=0)
        rng = np.random.default_rng(SEED)
        boot = [balanced_accuracy_score(y[i], p0[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
        ci = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))
        recstr = ' '.join(f'{c}:{r:.2f}' for c, r in zip(cls, rec))
        print(f"{tname:28s} {sname:24s} {len(y):5d} {np.mean(baccs):6.3f} {auc:6.3f}  {recstr}")
        rows.append(dict(test=tname, feature_set=sname, n=len(y), baseline=base,
                         bacc=float(np.mean(baccs)), ci_lo=float(ci[0]), ci_hi=float(ci[1]),
                         auc=float(auc), recalls=recstr))
    print()

pd.DataFrame(rows).to_csv(f'{OUT}/hba1c_decomposition.csv', index=False)
m = DF[['hba1c', 'mean_glucose']].dropna()
print(f"hba1c ~ mean_glucose Pearson r = {np.corrcoef(m.hba1c, m.mean_glucose)[0,1]:.3f} (n={len(m)})")
print(f"\nWROTE {OUT}/hba1c_decomposition.csv\nDONE")
