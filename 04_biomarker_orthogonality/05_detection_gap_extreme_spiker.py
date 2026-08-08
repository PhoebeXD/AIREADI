#!/usr/bin/env python3
"""Detection gap at the other extreme of the continuum.

Applies the Hypo-Prone detection design to extreme Spikers: one model, only the feature set
swapped, recall of the rare class from CGM features versus routine labs. extreme-Spiker uses
the same rule as the subtype analysis and is matched to Hypo-Prone in size and prevalence,
so the two rare groups are directly comparable.

in   processed/analysis_table_n1306_n14.csv
out  logs/detection_gap_extreme_spiker.md, plot
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, roc_auc_score

SEED = 42
np.random.seed(SEED)
BASE = os.environ.get("AIREADI_DATA_ROOT", "")
assert BASE, "set AIREADI_DATA_ROOT to the data root"
BR = f'{BASE}/canonical_n14_rerun'
OUT, LOGD = f'{BR}/plots', f'{BR}/logs'
os.makedirs(OUT, exist_ok=True); os.makedirs(LOGD, exist_ok=True)

CGM_C, BIO_C0, BIO_C = '#1B9E77', '#5B54A0', '#7570B3'
EXT_C = '#D55E00'          # Spiker orange, matching the pattern palette in Fig 5

CGM14 = json.load(open(f'{BR}/models/meta.json'))['features']
assert len(CGM14) == 14
BIO_NF = ['hba1c', 'hdl', 'total_cholesterol', 'crp', 'alt', 'ast', 'creatinine',
          'urine_albumin', 'wbc']

df = pd.read_csv(f'{BR}/processed/analysis_table_n1306_n14.csv')
vc = df['hypo_k3_pruned'].value_counts().to_dict()
assert len(vc) == 3, 'expected three pattern labels'
NAME = {0: 'Spiker', 1: 'Stable', 2: 'HP'}
df['grp'] = df['hypo_k3_pruned'].map(NAME)

# ---- extreme-Spiker definition (CGM-only axis, rarity-matched to HP)
HYPER = ['mean_glucose', 'pct_above_180', 'n_spikes_140']
df['hyper_sev'] = StandardScaler().fit_transform(df[HYPER].values).mean(axis=1)
n_ext = max(int((df.grp == 'HP').sum()), int(round(len(df) * 0.03)))
ext_ids = set(df.sort_values('hyper_sev', ascending=False).head(n_ext)['person_id'])
df['is_extreme'] = df['person_id'].isin(ext_ids) & (df.grp != 'HP')
df['grp4'] = np.where(df.grp == 'HP', 'HP',
             np.where(df.grp == 'Stable', 'Stable',
             np.where(df['is_extreme'], 'extreme-Spiker', 'mild-Spiker')))
CLASSES = ['Stable', 'mild-Spiker', 'extreme-Spiker', 'HP']
CODE = {c: i for i, c in enumerate(CLASSES)}

RET = []
def rec(s=''):
    RET.append(s); print(s)

n_e = int(df['is_extreme'].sum()); n_h = int((df.grp == 'HP').sum())
rec(f'extreme-Spiker n={n_e} ({100*n_e/len(df):.1f}%)  ·  HP n={n_h} ({100*n_h/len(df):.1f}%) '
    f'— rarity-matched by construction')
rec(f'class sizes: ' + str(df.grp4.value_counts().to_dict()))
rec()

# fewer than ~20 members in a TRAINING fold can never be isolated by a leaf and its recall
# collapses toward 0 REGARDLESS of whether the features carry signal. This figure is entirely
# about rare classes (extreme-Spiker 39, HP 39 -> 24-31 per training fold), i.e. precisely the
# regime where the default is an artifact rather than a measurement. The audit
# (logs/gbt_leafsize_audit.md) showed it distorts BOTH ways here: the HbA1c-alone extreme-Spiker
# cell was INFLATED 0.12 -> 0.45 by the default. So the figure now plots leaf=2 and the log
# quotes both, per the rule "never report a GBT per-class recall below 25 per training fold
# without re-running at min_samples_leaf=2 and quoting both".
GBT_LEAF = 2

def oof(cols, model, target, leaf=GBT_LEAF):
    """Pooled OOF over the 4-class problem; returns recall/precision of `target` + its OVR AUC."""
    s = df.dropna(subset=cols)
    X = s[cols].values.astype(float)
    y = np.array([CODE[v] for v in s['grp4']])
    t = CODE[target]
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pred = np.empty(len(y), int); proba = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        if model == 'lr':
            clf = LogisticRegression(max_iter=5000, random_state=SEED).fit(sc.transform(X[tr]), y[tr])
            Xte = sc.transform(X[te])
        else:
            clf = HistGradientBoostingClassifier(min_samples_leaf=leaf,
                                                 random_state=SEED).fit(X[tr], y[tr])
            Xte = X[te]
        pred[te] = clf.predict(Xte)
        p = clf.predict_proba(Xte)
        if t in clf.classes_:
            proba[te] = p[:, list(clf.classes_).index(t)]
    return (recall_score(y, pred, labels=[t], average=None, zero_division=0)[0],
            precision_score(y, pred, labels=[t], average=None, zero_division=0)[0],
            roc_auc_score((y == t).astype(int), proba),
            len(y), int((y == t).sum()))

ARMS = [('CGM\n14 feat.', CGM14, CGM_C),
        ('HbA1c\nalone', ['hba1c'], BIO_C0),
        ('non-fasting\n9 markers', BIO_NF, BIO_C)]

res = {}
for target in ['extreme-Spiker', 'HP']:
    rec(f'=== recall of {target} (4-class model, features swapped) ===')
    for lab, cols, _c in ARMS:
        r_lr, p_lr, a_lr, n, nt = oof(cols, 'lr', target)
        r_gb, p_gb, a_gb, _, _ = oof(cols, 'gbt', target, leaf=GBT_LEAF)
        r_gd, _, _, _, _ = oof(cols, 'gbt', target, leaf=20)   # sklearn default, for the record
        res[(target, lab)] = dict(lr=r_lr, gbt=r_gb, gbt_default=r_gd,
                                  prec_lr=p_lr, auc_lr=a_lr, n=n, nt=nt)
        _flag = ' <-- default is an artifact here' if abs(r_gd - r_gb) >= 0.05 else ''
        rec(f'  {lab.replace(chr(10)," "):24s} recall LR {r_lr:.2f} | '
            f'GBT(leaf={GBT_LEAF}) {r_gb:.2f} | GBT(leaf=20, sklearn default) {r_gd:.2f}{_flag}   '
            f'(prec LR {p_lr:.2f}, OVR-AUC LR {a_lr:.3f}; n={n}, {target}={nt}, '
            f'~{int(nt*0.8)} per training fold)')
    rec()

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=300)
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
xpos = np.arange(len(ARMS)); wd = 0.30
lr_v = [res[('extreme-Spiker', l)]['lr'] for l, _, _ in ARMS]
gb_v = [res[('extreme-Spiker', l)]['gbt'] for l, _, _ in ARMS]
cols = [c for _, _, c in ARMS]
ax.bar(xpos - wd/2, lr_v, wd, color=cols)
ax.bar(xpos + wd/2, gb_v, wd, color=cols, alpha=0.55)
for xi, vl, vg in zip(xpos, lr_v, gb_v):
    if abs(vl - vg) < 0.005:
        ax.text(xi, max(vl, vg) + 0.028, f'{vl:.2f}', ha='center', fontsize=7.6, fontweight='bold')
    else:
        ax.text(xi - wd/2, vl + 0.028, f'{vl:.2f}', ha='center', fontsize=7.4, fontweight='bold')
        ax.text(xi + wd/2, vg + 0.028, f'{vg:.2f}', ha='center', fontsize=7.4)

prev = n_e / len(df)
ax.axhline(prev, color='#999', ls=':', lw=0.9)
ax.text(len(ARMS) - 0.42, prev + 0.02, f'extreme-Spiker prevalence {prev*100:.1f}%',
        ha='right', va='bottom', fontsize=6.2, color='#777')
ax.set_xticks(xpos); ax.set_xticklabels([l for l, _, _ in ARMS], fontsize=6.8)
for xi, (l, _, _) in zip(xpos, ARMS):
    r = res[('extreme-Spiker', l)]
    ax.text(xi, -0.115, f"n={r['n']:,}·ext {r['nt']}", ha='center', va='top',
            fontsize=5.4, color='#888', transform=ax.get_xaxis_transform())
ax.set_xlim(-0.62, len(ARMS) - 0.38); ax.set_ylim(0, 1.08)
ax.set_ylabel('extreme-Spiker recall (sensitivity)', fontsize=8.5)
ax.set_title('Do labs find the extreme Spikers?\nsame 4-class model, features swapped',
             fontsize=9.2, fontweight='bold', pad=8)
h = [plt.Rectangle((0, 0), 1, 1, fc='#7a7a7a'), plt.Rectangle((0, 0), 1, 1, fc='#7a7a7a', alpha=0.55)]
ax.legend(h, ['logistic regression', f'gradient boosting (leaf={GBT_LEAF})'], frameon=False,
          fontsize=6.8, loc='upper right')
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.tick_params(labelsize=7.2, length=2.5)
fig.text(0.5, 0.005,
         f'extreme-Spiker = the rule (top ~3% on a CGM-only hyper-severity score, HP excluded, '
         f'n={n_e}) — same rarity as HP (n={n_h}).\ngradient boosting at min_samples_leaf='
         f'{GBT_LEAF}: at this class size the sklearn default (20) cannot isolate the minority '
         f'in a leaf (see logs/gbt_leafsize_audit.md)',
         ha='center', va='bottom', fontsize=5.4, color='#888', linespacing=1.35)
fig.subplots_adjust(bottom=0.235, top=0.845, left=0.155, right=0.975)
for e in ('png', 'pdf'):
    fig.savefig(f'{OUT}/detection_gap_extreme_spiker.{e}', facecolor='white')

with open(f'{LOGD}/detection_gap_extreme_spiker.md', 'w') as f:
    f.write('# Detection gap — extreme Spikers (Figure-5b design, outcome swapped)\n\n```\n'
            + '\n'.join(RET) + '\n```\n')
print(f'\nWROTE {OUT}/detection_gap_extreme_spiker.{{png,pdf}}')
print(f'WROTE {LOGD}/detection_gap_extreme_spiker.md')
