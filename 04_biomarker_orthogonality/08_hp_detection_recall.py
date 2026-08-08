#!/usr/bin/env python3
"""Can routine labs detect the Hypo-Prone group?

One pipeline, one label scheme, one recall definition: a three-class model over the pattern
labels, evaluated by pooled out-of-fold recall of Hypo-Prone. Only the feature set is swapped
between arms, so a difference between arms cannot be attributed to differing class counts,
models or evaluation rules.

Full-cohort arms
    CGM features        the clustering features, a positive control rather than an
                        independent predictor: the labels are k-means assignments on these
                        same features, so its recall is a ceiling, not a finding
    HbA1c alone         the single standard-of-care glycemic test
    non-fasting panel   the nine analytes that do not require a fasted draw

The cohort carries no fasting flag; the only fasting information is self-reported hours since
last eating, and the median is a few hours. Markers that require a fasted draw are therefore
excluded from the full-cohort arms rather than computed on mostly-postprandial blood.

Fasted paired arm
    The complete fifteen-marker panel is also given its chance, on the subset that reports at
    least twelve hours fasted and is complete on all fifteen. That subset holds very few
    Hypo-Prone members, so an unweighted three-class model predicting the minority never is
    the expected outcome whether or not the labs carry signal. A lone lab bar at zero there
    cannot distinguish "the labs are blind" from "too few cases to recall anything". The arm
    is therefore run as a PAIR on identical rows with identical folds, swapping only the
    features: if the CGM arm recovers those same people and the labs do not, power is
    demonstrably sufficient and the zero means what it appears to mean. It is reported as a
    supporting record, not as a headline number.

Gradient boosting in the paired arm uses min_samples_leaf=2. The scikit-learn default of 20
exceeds the entire Hypo-Prone training class at that subset size, so no leaf can isolate it and
recall collapses to zero independently of signal — an artifact rather than a measurement.
Wherever a per-class recall is computed on fewer than about 25 members per training fold, both
the default and the small-leaf value are reported.

No unit conversion is applied to the insulin or HOMA-IR columns. Every arm here is either
rank-based or standardized within fold, so a positive scalar on a feature cannot move any
result; the conversion matters only where an absolute value is printed.

in   processed/analysis_table_n1306_n14.csv, models/meta.json, clinical_data/observation.csv
out  logs/hp_detection_recall.{md,csv}, logs/hp_detection_recall_paired.csv
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score

SEED = 42
np.random.seed(SEED)
BASE = os.environ.get("AIREADI_DATA_ROOT", "")
assert BASE, "set AIREADI_DATA_ROOT to the data root"
BR = f'{BASE}/canonical_n14_rerun'
LOGD = f'{BR}/logs'
os.makedirs(LOGD, exist_ok=True)

HP_CODE = 2                 # Hypo-Prone
FAST_MIN = 12               # hours since last ate, self-report
FAST_CONCEPT = 2005200151   # OMOP observation concept, "hours since last ate"
SMALL_LEAF = 2
DEFAULT_LEAF = 20           # scikit-learn default, reported alongside where the class is small
LEAF_RULE_N = 25            # per-training-fold class size below which both leaf values are quoted
PAT = {0: 'Spiker', 1: 'Stable', 2: 'Hypo-Prone'}

CGM_FEATS = json.load(open(f'{BR}/models/meta.json'))['features']
BIO_NF = ['hba1c', 'hdl', 'total_cholesterol', 'crp', 'alt', 'ast', 'creatinine',
          'urine_albumin', 'wbc']
BIO_FULL = ['hba1c', 'fasting_glucose', 'insulin', 'c_peptide', 'homa_ir_corrected',
            'triglycerides', 'hdl', 'ldl', 'total_cholesterol', 'crp', 'alt', 'ast',
            'creatinine', 'urine_albumin', 'wbc']

RET = []
def rec(s=''):
    RET.append(s); print(s, flush=True)


# ---------------------------------------------------------------- load
df = pd.read_csv(f'{BR}/processed/analysis_table_n1306_n14.csv')
assert df.person_id.is_unique, 'duplicate person_id in the analysis table'
vc = df['hypo_k3_pruned'].value_counts().to_dict()
assert len(vc) == 3, 'expected three pattern labels'
df['person_id'] = df['person_id'].astype(int)
n_rows = len(df)

obs = pd.read_csv(f'{BASE}/clinical_data/observation.csv',
                  usecols=['person_id', 'observation_concept_id', 'value_as_number'])
fh = (obs[obs['observation_concept_id'] == FAST_CONCEPT][['person_id', 'value_as_number']]
      .rename(columns={'value_as_number': 'fast_h'}))
fh['person_id'] = fh['person_id'].astype(int)
assert fh.person_id.is_unique, 'more than one fasting response per person'
df = df.merge(fh, on='person_id', how='left')
assert len(df) == n_rows, 'fasting merge changed the row count'

n_hp = int((df['hypo_k3_pruned'] == HP_CODE).sum())
hp_prev = n_hp / len(df)
df_fast = df[df['fast_h'] >= FAST_MIN]
rec(f'cohort n={len(df):,}  ·  ' + '  '.join(f'{PAT[g]} {vc[g]}' for g in (0, 1, 2)))
rec(f'Hypo-Prone prevalence {hp_prev * 100:.1f}%  ({n_hp}/{len(df):,})')
rec(f'>={FAST_MIN} h self-reported fasted: n={len(df_fast):,} '
    f'({100 * len(df_fast) / len(df):.0f}% of cohort), Hypo-Prone '
    f'{int((df_fast["hypo_k3_pruned"] == HP_CODE).sum())}')
rec(f'CGM arm: {len(CGM_FEATS)} features (from models/meta.json)')
rec()


def oof_predict(frame, cols, model, leaf=DEFAULT_LEAF):
    """Pooled out-of-fold predictions of the three-class problem on `frame`.

    Complete-case on `cols`. StandardScaler is fitted inside each training fold for the linear
    model; gradient boosting is scale-free and takes the raw values.
    """
    s = frame.dropna(subset=cols)
    X = s[cols].values.astype(float)
    y = s['hypo_k3_pruned'].values
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pred = np.empty(len(y), int)
    for tr, te in skf.split(X, y):
        if model == 'lr':
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=5000,
                                     random_state=SEED).fit(sc.transform(X[tr]), y[tr])
            pred[te] = clf.predict(sc.transform(X[te]))
        else:
            clf = HistGradientBoostingClassifier(min_samples_leaf=leaf,
                                                 random_state=SEED).fit(X[tr], y[tr])
            pred[te] = clf.predict(X[te])
    return y, pred


def hp_scores(frame, cols, model, leaf=DEFAULT_LEAF):
    y, pred = oof_predict(frame, cols, model, leaf)
    return dict(recall=recall_score(y, pred, labels=[HP_CODE], average=None, zero_division=0)[0],
                precision=precision_score(y, pred, labels=[HP_CODE], average=None,
                                          zero_division=0)[0],
                hits=int(((pred == HP_CODE) & (y == HP_CODE)).sum()),
                false_pos=int(((pred == HP_CODE) & (y != HP_CODE)).sum()),
                n=len(y), n_hp=int((y == HP_CODE).sum()))


# ---------------------------------------------------------------- full-cohort arms
ARMS = [('CGM features',      CGM_FEATS, df),
        ('HbA1c alone',       ['hba1c'], df),
        ('non-fasting panel', BIO_NF,    df),
        (f'full panel, >={FAST_MIN} h fasted', BIO_FULL, df_fast)]

rec('=== Hypo-Prone recall — same three-class model, features swapped ===')
rows = []
for name, cols, frame in ARMS:
    lr = hp_scores(frame, cols, 'lr')
    gb = hp_scores(frame, cols, 'gbt', leaf=DEFAULT_LEAF)
    per_fold = gb['n_hp'] * 0.8
    small = per_fold < LEAF_RULE_N
    # below the leaf rule the default cannot isolate the minority class; quote both
    gb2 = hp_scores(frame, cols, 'gbt', leaf=SMALL_LEAF) if small else None

    rows.append(dict(arm=name, n_features=len(cols), n=lr['n'], n_hp=lr['n_hp'],
                     recall_lr=lr['recall'], recall_gbt=gb['recall'],
                     recall_gbt_leaf2=(gb2['recall'] if gb2 else np.nan),
                     precision_lr=lr['precision'], precision_gbt=gb['precision'],
                     hits_lr=lr['hits'], hits_gbt=gb['hits'],
                     false_pos_lr=lr['false_pos'], false_pos_gbt=gb['false_pos'],
                     plotted=not small))
    extra = (f' | GBT(leaf={SMALL_LEAF}) {gb2["recall"]:.2f}' if gb2 else '')
    flag = (f'   [Hypo-Prone ~{per_fold:.0f} per training fold, below {LEAF_RULE_N}: '
            f'underpowered, supporting record only]' if small else '')
    rec(f'  {name:32s} LR {lr["recall"]:.2f} | GBT {gb["recall"]:.2f}{extra}   '
        f'(n={lr["n"]:,}, Hypo-Prone={lr["n_hp"]}){flag}')

res = pd.DataFrame(rows)
main = res[res.plotted]
cgm_row = res[res.arm == 'CGM features'].iloc[0]
lab_rows = main[main.arm != 'CGM features']
lab_vals = np.r_[lab_rows.recall_lr.values, lab_rows.recall_gbt.values]
cgm_vals = np.r_[cgm_row.recall_lr, cgm_row.recall_gbt]
rec()
rec(f'  CGM arm across models      {cgm_vals.min():.2f}-{cgm_vals.max():.2f}')
rec(f'  lab arms across models     {lab_vals.min():.2f}-{lab_vals.max():.2f}   '
    f'(quote the range, not a point value)')
rec(f'  reference: predicting Hypo-Prone at random would recall it at its prevalence, '
    f'{hp_prev * 100:.1f}%')
rec()

# ---------------------------------------------------------------- fasted paired arm
# Identical rows, identical folds, identical model — only the feature set differs. This is the
# comparison that makes the lab zero interpretable at this sample size.
fast_cc = df[(df['fast_h'] >= FAST_MIN) & df[BIO_FULL].notna().all(axis=1)]
n_hp_f = int((fast_cc['hypo_k3_pruned'] == HP_CODE).sum())
rec(f'=== fasted paired arm — identical rows (n={len(fast_cc):,}, Hypo-Prone={n_hp_f}) ===')
rec(f'  gradient boosting at min_samples_leaf={SMALL_LEAF}: the default ({DEFAULT_LEAF}) '
    f'exceeds the Hypo-Prone training class (~{n_hp_f * 0.8:.0f} per fold) and cannot '
    f'isolate it in a leaf.')

pair_rows = []
for name, cols in [('CGM features', CGM_FEATS), ('full panel (15 markers)', BIO_FULL)]:
    for model, leaf in [('lr', None), ('gbt', SMALL_LEAF), ('gbt', DEFAULT_LEAF)]:
        s = hp_scores(fast_cc, cols, model, leaf=(leaf or DEFAULT_LEAF))
        tag = model if model == 'lr' else f'gbt(leaf={leaf})'
        assert s['n'] == len(fast_cc), 'paired arm rows must be identical across feature sets'
        pair_rows.append(dict(arm=name, model=tag, n=s['n'], n_hp=n_hp_f,
                              recovered=s['hits'], recall=s['recall'],
                              false_pos=s['false_pos']))
        rec(f'  {name:24s} {tag:14s} recovered {s["hits"]}/{n_hp_f}  '
            f'(recall {s["recall"]:.2f}, false positives {s["false_pos"]})')
pair = pd.DataFrame(pair_rows)

cgm_best = int(pair[pair.arm == 'CGM features'].recovered.max())
bio_best = int(pair[pair.arm != 'CGM features'].recovered.max())
rec(f'  => on the same {len(fast_cc):,} people, CGM features recover up to {cgm_best}/{n_hp_f} '
    f'and the complete lab panel up to {bio_best}/{n_hp_f}.')
if cgm_best > bio_best:
    rec('     The subset can support recall of this class, so the lab result is a detection '
        'failure rather than a power failure.')
else:
    rec('     The CGM arm does not separate them either at this size, so this subset cannot '
        'adjudicate the lab arm; rely on the full-cohort arms above.')

# ---------------------------------------------------------------- outputs
res.to_csv(f'{LOGD}/hp_detection_recall.csv', index=False)
pair.to_csv(f'{LOGD}/hp_detection_recall_paired.csv', index=False)
with open(f'{LOGD}/hp_detection_recall.md', 'w') as f:
    f.write('# Hypo-Prone detection — CGM features versus routine labs\n\n')
    f.write('Three-class model over the pattern labels, pooled out-of-fold recall of '
            'Hypo-Prone, five-fold stratified. Only the feature set differs between arms. '
            'The CGM arm is a positive control: the labels are k-means assignments on those '
            'same features, so its recall is a ceiling.\n\n```\n' + '\n'.join(RET) + '\n```\n')
    f.write('\n## Arms\n\n')
    f.write(res.round(4).to_markdown(index=False))
    f.write('\n\n## Fasted paired arm\n\n')
    f.write(pair.round(4).to_markdown(index=False))
    f.write('\n')

print(f'\nWROTE {LOGD}/hp_detection_recall.{{md,csv}}')
print(f'WROTE {LOGD}/hp_detection_recall_paired.csv')
