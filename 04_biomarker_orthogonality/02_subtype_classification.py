#!/usr/bin/env python3
"""Biomarker classification at finer subgroup resolution.

Extends the three-way test to the Spiker sub-structure. extreme-Spiker is the top ~3% of
the cohort on a CGM hyper-severity z-score (mean of standardized mean_glucose,
pct_above_180 and n_spikes_140), rarity-matched to Hypo-Prone and excluding it;
mild-Spiker is every other Spiker.

    T1  Spiker vs Stable, HP removed                     baseline 0.500
    T2  mild-Spiker vs extreme-Spiker, HP removed        baseline 0.500
    T3  HP / Spiker / Stable                             baseline 0.333
    T4  HP / Stable / mild-Spiker / extreme-Spiker       baseline 0.250

Within-fold scaling, pooled out-of-fold predictions, balanced accuracy over
RepeatedStratifiedKFold(5x10), 2000-resample bootstrap CI, per-class recall and ROC-AUC.
Neither model sets class_weight: a minority class collapsing to near-zero recall under an
unweighted model is itself the result, and imbalance is handled on the evaluation side.
CGM-only is the positive control in every test.

in   processed/analysis_table_n1306_n14.csv, clinical_data/observation.csv
out  subtype_classification.{md,csv,json}, plot
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score

SEED = 42
np.random.seed(SEED)
N_REPEATS = 10
N_SPLITS  = 5
N_BOOT    = 2000
N_PERM    = 500          # 1/501 p-resolution; ample for the above-chance test
N_JOBS    = 4

BASE = os.environ.get("AIREADI_DATA_ROOT", "")
assert BASE, "set AIREADI_DATA_ROOT to the data root"
BR = f'{BASE}/canonical_n14_rerun'
OUT    = f'{BR}/subtype_classification'
os.makedirs(OUT, exist_ok=True)

# n14 clustering feature set (canonical-15 minus n_lows_54) — must match models/meta.json.
CGM_N14 = ['mean_glucose', 'glucose_sd', 'glucose_cv', 'mage', 'pct_above_140',
         'pct_above_180', 'pct_below_70', 'pct_below_54', 'n_lows_70', 'n_spikes_140',
         'avg_rise_rate', 'avg_fall_rate', 'day_night_diff', 'n_reactive_events']
_META = json.load(open(f'{BR}/models/meta.json'))
assert CGM_N14 == _META['features'], 'CGM control must be the branch feature set'
# (fasting glucose, fasting insulin, HOMA-IR, triglycerides) require a fasted draw, and
# AI-READI has no fasting flag, only self-report. The arms below therefore escalate the same
# way the hypo-prone detection analysis does: best single clinical test -> broad
# fasting-clean panel -> complete panel restricted to valid specimens, so one standard
# applies throughout.
PANEL_LEGACY6 = ['hba1c', 'fasting_glucose', 'fasting_insulin', 'homa_ir_corrected',
                 'triglycerides', 'hdl']          # retired; kept for provenance only
BIO_HBA1C = ['hba1c']
BIO_NF = ['hba1c', 'hdl', 'total_cholesterol', 'crp', 'alt', 'ast', 'creatinine',
          'urine_albumin', 'wbc']                                    # 9, no fasted draw needed
BIO_FULL = ['hba1c', 'fasting_glucose', 'insulin', 'c_peptide', 'homa_ir_corrected',
            'triglycerides', 'hdl', 'ldl', 'total_cholesterol', 'crp', 'alt', 'ast',
            'creatinine', 'urine_albumin', 'wbc']                    # 15, fasted subset only
FAST_MIN = 12
HYPER = ['mean_glucose', 'pct_above_180', 'n_spikes_140']

# --------------------------------------------------------------------------- load
DF = pd.read_csv(f'{BR}/processed/analysis_table_n1306_n14.csv')
assert DF['hypo_k3_n14'].isna().sum() == 0
NAME3 = {0: 'Spiker', 1: 'Stable', 2: 'HP'}
DF['grp'] = DF['hypo_k3_n14'].map(NAME3)
assert DF[CGM_N14].isna().sum().sum() == 0, 'CGM features must be complete'

# fasting duration, self-report (OMOP observation 2005200151, "hours since last ate";
# one response per person). Same source as Figure 3 and Figure 5b.
_obs = pd.read_csv(f'{BASE}/clinical_data/observation.csv',
                   usecols=['person_id', 'observation_concept_id', 'value_as_number'])
_fh = (_obs[_obs['observation_concept_id'] == 2005200151][['person_id', 'value_as_number']]
       .rename(columns={'value_as_number': 'fast_h'}))
DF['person_id'] = DF['person_id'].astype(int)
_fh['person_id'] = _fh['person_id'].astype(int)
DF = DF.merge(_fh, on='person_id', how='left')
assert DF.person_id.is_unique, 'fasting merge changed row count'
FASTED = (DF['fast_h'] >= FAST_MIN).fillna(False)
print(f'fasted (>={FAST_MIN} h self-report): n={int(FASTED.sum())} '
      f'({100*FASTED.mean():.1f}% of cohort); HP among them = '
      f'{int((FASTED & (DF["hypo_k3_n14"] == 2)).sum())}')

# --------------------------------------------- extreme-Spiker definition
z = StandardScaler().fit_transform(DF[HYPER].values)
DF['hyper_sev'] = z.mean(axis=1)
n_hp   = int((DF.grp == 'HP').sum())
n_ext  = max(n_hp, int(round(len(DF) * 0.03)))
ext_ids = set(DF.sort_values('hyper_sev', ascending=False).head(n_ext)['person_id'])
DF['is_extreme'] = DF['person_id'].isin(ext_ids) & (DF.grp != 'HP')
n_ext = int(DF['is_extreme'].sum())
DF['grp4'] = np.where(DF.grp == 'HP', 'HP',
             np.where(DF.grp == 'Stable', 'Stable',
             np.where(DF['is_extreme'], 'extreme-Spiker', 'mild-Spiker')))
comp = DF[DF['is_extreme']]['grp'].value_counts().to_dict()
n_mild = int(((DF.grp == 'Spiker') & (~DF['is_extreme'])).sum())
print(f'loaded n={len(DF)}  grp={DF.grp.value_counts().to_dict()}')
print(f'extreme-Spiker n={n_ext} (composition {comp}); mild-Spiker n={n_mild}')

# --------------------------------------------------------------------------- tests
# each test: name, dataframe-mask -> (label array as strings), baseline, classes order
def mk_labels(mask, labelfn):
    sub = DF[mask].copy()
    y = np.asarray(labelfn(sub))     # labelfn may return a Series or np.where ndarray
    return sub, y

TESTS = []
# T1 Spiker vs Stable (HP removed)
TESTS.append(dict(
    key='T1_spiker_vs_stable', title='Spiker vs Stable (HP removed)',
    mask=(DF.grp != 'HP'),
    labelfn=lambda s: s['grp'],           # Spiker / Stable
    classes=['Stable', 'Spiker'], baseline=0.5))
# T2 mild vs extreme Spiker (HP + Stable removed)
TESTS.append(dict(
    key='T2_mild_vs_extreme', title='mild-Spiker vs extreme-Spiker (HP removed)',
    mask=(DF.grp == 'Spiker'),
    labelfn=lambda s: np.where(s['is_extreme'], 'extreme-Spiker', 'mild-Spiker'),
    classes=['mild-Spiker', 'extreme-Spiker'], baseline=0.5))
# T3 HP / Spiker / Stable
TESTS.append(dict(
    key='T3_hp_spiker_stable', title='HP / Spiker / Stable (3-class)',
    mask=pd.Series(True, index=DF.index),
    labelfn=lambda s: s['grp'],
    classes=['Stable', 'Spiker', 'HP'], baseline=1/3))
# T4 HP / Stable / mild / extreme
TESTS.append(dict(
    key='T4_hp_stable_mild_extreme', title='HP / Stable / mild-Spiker / extreme-Spiker (4-class)',
    mask=pd.Series(True, index=DF.index),
    labelfn=lambda s: s['grp4'],
    classes=['Stable', 'mild-Spiker', 'extreme-Spiker', 'HP'], baseline=0.25))

# name -> (columns, row restriction). Row restriction is a boolean Series on DF's index,
# or None for the full untreated cohort.
FEATURE_SETS = {
    'CGM 14-feat':    (CGM_N14,   None),
    'HbA1c alone':    (BIO_HBA1C, None),
    'non-fasting 9':  (BIO_NF,    None),
    'full 15 fasted': (BIO_FULL,  FASTED),
}
PRIMARY_BIO = 'non-fasting 9'   # headline biomarker arm: fasting-clean AND full cohort
# escalation ladder, best-supported first; verdict_for() walks it and uses the first arm
# that survives MIN_PER_CLASS, naming the arm it landed on.
BIO_ARM_ORDER = ['non-fasting 9', 'HbA1c alone', 'full 15 fasted']
# fold, recall is a coin flip, the bootstrap CI resamples the same handful of people and the
# permutation null is degenerate. The 08-02/03 session log (Part E) states the guard as
# 5 * N_SPLITS = 25 (">=5 per fold in every class") but the shipped code never implemented
# it, so the 03:53 run evaluated cells it was designed to refuse (fasted extreme-Spiker 5,
# fasted HP 6). Now matches the recorded design.
MIN_PER_CLASS = 5 * N_SPLITS    # >=5 members per fold in EVERY class
MODELS = ['LR', 'HistGBT']

# --------------------------------------------------------------------------- core
def make_clf(model):
    # Unweighted. Balanced accuracy + ROC-AUC are the
    # imbalance-robust readouts; a minority class collapsing to recall->0 under an
    # unweighted classifier is itself the finding (association vs individual assignment).
    if model == 'LR':
        return LogisticRegression(max_iter=5000, random_state=SEED)
    return HistGradientBoostingClassifier(max_iter=100, early_stopping=False,
                                          random_state=SEED)

def oof_predict(X, y_int, model, fold_seed, return_proba=False, n_classes=None):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=fold_seed)
    pred = np.empty(len(y_int), dtype=int)
    proba = np.zeros((len(y_int), n_classes)) if return_proba else None
    for tr, te in skf.split(X, y_int):
        sc = StandardScaler().fit(X[tr])
        clf = make_clf(model).fit(sc.transform(X[tr]), y_int[tr])
        Xte = sc.transform(X[te])
        pred[te] = clf.predict(Xte)
        if return_proba:
            p = clf.predict_proba(Xte)
            # align to global class order 0..n_classes-1
            for j, c in enumerate(clf.classes_):
                proba[te, c] = p[:, j]
    return (pred, proba) if return_proba else pred

def auc_score(y_int, proba, n_classes):
    try:
        if n_classes == 2:
            return roc_auc_score(y_int, proba[:, 1])
        return roc_auc_score(y_int, proba, multi_class='ovr', average='macro')
    except Exception:
        return np.nan

def evaluate(X, y_int, model, n_classes, class_names):
    baccs, recs = [], {c: [] for c in range(n_classes)}
    pred0 = None
    for rep in range(N_REPEATS):
        pred = oof_predict(X, y_int, model, SEED + rep)
        baccs.append(balanced_accuracy_score(y_int, pred))
        r = recall_score(y_int, pred, labels=list(range(n_classes)),
                         average=None, zero_division=0)
        for c in range(n_classes):
            recs[c].append(r[c])
        if rep == 0:
            pred0 = pred
    baccs = np.array(baccs)
    # AUC from OOF probabilities (single split, seed)
    _, proba = oof_predict(X, y_int, model, SEED, return_proba=True, n_classes=n_classes)
    auc = auc_score(y_int, proba, n_classes)
    # bootstrap 95% CI on repeat-0 OOF preds (subject resampling)
    rng = np.random.default_rng(SEED)
    nb = len(y_int)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, nb, nb)
        boot[b] = balanced_accuracy_score(y_int[idx], pred0[idx])
    return dict(
        bacc_mean=float(baccs.mean()), bacc_sd=float(baccs.std(ddof=1)),
        ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5)),
        auc=float(auc) if auc == auc else None,
        recall={class_names[c]: float(np.mean(recs[c])) for c in range(n_classes)},
        min_recall=float(min(np.mean(recs[c]) for c in range(n_classes))),
        bacc_single=float(balanced_accuracy_score(y_int, oof_predict(X, y_int, model, SEED))))

def _perm_one(X, y_int, model, seed):
    rng = np.random.default_rng(seed)
    yp = rng.permutation(y_int)
    return balanced_accuracy_score(yp, oof_predict(X, yp, model, SEED))

def perm_pvalue(X, y_int, model, obs_single):
    perms = np.array(Parallel(n_jobs=N_JOBS)(
        delayed(_perm_one)(X, y_int, model, 10_000 + i) for i in range(N_PERM)))
    return (1 + int((perms >= obs_single).sum())) / (1 + N_PERM), float(perms.mean())

# --------------------------------------------------------------------------- run
results = {}   # (test_key, feature_set, model) -> metrics
meta_tests = {}
for T in TESTS:
    sub, y_str = mk_labels(T['mask'], T['labelfn'])
    cls = T['classes']
    code = {c: i for i, c in enumerate(cls)}
    n_classes = len(cls)
    for fs, (cols, rowmask) in FEATURE_SETS.items():
        cc = sub[cols].notna().all(axis=1).values
        if rowmask is not None:
            cc = cc & rowmask.reindex(sub.index).fillna(False).values
        Xall = sub.loc[cc, cols].values.astype(float)
        y_int = np.array([code[v] for v in np.asarray(y_str)[cc]])
        n = len(y_int)
        dist = {cls[i]: int((y_int == i).sum()) for i in range(n_classes)}
        if fs == PRIMARY_BIO:
            meta_tests[T['key']] = dict(title=T['title'], baseline=T['baseline'],
                                        n=n, classes=cls, dist=dist)
        # a stratified 5-fold needs >= N_SPLITS members in EVERY class. The fasted arm is
        # small (HP 6 of 278), so some cells are simply not evaluable — record that rather
        # than emit a number computed on 2 people.
        if min(dist.values()) < MIN_PER_CLASS:
            for model in MODELS:
                results[(T['key'], fs, model)] = dict(
                    skipped=True, n=n, dist=dist,
                    reason=f'smallest class has {min(dist.values())} < {MIN_PER_CLASS} rows')
            print(f"[{T['key']:26s} | {fs:14s} | SKIPPED] n={n} dist={dist} "
                  f"(smallest class {min(dist.values())} < {MIN_PER_CLASS})")
            continue
        for model in MODELS:
            m = evaluate(Xall, y_int, model, n_classes, cls)
            p, pmean = perm_pvalue(Xall, y_int, model, m['bacc_single'])
            m.update(perm_p=p, perm_mean=pmean, n=n, dist=dist)
            results[(T['key'], fs, model)] = m
            aucs = f"{m['auc']:.3f}" if m['auc'] is not None else 'na'
            print(f"[{T['key']:26s} | {fs:14s} | {model:7s}] "
                  f"bacc={m['bacc_mean']:.3f}+/-{m['bacc_sd']:.3f} "
                  f"CI[{m['ci_lo']:.3f},{m['ci_hi']:.3f}] AUC={aucs} "
                  f"minRec={m['min_recall']:.2f} p={p:.4f} (base {T['baseline']:.3f})")

# --------------------------------------------------------------------------- verdict
ASSIGN_RECALL = 0.50   # min per-class recall required to claim individual ASSIGNMENT

def verdict_for(test_key):
    """Grade on the ASSIGNMENT-vs-ASSOCIATION distinction (Fig-3 framing).
    A test is above chance if bacc CI-low > baseline AND perm p<0.05 (population
    signal exists). But that is ASSOCIATION, not the ability to classify an
    individual — for that the hardest class must actually be recalled. So:
      AT CHANCE               : not above chance
      ASSOCIATION ONLY        : above chance but a class collapses (min recall < 0.10)
      ASSOCIATION, NOT ASSIGN : above chance, min recall 0.10-0.50 (ranking, not
                                individual assignment — biomarkers see the mean, Fig 3/4)
      INDIVIDUALLY SEPARABLE  : above chance AND min recall >= 0.50 (grade by bacc)
    balanced accuracy alone can pass while the minority is mostly missed, so the
    min-recall gate is what keeps the label honest."""
    T = meta_tests[test_key]; base = T['baseline']
    # extreme-Spiker is 24 on the non-fasting-9 complete case, one short of 25). Returning
    # a bare NOT EVALUABLE there would throw away the HbA1c-alone arm, which IS evaluable
    # (extreme-Spiker 33). So fall back
    # down the escalation ladder and SAY which arm the verdict came from.
    arm = next((a for a in BIO_ARM_ORDER
                if any(not results[(test_key, a, mm)].get('skipped') for mm in MODELS)),
               None)
    if arm is None:
        return None, None, None, False, ('NOT EVALUABLE on any biomarker arm — smallest '
                                         f'class < {MIN_PER_CLASS} rows throughout')
    _avail = [mm for mm in MODELS if not results[(test_key, arm, mm)].get('skipped')]
    bm = max(_avail, key=lambda mm: results[(test_key, arm, mm)]['bacc_mean'])
    r = results[(test_key, arm, bm)]
    b, mr = r['bacc_mean'], r['min_recall']
    auc = r['auc']; aucs = f"{auc:.3f}" if auc is not None else 'na'
    above = (r['ci_lo'] > base) and (r['perm_p'] < 0.05)
    assigns = above and (mr >= ASSIGN_RECALL)
    if not above:
        grade = f'AT CHANCE — no signal (bacc {b:.3f}, AUC {aucs})'
    elif mr < 0.10:
        grade = (f'ASSOCIATION ONLY — majority-collapse, a class recall→0; graded '
                 f'AUC {aucs} but NO individual assignment (bacc {b:.3f})')
    elif not assigns:
        grade = (f'ASSOCIATION, NOT INDIVIDUAL ASSIGNMENT — above-chance ranking '
                 f'(AUC {aucs}) but min per-class recall only {mr:.2f}; cannot assign '
                 f'individuals (bacc {b:.3f})')
    else:
        strength = 'WEAK' if b < 0.60 else ('MODERATE' if b < 0.75 else 'STRONG')
        grade = f'INDIVIDUALLY SEPARABLE — {strength} (bacc {b:.3f}, min recall {mr:.2f}, AUC {aucs})'
    if arm != PRIMARY_BIO:
        grade += (f' [fallback arm: the primary arm "{PRIMARY_BIO}" is not evaluable here '
                  f'(smallest class < {MIN_PER_CLASS}); graded on "{arm}" instead]')
    return arm, bm, r, assigns, grade

# --------------------------------------------------------------------------- write md
H = []
def h(s=''): H.append(s)
h('# Biomarker classification of glycemic subgroups (Spiker sub-structure + HP)')
h()
h('**VERDICT:** _(filled after results)_')
h()
h('Extends the three-way test (biomarker-only vs CGM-only for HP/Spiker/Stable) to the finer Spiker '
  'sub-structure. `extreme-Spiker`: top ~3% of the cohort on '
  'a pure CGM hyper-severity z-score (mean of standardized mean_glucose, pct_above_180, '
  f'n_spikes_140), rarity-matched to HP, EXCLUDING HP -> n={n_ext} (composition {comp}); '
  f'`mild-Spiker` = every other Spiker (n={n_mild}). The extreme axis is built from CGM '
  'ONLY; biomarkers never see it.')
h()
h('Pipeline = as elsewhere (within-fold StandardScaler -> classifier -> pooled OOF), '
  'generalized to each label set. LogisticRegression + HistGradientBoostingClassifier, '
  'both **UNWEIGHTED** ('
  'Neither model sets class_weight. '
  'For the class-balanced counterpart see 09c_imbalance_decomposition). Balanced accuracy over '
  f'RepeatedStratifiedKFold({N_SPLITS}x{N_REPEATS}); {N_BOOT}-boot 95% CI; >={N_PERM}-perm '
  'null; per-class recall; ROC-AUC (binary standard / multiclass macro-OVR). Imbalance is '
  'handled on the EVALUATION side (balanced accuracy + AUC); a minority class collapsing to '
  'recall->0 under an unweighted model is itself the finding. '
  f'Seed {SEED}. CGM-only ({len(CGM_N14)} n14 clustering features) is the positive control.')
h()
h('## INTERPRETATION RULE (assignment vs association — the Fig-3 distinction)')
h('A test has a population SIGNAL iff (i) bacc bootstrap CI-low > baseline AND (ii) perm '
  'p<0.05. But a signal is ASSOCIATION, not the ability to classify an INDIVIDUAL — for '
  'that the hardest class must actually be recalled. Grades: AT CHANCE (no signal); '
  'ASSOCIATION ONLY (signal but a class collapses, min recall <0.10); ASSOCIATION, NOT '
  'INDIVIDUAL ASSIGNMENT (signal, min recall 0.10-0.50 — a ranking, not individual '
  'assignment); INDIVIDUALLY SEPARABLE (signal AND min recall >=0.50, graded WEAK<0.60 / '
  'MODERATE 0.60-0.75 / STRONG >=0.75 by bacc). balanced accuracy alone can pass while the '
  'minority is mostly missed, so the min-recall gate keeps the label honest; ROC-AUC is '
  'reported as the population-ranking strength regardless.')
h()
h('## Tests')
for T in TESTS:
    mt = meta_tests[T['key']]
    h(f"- **{T['key']}** — {mt['title']}: baseline {mt['baseline']:.3f}, "
      f"biomarker complete-case n={mt['n']} {mt['dist']}")
h()
h('---')
h()
h('## RESULTS')
h()
with open(f'{OUT}/subtype_classification.md', 'w') as f:
    f.write('\n'.join(H) + '\n')
print('\nheader flushed.')

# results tables
B = []
def bb(s=''): B.append(s)
for T in TESTS:
    tk = T['key']; mt = meta_tests[tk]
    arm, bm, r, sep, grade = verdict_for(tk)
    bb(f"### {mt['title']}")
    bb(f"baseline **{mt['baseline']:.3f}** · complete-case n={mt['n']} · classes {mt['dist']}")
    bb()
    bb('| feature set | model | bal acc (mean±SD) | 95% CI | AUC | per-class recall | perm p |')
    bb('|---|---|---|---|---|---|---|')
    for fs in FEATURE_SETS:
        for model in MODELS:
            m = results[(tk, fs, model)]
            if m.get('skipped'):
                bb(f"| {fs} | {model} | n={m['n']} {m['dist']} | — | — | "
                   f"NOT EVALUABLE ({m['reason']}) | — |")
                continue
            aucs = f"{m['auc']:.3f}" if m['auc'] is not None else 'na'
            rec = ' / '.join(f"{k} {v:.2f}" for k, v in m['recall'].items())
            bb(f"| {fs} (n={m['n']}) | {model} | {m['bacc_mean']:.3f} ± {m['bacc_sd']:.3f} | "
               f"[{m['ci_lo']:.3f}, {m['ci_hi']:.3f}] | {aucs} | {rec} | {m['perm_p']:.4f} |")
    _cgm = [results[(tk, 'CGM 14-feat', mm)]['bacc_mean'] for mm in MODELS
            if not results[(tk, 'CGM 14-feat', mm)].get('skipped')]
    bb(f"\n**{tk} verdict (biomarker arm '{arm}', {bm}):** {grade}. "
       f"CGM control bacc {max(_cgm):.3f}." if _cgm else
       f"\n**{tk} verdict (biomarker arm '{arm}', {bm}):** {grade}.")
    bb()

# overall verdict line
def sep_word(tk):
    _, _, _, sep, grade = verdict_for(tk)
    return grade
verdict = (
    "Biomarkers — "
    f"T1 Spiker/Stable: {sep_word('T1_spiker_vs_stable')}; "
    f"T2 mild/extreme Spiker: {sep_word('T2_mild_vs_extreme')}; "
    f"T3 HP/Spiker/Stable: {sep_word('T3_hp_spiker_stable')}; "
    f"T4 HP/Stable/mild/extreme: {sep_word('T4_hp_stable_mild_extreme')}. "
    "CGM-only is the positive control throughout.")
bb(f"## VERDICT\n{verdict}")

with open(f'{OUT}/subtype_classification.md', 'a') as f:
    f.write('\n'.join(B) + '\n')
with open(f'{OUT}/subtype_classification.md') as f:
    txt = f.read()
txt = txt.replace('**VERDICT:** _(filled after results)_', f'**VERDICT:** {verdict}')
with open(f'{OUT}/subtype_classification.md', 'w') as f:
    f.write(txt)

# csv
rows = []
for tk in [t['key'] for t in TESTS]:
    for fs in FEATURE_SETS:
        for model in MODELS:
            m = results[(tk, fs, model)]
            if m.get('skipped'):
                rows.append(dict(test=tk, feature_set=fs, model=model, n=m['n'],
                                 baseline=meta_tests[tk]['baseline'], skipped=True,
                                 reason=m['reason']))
                continue
            row = dict(test=tk, feature_set=fs, model=model, n=m['n'],
                       baseline=meta_tests[tk]['baseline'],
                       bacc_mean=m['bacc_mean'], bacc_sd=m['bacc_sd'],
                       ci_lo=m['ci_lo'], ci_hi=m['ci_hi'], auc=m['auc'],
                       min_recall=m['min_recall'], perm_p=m['perm_p'],
                       perm_null_mean=m['perm_mean'])
            for k, v in m['recall'].items():
                row[f'recall_{k}'] = v
            rows.append(row)
pd.DataFrame(rows).to_csv(f'{OUT}/subtype_classification.csv', index=False)
with open(f'{OUT}/subtype_meta.json', 'w') as f:
    json.dump(dict(seed=SEED, n_extreme=n_ext, n_mild=n_mild,
                   extreme_composition=comp, tests=meta_tests), f, indent=2, default=str)

# --------------------------------------------------------------------------- plot
fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
for ax, T in zip(axes, TESTS):
    tk = T['key']; base = T['baseline']
    xb = np.arange(len(FEATURE_SETS)); width = 0.36
    mcolor = {'LR': '#4c72b0', 'HistGBT': '#dd8452'}
    for k, model in enumerate(MODELS):
        _g = lambda fs, k, d=np.nan: (np.nan if results[(tk, fs, model)].get('skipped')
                                      else results[(tk, fs, model)][k])
        means = [_g(fs, 'bacc_mean') for fs in FEATURE_SETS]
        los = [_g(fs, 'bacc_mean') - _g(fs, 'ci_lo') for fs in FEATURE_SETS]
        his = [_g(fs, 'ci_hi') - _g(fs, 'bacc_mean') for fs in FEATURE_SETS]
        pos = xb + (k - 0.5) * width
        ax.bar(pos, means, width, color=mcolor[model], edgecolor='black', linewidth=0.6,
               label=model, zorder=3, yerr=[los, his], capsize=3,
               error_kw=dict(ecolor='black', lw=1))
        for x, v in zip(pos, means):
            ax.text(x, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=7.5)
    ax.axhline(base, ls='--', color='gray', lw=1.2, label=f'baseline {base:.2f}')
    ax.set_xticks(xb); ax.set_xticklabels(list(FEATURE_SETS.keys()), fontsize=7.0, rotation=18, ha='right')
    ax.set_ylim(0, 1.05); ax.set_title(T['title'], fontsize=9)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    if ax is axes[0]:
        ax.set_ylabel('Balanced accuracy (5x10 CV, OOF)')
        ax.legend(fontsize=7.5, loc='upper left')
fig.suptitle('Can biomarkers classify the glycemic subgroups? '
             '(biomarker-only vs CGM-only positive control)', fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(f'{OUT}/subtype_classification.png', dpi=150, bbox_inches='tight')
fig.savefig(f'{OUT}/subtype_classification.pdf', bbox_inches='tight')

print(f'\nVERDICT: {verdict}')
print(f'\noutputs -> {OUT}/')
print('DONE')
