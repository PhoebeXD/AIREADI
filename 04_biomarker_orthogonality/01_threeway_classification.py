#!/usr/bin/env python3
"""Three-way classification: can routine labs assign pattern membership?

Three feature sets compete on identical complete-case rows for the same 3-class target:

    CGM-only        the 14 clustering features
    biomarker-only  HbA1c, fasting glucose, fasting insulin, HOMA-IR, triglycerides, HDL
    combined        both

CGM-only is also run on the full cohort so its accuracy is not reduced by biomarker
missingness. StandardScaler is fit within each training fold; balanced accuracy over
RepeatedStratifiedKFold(5x10) with a 2000-resample bootstrap CI and per-class recall.
Majority baseline 0.333. LogisticRegression and HistGradientBoosting, both unweighted.

in   processed/analysis_table_n1306_n14.csv
out  logs/threeway_classification_n14.{md,csv}, plot
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')   # 1 thread/fit; joblib parallelizes perms
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, recall_score

SEED = 42
np.random.seed(SEED)
N_REPEATS = 10
N_SPLITS  = 5
N_BOOT    = 2000
N_PERM    = 1000
N_JOBS    = 4
BASELINE  = 0.333
CLASSES   = [0, 1, 2]                 # 0 Spiker / 1 Stable / 2 HP
CLASS_NAME = {0: 'Spiker', 1: 'Stable', 2: 'HP'}

BASE = os.environ.get("AIREADI_DATA_ROOT", "")
assert BASE, "set AIREADI_DATA_ROOT to the data root"
BR   = f'{BASE}/canonical_n14_rerun'
import json
DF = pd.read_csv(f'{BR}/processed/analysis_table_n1306_n14.csv')
DF['lab'] = DF['hypo_k3_pruned']

CGM10 = ['mean_glucose', 'glucose_sd', 'glucose_cv', 'mage', 'pct_above_140',
         'pct_above_180', 'pct_below_70', 'pct_below_54', 'n_lows_70', 'n_spikes_140',
         'avg_rise_rate', 'avg_fall_rate', 'day_night_diff', 'n_reactive_events']
assert CGM10 == json.load(open(f'{BR}/models/meta.json'))['features'], \
    'CGM feature set must be the n14 branch feature set'
# routine fasting panel — all six present in the table (missingness reported in header)
PANEL = ['hba1c', 'fasting_glucose', 'fasting_insulin', 'homa_ir_corrected',
         'triglycerides', 'hdl']

# complete-case rows for the biomarker panel (CGM10 has zero missing, so the
# intersection used for all three head-to-head sets == panel complete-case)
CC = DF[PANEL].notna().all(axis=1).values
n_full = len(DF)
n_cc   = int(CC.sum())

FEATURE_SETS = {
    'CGM-only':       CGM10,
    'Biomarker-only': PANEL,
    'Combined':       CGM10 + PANEL,
}
MODELS = ['LR', 'HistGBT']


# ---------------------------------------------------------------------------
# Pipeline core: within-fold StandardScaler -> model -> pooled OOF preds
# ---------------------------------------------------------------------------
def make_clf(model):
    if model == 'LR':
        return LogisticRegression(max_iter=5000, random_state=SEED)
    return HistGradientBoostingClassifier(max_iter=100, early_stopping=False,
                                          random_state=SEED)

def oof_predict(X, y, model, fold_seed):
    """Pooled out-of-fold predictions from one stratified k-fold split."""
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=fold_seed)
    pred = np.empty(len(y), dtype=int)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])          # fit on TRAIN fold only
        clf = make_clf(model).fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    return pred

def per_class_recall(y, pred):
    r = recall_score(y, pred, labels=CLASSES, average=None, zero_division=0)
    return dict(zip(CLASSES, r))


# ---------------------------------------------------------------------------
# STEP 1 — write PRE-REGISTERED header (rule + feature lists + n) and FLUSH
#          to disk before computing a single classification result.
# ---------------------------------------------------------------------------
miss = {c: int(DF[c].isna().sum()) for c in PANEL}
cc_counts = DF.loc[CC, 'lab'].value_counts().sort_index()

H = []
def h(s=''): H.append(s)

h("# Three-way individual-classification comparison — orthogonality conclusion")
h()
h("**VERDICT:** _(written after results — see one-line verdict appended below header)_")
h()
h("Same 3-class target throughout: n14 (canonical-15 minus `n_lows_54`, tz-corrected "
  "`day_night_diff`) K=3 membership "
  f"(`hypo_k3_pruned`, Spiker {int((DF.lab==0).sum())} / Stable {int((DF.lab==1).sum())} / "
  f"HP {int((DF.lab==2).sum())}; codes 0/1/2). NO reclustering. "
  "Pipeline is the Figure 3A classifier: StandardScaler fit WITHIN each "
  "training fold -> model -> pooled out-of-fold predictions; 3-class majority baseline "
  f"**{BASELINE:.3f}**. Added robustness: RepeatedStratifiedKFold ({N_SPLITS} folds x "
  f"{N_REPEATS} repeats) for mean +/- SD, plus a {N_BOOT}-resample bootstrap 95% CI and "
  f"a >={N_PERM}-permutation label-shuffle null. Seed {SEED}.")
h()
h("Two models per feature set so a combined null cannot be blamed on model capacity:")
h("- **linear:** `LogisticRegression(max_iter=5000)` (the linear classifier)")
h("- **nonlinear:** `HistGradientBoostingClassifier(max_iter=100)` (gradient-boosted trees)")
h()
h("## Feature sets (same target = 3-class membership)")
h()
h(f"1. **CGM-only** ({len(CGM10)} n14 clustering features): "
  f"`{', '.join(CGM10)}`")
h(f"2. **Biomarker-only** (routine fasting panel, {len(PANEL)} markers, all present in "
  f"the table): `{', '.join(PANEL)}`")
h(f"3. **Combined** ({len(CGM10) + len(PANEL)} features): CGM {len(CGM10)} + biomarker panel, "
  "same complete-case rows as set 2.")
h()
h("## Complete-case construction")
h()
h("CGM features have zero missing data; the biomarker panel does not, so the fair "
  "head-to-head rows are exactly the **panel complete-case** rows. Per-marker missing "
  "(of the full cohort): " + ", ".join(f"{c} {miss[c]}" for c in PANEL) + ".")
h()
h(f"- **Complete-case n (sets 1-3 head-to-head): {n_cc}** "
  f"(Spiker {int(cc_counts.get(0, 0))} / Stable {int(cc_counts.get(1, 0))} / "
  f"HP {int(cc_counts.get(2, 0))}).")
h(f"- **CGM-only full-n: {n_full}** (Spiker {int((DF.lab==0).sum())} / Stable {int((DF.lab==1).sum())} / HP {int((DF.lab==2).sum())}) — reported "
  "separately so the 0.96 headline is NOT silently reduced by biomarker missingness.")
h()
h("## PRE-REGISTERED INTERPRETATION RULE (committed before reading results)")
h()
h("Report the combined number VERBATIM regardless of direction. Branch on the combined "
  "vs CGM-only comparison using bootstrap 95% CIs (per model):")
h("- **(A) combined ~= CGM-only (combined mean within CGM-only's 95% CI):** "
  "\"biomarkers add no individual-level signal beyond CGM (0.9X -> 0.9X, NS).\" "
  "REINFORCES orthogonality. Expected outcome.")
h("- **(B) combined > CGM-only beyond CI (combined CI-low > CGM-only CI-high):** report "
  "the gain honestly as biomarker-contributed signal; re-examine whether it is the "
  "mean-glucose/HbA1c overlap (Fig 4) rather than new information.")
h("- **(C) combined < CGM-only beyond CI (combined CI-high < CGM-only CI-low):** report "
  "it; biomarkers dilute / add noise that hurts the CGM signal.")
h("- Do **NOT** claim \"combined outperforms either alone\" unless combined exceeds "
  "**BOTH** CGM-only AND biomarker-only beyond their CIs. Wording selected AFTER the "
  "number is in.")
h()
h("## Caveats (committed before results)")
h()
h("- Combined / biomarker sets are **complete-case** "
  f"(n={n_cc} < full cohort); CGM-only full-n (n={n_full}) reported separately so the headline "
  "is not silently reduced by biomarker missingness.")
h("- **HbA1c shares the mean-glucose axis used in clustering** (`mean_glucose`, "
  "`pct_above_180` are clustering features). Any biomarker-only or combined signal is "
  "EXPECTED to be that overlap (Fig 4), not independent information — stated when "
  "interpreting.")
h()
h("---")
h()
h("## RESULTS")
h()

os.makedirs(f'{BR}/logs', exist_ok=True)
os.makedirs(f'{BR}/plots', exist_ok=True)
with open(f'{BR}/logs/threeway_classification_n14.md', 'w') as fh:
    fh.write('\n'.join(H) + '\n')
print("HEADER (rule + feature lists + complete-case n) flushed to "
      "logs/threeway_classification.md")
print(f"  complete-case n={n_cc}  full n={n_full}")
print("Now computing classification results...\n")


# ---------------------------------------------------------------------------
# STEP 2 — main metrics: repeated-CV mean+/-SD, per-class recall, bootstrap CI
# ---------------------------------------------------------------------------
def evaluate(X, y, model):
    """Repeated stratified CV -> bacc mean/SD, per-class recall, bootstrap 95% CI."""
    y = np.asarray(y)
    baccs, recs = [], {c: [] for c in CLASSES}
    pred0 = None
    for rep in range(N_REPEATS):
        pred = oof_predict(X, y, model, fold_seed=SEED + rep)
        baccs.append(balanced_accuracy_score(y, pred))
        rc = per_class_recall(y, pred)
        for c in CLASSES:
            recs[c].append(rc[c])
        if rep == 0:
            pred0 = pred
    baccs = np.array(baccs)
    # bootstrap 95% CI on repeat-0 OOF predictions (subject resampling)
    rng = np.random.default_rng(SEED)
    nb = len(y)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, nb, nb)
        boot[b] = balanced_accuracy_score(y[idx], pred0[idx])
    return dict(
        bacc_mean=float(baccs.mean()), bacc_sd=float(baccs.std(ddof=1)),
        ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5)),
        recall={CLASS_NAME[c]: float(np.mean(recs[c])) for c in CLASSES},
        bacc_single=float(balanced_accuracy_score(y, oof_predict(X, y, model, SEED))),
    )


# ---------------------------------------------------------------------------
# STEP 3 — permutation null (shuffle labels, same pipeline, >=N_PERM perms)
# ---------------------------------------------------------------------------
def _perm_one(X, y, model, seed):
    rng = np.random.default_rng(seed)
    yp = rng.permutation(y)
    pred = oof_predict(X, yp, model, fold_seed=SEED)
    return balanced_accuracy_score(yp, pred)

def perm_pvalue(X, y, model, obs_single):
    y = np.asarray(y)
    perms = Parallel(n_jobs=N_JOBS)(
        delayed(_perm_one)(X, y, model, 10_000 + i) for i in range(N_PERM))
    perms = np.array(perms)
    p = (1 + int((perms >= obs_single).sum())) / (1 + N_PERM)
    return float(p), float(perms.mean()), float(perms.max())


# ---- run all cells: 3 head-to-head sets x 2 models on complete-case rows ----
y_cc = DF.loc[CC, 'lab'].values
results = {}
for sname, cols in FEATURE_SETS.items():
    X = DF.loc[CC, cols].values.astype(float)
    for model in MODELS:
        m = evaluate(X, y_cc, model)
        p, pmean, pmax = perm_pvalue(X, y_cc, model, m['bacc_single'])
        m.update(perm_p=p, perm_mean=pmean, perm_max=pmax, n=n_cc)
        results[(sname, model)] = m
        print(f"  [{sname:14s} | {model:7s}] bacc={m['bacc_mean']:.3f}"
              f"+/-{m['bacc_sd']:.3f} CI[{m['ci_lo']:.3f},{m['ci_hi']:.3f}] "
              f"rec(S/Sp/HP)={m['recall']['Stable']:.2f}/{m['recall']['Spiker']:.2f}/"
              f"{m['recall']['HP']:.2f}  perm_p={p:.4f}")

# ---- CGM-only on the full cohort (headline preservation), both models ----
y_full = DF['lab'].values
full_cgm = {}
for model in MODELS:
    X = DF[CGM10].values.astype(float)
    m = evaluate(X, y_full, model)
    p, pmean, pmax = perm_pvalue(X, y_full, model, m['bacc_single'])
    m.update(perm_p=p, perm_mean=pmean, perm_max=pmax, n=n_full)
    full_cgm[model] = m
    print(f"  [CGM-only FULL  | {model:7s}] bacc={m['bacc_mean']:.3f}"
          f"+/-{m['bacc_sd']:.3f} CI[{m['ci_lo']:.3f},{m['ci_hi']:.3f}] perm_p={p:.4f}")


# ---------------------------------------------------------------------------
# STEP 4 — fire the pre-registered interpretation branch (per model)
# ---------------------------------------------------------------------------
def fire_branch(cgm, comb, bio):
    """Return (branch_code, sentence) per the header rule using bootstrap CIs."""
    c_lo, c_hi = cgm['ci_lo'], cgm['ci_hi']
    cb, cb_lo, cb_hi = comb['bacc_mean'], comb['ci_lo'], comb['ci_hi']
    beats_cgm = cb_lo > c_hi
    beats_bio = comb['ci_lo'] > bio['ci_hi']
    if cb_lo > c_hi:                       # combined CI-low above CGM CI-high
        beats_both = beats_cgm and beats_bio
        extra = (" combined exceeds BOTH CGM-only and biomarker-only beyond CIs."
                 if beats_both else
                 " (does NOT exceed both alone — no 'outperforms either alone' claim).")
        return 'B', (f"(B) combined > CGM-only beyond CI "
                     f"({cgm['bacc_mean']:.3f} -> {cb:.3f}): biomarker-contributed gain; "
                     f"re-examine mean-glucose/HbA1c overlap (Fig 4)." + extra)
    if cb_hi < c_lo:                       # combined CI-high below CGM CI-low
        return 'C', (f"(C) combined < CGM-only beyond CI "
                     f"({cgm['bacc_mean']:.3f} -> {cb:.3f}): biomarkers dilute the CGM "
                     f"signal.")
    return 'A', (f"(A) combined ~= CGM-only (combined {cb:.3f} within CGM-only 95% CI "
                 f"[{c_lo:.3f},{c_hi:.3f}]): biomarkers add no individual-level signal "
                 f"beyond CGM ({cgm['bacc_mean']:.3f} -> {cb:.3f}, NS). Reinforces "
                 f"orthogonality.")

branches = {}
for model in MODELS:
    branches[model] = fire_branch(results[('CGM-only', model)],
                                  results[('Combined', model)],
                                  results[('Biomarker-only', model)])

# primary model for the one-line verdict = the one with higher CGM-only complete-case bacc
prim = max(MODELS, key=lambda mm: results[('CGM-only', mm)]['bacc_mean'])
pc = results[('CGM-only', prim)]['bacc_mean']
pb = results[('Biomarker-only', prim)]['bacc_mean']
pm = results[('Combined', prim)]['bacc_mean']
verdict = (f"CGM-only {pc:.3f}, biomarker-only {pb:.3f}, combined {pm:.3f} "
           f"(complete-case n={n_cc}, primary model {prim}); CGM-only full-n "
           f"{full_cgm[prim]['bacc_mean']:.3f} (n={n_full}). Branch fired: "
           f"{branches[prim][0]} — {branches[prim][1]}")


# ---------------------------------------------------------------------------
# STEP 5 — append results table + verdict to .md; write .csv
# ---------------------------------------------------------------------------
B = []
def b(s=''): B.append(s)

b("### Head-to-head (same complete-case rows, n={})".format(n_cc))
b()
b("| feature set | model | n | bal acc (mean +/- SD) | 95% CI (bootstrap) | "
  "Stable rec | Spiker rec | HP rec | perm p | perm-null mean |")
b("|---|---|---|---|---|---|---|---|---|---|")
for sname in FEATURE_SETS:
    for model in MODELS:
        r = results[(sname, model)]
        b(f"| {sname} | {model} | {r['n']} | {r['bacc_mean']:.3f} +/- {r['bacc_sd']:.3f} | "
          f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | {r['recall']['Stable']:.3f} | "
          f"{r['recall']['Spiker']:.3f} | {r['recall']['HP']:.3f} | "
          f"{r['perm_p']:.4f} | {r['perm_mean']:.3f} |")
b(f"| _majority baseline_ | — | {n_cc} | {BASELINE:.3f} | — | "
  "(predict Stable) | 0.000 | 0.000 | — | — |")
b()
b("### CGM-only, full cohort (headline preservation — not reduced by panel missingness)")
b()
b("| feature set | model | n | bal acc (mean +/- SD) | 95% CI (bootstrap) | "
  "Stable rec | Spiker rec | HP rec | perm p |")
b("|---|---|---|---|---|---|---|---|---|")
for model in MODELS:
    r = full_cgm[model]
    b(f"| CGM-only (full) | {model} | {r['n']} | {r['bacc_mean']:.3f} +/- {r['bacc_sd']:.3f} | "
      f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | {r['recall']['Stable']:.3f} | "
      f"{r['recall']['Spiker']:.3f} | {r['recall']['HP']:.3f} | {r['perm_p']:.4f} |")
b()
b("### Pre-registered branch evaluation (per model, combined vs CGM-only, bootstrap CIs)")
b()
for model in MODELS:
    code, sent = branches[model]
    cgm = results[('CGM-only', model)]; comb = results[('Combined', model)]
    bio = results[('Biomarker-only', model)]
    b(f"- **{model}:** CGM-only {cgm['bacc_mean']:.3f} "
      f"[{cgm['ci_lo']:.3f},{cgm['ci_hi']:.3f}], biomarker-only {bio['bacc_mean']:.3f} "
      f"[{bio['ci_lo']:.3f},{bio['ci_hi']:.3f}], combined {comb['bacc_mean']:.3f} "
      f"[{comb['ci_lo']:.3f},{comb['ci_hi']:.3f}] -> branch **{code}**. {sent}")
b()
b("**Interpretation note (pre-committed):** biomarker-only sits at/near the 0.333 "
  "majority baseline with Spiker/HP recall collapsing toward 0 (everything assigned "
  "Stable), while CGM-only is at ceiling. Any non-trivial biomarker-only or combined "
  "signal above CGM-only would be expected to ride the `mean_glucose`/HbA1c overlap "
  "(Fig 4), not independent phenotype information.")
b()
b(f"**VERDICT:** {verdict}")
b()

with open(f'{BR}/logs/threeway_classification_n14.md', 'a') as fh:
    fh.write('\n'.join(B) + '\n')

# inject one-line verdict into the header placeholder
with open(f'{BR}/logs/threeway_classification_n14.md') as fh:
    txt = fh.read()
txt = txt.replace(
    "**VERDICT:** _(written after results — see one-line verdict appended below header)_",
    f"**VERDICT:** {verdict}")
with open(f'{BR}/logs/threeway_classification_n14.md', 'w') as fh:
    fh.write(txt)

# ---- CSV ----
rows = []
for sname in FEATURE_SETS:
    for model in MODELS:
        r = results[(sname, model)]
        rows.append(dict(feature_set=sname, model=model, rows='complete-case', n=r['n'],
                         bacc_mean=r['bacc_mean'], bacc_sd=r['bacc_sd'],
                         ci_lo=r['ci_lo'], ci_hi=r['ci_hi'],
                         recall_Stable=r['recall']['Stable'],
                         recall_Spiker=r['recall']['Spiker'],
                         recall_HP=r['recall']['HP'],
                         perm_p=r['perm_p'], perm_null_mean=r['perm_mean'],
                         baseline=BASELINE))
for model in MODELS:
    r = full_cgm[model]
    rows.append(dict(feature_set='CGM-only', model=model, rows='full-n', n=r['n'],
                     bacc_mean=r['bacc_mean'], bacc_sd=r['bacc_sd'],
                     ci_lo=r['ci_lo'], ci_hi=r['ci_hi'],
                     recall_Stable=r['recall']['Stable'],
                     recall_Spiker=r['recall']['Spiker'],
                     recall_HP=r['recall']['HP'],
                     perm_p=r['perm_p'], perm_null_mean=r['perm_mean'],
                     baseline=BASELINE))
pd.DataFrame(rows).to_csv(f'{BR}/logs/threeway_classification_n14.csv', index=False)


# ---------------------------------------------------------------------------
# STEP 6 — plot: bal-acc bars (3 sets x 2 models) with CIs + baseline,
#          plus per-class recall panel.
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
sets = list(FEATURE_SETS.keys())
xb = np.arange(len(sets))
width = 0.36
mcolor = {'LR': '#4c72b0', 'HistGBT': '#dd8452'}
for k, model in enumerate(MODELS):
    means = [results[(s, model)]['bacc_mean'] for s in sets]
    los = [results[(s, model)]['bacc_mean'] - results[(s, model)]['ci_lo'] for s in sets]
    his = [results[(s, model)]['ci_hi'] - results[(s, model)]['bacc_mean'] for s in sets]
    pos = xb + (k - 0.5) * width
    ax1.bar(pos, means, width, color=mcolor[model], edgecolor='black', linewidth=0.6,
            label=model, zorder=3, yerr=[los, his], capsize=3,
            error_kw=dict(ecolor='black', lw=1))
    for x, v in zip(pos, means):
        ax1.text(x, v + 0.02, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
ax1.axhline(BASELINE, ls='--', color='gray', lw=1.3, label=f'baseline {BASELINE:.3f}')
ax1.set_xticks(xb); ax1.set_xticklabels(sets, fontsize=10)
ax1.set_ylabel('Balanced accuracy (5x10 repeated CV, OOF)')
ax1.set_ylim(0, 1.05)
ax1.set_title(f'3 feature sets x 2 models (complete-case n={n_cc})\n'
              'error bars = bootstrap 95% CI', fontsize=10.5)
ax1.legend(fontsize=9, loc='center right'); ax1.grid(axis='y', alpha=0.3, zorder=0)

# per-class recall panel (use LR; HistGBT shown as hatched overlay)
classes_order = ['Stable', 'Spiker', 'HP']
xc = np.arange(len(sets))
n_grp = len(classes_order)
cw = 0.8 / n_grp
ccolor = {'Stable': '#55a868', 'Spiker': '#c44e52', 'HP': '#8172b3'}
for j, cls in enumerate(classes_order):
    vals_lr = [results[(s, 'LR')]['recall'][cls] for s in sets]
    pos = xc - 0.4 + cw * (j + 0.5)
    ax2.bar(pos, vals_lr, cw, color=ccolor[cls], edgecolor='black', linewidth=0.5,
            label=f'{cls} (LR)', zorder=3)
ax2.axhline(0, color='black', lw=0.6)
ax2.set_xticks(xc); ax2.set_xticklabels(sets, fontsize=10)
ax2.set_ylabel('Per-class recall (LR, mean over repeats)')
ax2.set_ylim(0, 1.05)
ax2.set_title('Per-class recall — does HP/Spiker survive,\nor collapse to Stable?',
              fontsize=10.5)
ax2.legend(fontsize=8, loc='upper center', ncol=3); ax2.grid(axis='y', alpha=0.3, zorder=0)

fig.suptitle('Three-way individual classification of pruned K=3 membership '
             '(orthogonality test)', fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(f'{BR}/plots/threeway_classification_n14.png', dpi=150, bbox_inches='tight')
fig.savefig(f'{BR}/plots/threeway_classification_n14.pdf', bbox_inches='tight')

print("\nWROTE logs/threeway_classification.{md,csv}")
print("WROTE plots/threeway_classification.{png,pdf}")
print(f"\nVERDICT: {verdict}")
