"""Biomarker characterization across the three patterns, 15 analytes.

The cohort has no fasting flag and the median time since last eating is 3 h, so markers are
split by what a non-fasted draw supports:

    fasting-independent (HbA1c + 9 non-glycemic labs)  full untreated cohort
    fasting-dependent (fasting glucose, insulin, C-peptide, HOMA-IR, triglycerides)
                                                       >= 12 h self-reported fasted subset

Kruskal-Wallis across the three patterns with Benjamini-Hochberg FDR, eta-squared effect
sizes, and pairwise Mann-Whitney contrasts.

in   processed/analysis_table (n14 labels), clinical_data/observation.csv
out  logs/characterization_fitforpurpose_n1306_n14.{md,csv}, pairwise CSV
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

BASE = os.environ.get("AIREADI_DATA_ROOT", "")
assert BASE, "set AIREADI_DATA_ROOT to the data root"
FAST_MIN = 12          # hours since last ate, self-report
PAT = {0: 'Spiker', 1: 'Stable', 2: 'Hypo-Prone'}

TABLE  = f'{BASE}/canonical_n14_rerun/processed/analysis_table_n1306_n14.csv'
LABCOL = 'hypo_k3_n14'
META   = f'{BASE}/canonical_n14_rerun/models/meta.json'
SUFFIX = '_n14'
print(f'labels={LABCOL}')

# marker -> (label, needs_fasting)
MARKERS = [
    ('hba1c',             'HbA1c (%)',               False),
    ('fasting_glucose',   'Fasting glucose (mg/dL)', True),
    ('insulin',           'Fasting insulin (uU/mL)', True),
    ('c_peptide',         'C-peptide (ng/mL)',       True),
    ('homa_ir_corrected', 'HOMA-IR (corrected)',     True),
    ('triglycerides',     'Triglycerides',           True),
    ('hdl',               'HDL',                     False),
    ('ldl',               'LDL',                     False),
    ('total_cholesterol', 'Total cholesterol',       False),
    ('crp',               'CRP (mg/L)',              False),
    ('alt',               'ALT (U/L)',               False),
    ('ast',               'AST (U/L)',               False),
    ('creatinine',        'Creatinine',              False),
    ('urine_albumin',     'Urine albumin',           False),
    ('wbc',               'WBC',                     False),
]

df = pd.read_csv(TABLE)

# Unit convention for the stored insulin columns. `insulin` and `fasting_insulin` are stored in
# ng/mL, so conversion to uU/mL is x28.70 (insulin MW 5808; 1 uU/mL = 6 pmol/L), and
# `homa_ir_corrected` -- computed with a x6 constant -- sits 4.783x below true HOMA-IR. The unit
# is established in Methods on three independent lines: the OMOP unit label, shared with
# C-peptide, whose own reference range fixes what that label means; the C-peptide-to-insulin
# molar ratio, which is physiological only under ng/mL; and the absolute insulin and HOMA-IR of
# fasted, lean, normoglycemic participants.
# The rescale is a positive scalar, so every rank-based result here -- Spearman, AUC, balanced
# accuracy, Kruskal-Wallis / Mann-Whitney p, FDR verdict -- is unchanged; only absolutes move.
NGML_TO_UU = (1e6 / 5808.0) / 6.0     # 28.70 uU/mL per ng/mL
X6_TO_TRUE = NGML_TO_UU / 6.0         # 4.783 -> true HOMA-IR
df['insulin'] = df['insulin'] * NGML_TO_UU
df['homa_ir_corrected'] = df['homa_ir_corrected'] * X6_TO_TRUE
assert df.person_id.is_unique, 'duplicate person_id'
df['hypo_k3'] = df[LABCOL]            # unified internal name

# Size vector keyed by pattern name -- n_Spiker, n_Stable, n_Hypo-Prone -- not by cluster code.
# A positional tuple depends on the code-to-pattern mapping, so a re-fit that permutes the codes
# would compare the wrong cells without saying so.
N_OBS = {PAT[g]: int((df['hypo_k3'] == g).sum()) for g in sorted(PAT)}
assert all(n > 0 for n in N_OBS.values()), f'a pattern is empty: {N_OBS}'

# The expected sizes are read from the fitted model's metadata, not written out here. Other
# label sets exist for the same participants and every one of them has three non-empty clusters
# over the same cohort, so neither a count of labels nor a row total can tell them apart -- only
# the sizes can. Keeping those numbers in the artifact rather than in this file leaves the source
# free of cohort values while still catching a label column from the wrong solution.
N_EXP = json.load(open(META))['sizes']
assert N_OBS == N_EXP, (f'label column does not match the fitted model -- '
                        f'observed {N_OBS}, model {N_EXP}; check LABCOL={LABCOL}')

# fasting duration, self-report (same source Fig 3 uses)
obs = pd.read_csv(f'{BASE}/clinical_data/observation.csv',
                  usecols=['person_id', 'observation_concept_id', 'value_as_number'])
fh = (obs[obs['observation_concept_id'] == 2005200151][['person_id', 'value_as_number']]
      .rename(columns={'value_as_number': 'fast_h'}))
df['person_id'] = df['person_id'].astype(int)
fh['person_id'] = fh['person_id'].astype(int)
df = df.merge(fh, on='person_id', how='left')
assert df.person_id.is_unique, 'fasting merge changed row count'

n_fast = int((df['fast_h'] >= FAST_MIN).sum())
print(f'cohort n={len(df)}; >={FAST_MIN} h fasted n={n_fast} ({100*n_fast/len(df):.0f}%)')
for g in [0, 1, 2]:
    m = (df['fast_h'] >= FAST_MIN) & (df['hypo_k3'] == g)
    print(f'  {PAT[g]:11s} fasted {int(m.sum()):4d} / {int((df.hypo_k3==g).sum()):4d}')


def eta_squared(H, n, k):
    """Eta-squared for Kruskal-Wallis: (H - k + 1) / (n - k). Floored at 0."""
    return max(0.0, (H - k + 1) / (n - k))


def band(e):
    return 'large' if e >= 0.14 else 'medium' if e >= 0.06 else 'small' if e >= 0.01 else 'negligible'


rows, pw_rows = [], []
for col, label, needs_fast in MARKERS:
    base = df[df['fast_h'] >= FAST_MIN] if needs_fast else df
    s = base.dropna(subset=[col])
    groups = [s.loc[s.hypo_k3 == g, col].values for g in [0, 1, 2]]
    ns = [len(x) for x in groups]
    H, p = kruskal(*groups)
    e2 = eta_squared(H, len(s), 3)
    rows.append(dict(variable=col, label=label, fasted_subset=needs_fast,
                     cohort=f'>={FAST_MIN}h fasted' if needs_fast else 'full',
                     n_total=len(s), n_Spiker=ns[0], n_Stable=ns[1], n_HP=ns[2],
                     med_Spiker=np.median(groups[0]), med_Stable=np.median(groups[1]),
                     med_HP=np.median(groups[2]),
                     kruskal_H=H, kruskal_p=p, eta2=e2, effect_size=band(e2)))
    pw = dict(variable=col, label=label)
    for a, b, nm in [(0, 1, 'Spiker_vs_Stable'), (0, 2, 'Spiker_vs_HP'), (1, 2, 'Stable_vs_HP')]:
        _, pp = mannwhitneyu(groups[a], groups[b], alternative='two-sided')
        pw[nm + '_p'] = pp
        pw[nm + '_bonf'] = min(1.0, pp * 3)
    pw_rows.append(pw)

res = pd.DataFrame(rows)
res['fdr_p'] = multipletests(res['kruskal_p'], method='fdr_bh')[1]
res['fdr_sig'] = np.where(res.fdr_p < 0.001, '***',
                  np.where(res.fdr_p < 0.01, '**',
                   np.where(res.fdr_p < 0.05, '*', 'ns')))
pwd = pd.DataFrame(pw_rows)

res.to_csv(f'{BASE}/logs/characterization_fitforpurpose_n1306{SUFFIX}.csv', index=False)
pwd.to_csv(f'{BASE}/logs/characterization_fitforpurpose_n1306{SUFFIX}_pairwise.csv', index=False)

pd.set_option('display.width', 250)
print('\n=== FIT-FOR-PURPOSE 15-biomarker characterization ===')
print(res[['label', 'cohort', 'n_total', 'med_Spiker', 'med_Stable', 'med_HP',
           'eta2', 'effect_size', 'fdr_p', 'fdr_sig']].to_string(index=False))
print('\n=== pairwise (Bonferroni x3) — significant markers only ===')
sigv = set(res.loc[res.fdr_sig != 'ns', 'variable'])
print(pwd[pwd.variable.isin(sigv)][['label', 'Spiker_vs_Stable_bonf',
                                    'Spiker_vs_HP_bonf', 'Stable_vs_HP_bonf']].to_string(index=False))

n_sig = int((res.fdr_sig != 'ns').sum())
print(f'\nSIGNIFICANT after BH-FDR: {n_sig}/15')
print('  ' + ', '.join(res.loc[res.fdr_sig != 'ns', 'label']))

# ---- markdown companion
with open(f'{BASE}/logs/characterization_fitforpurpose_n1306{SUFFIX}.md', 'w') as f:
    f.write('# Biomarker characterization, 15 analytes\n\n')
    f.write(f'Labels {LABCOL} '
            f'({" / ".join(f"{k} {v:,}" for k, v in N_OBS.items())}). '
            f'Fasting-dependent markers evaluated on the >={FAST_MIN} h self-report-fasted subset '
            f'(n={n_fast}, {100*n_fast/len(df):.0f}% of cohort); fasting-independent markers on the '
            f'full cohort. KW across 3 phenotypes, BH-FDR within the 15-marker family, '
            f'eta-squared effect size.\n\n')
    f.write(res[['label', 'cohort', 'n_total', 'n_Spiker', 'n_Stable', 'n_HP',
                 'med_Spiker', 'med_Stable', 'med_HP', 'eta2', 'effect_size',
                 'fdr_p', 'fdr_sig']].to_markdown(index=False))
    f.write('\n\n## Pairwise (Bonferroni x3)\n\n')
    f.write(pwd[['label', 'Spiker_vs_Stable_bonf', 'Spiker_vs_HP_bonf',
                 'Stable_vs_HP_bonf']].to_markdown(index=False))
    f.write(f'\n\n**{n_sig}/15 significant after BH-FDR.**\n')

print(f'\nWROTE logs/characterization_fitforpurpose_n1306{SUFFIX}.{{csv,md}} + _pairwise.csv')
