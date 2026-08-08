"""Measured behaviour and body size across the three patterns.

The patterns are derived from glucose alone, so the obvious alternative explanation is that
they are a proxy for how people live: more active people, better sleepers, leaner people. This
tests that directly on everything the cohort actually measures — ten device-derived
activity, sleep and stress features from the wearable, plus two anthropometric measures.

Two tests per feature, each with its own Benjamini-Hochberg family:

    Kruskal-Wallis across all three patterns   the three-group view
    Mann-Whitney, Spiker versus Stable         the two poles of the hyperglycemia axis

Both are reported because they answer different questions and a reader can reasonably ask for
either. Running them as separate FDR families rather than one pooled family of 24 is the
conservative choice for a null result: pooling would shrink every p-value's correction burden
by half relative to the family it belongs to.

Diet is not tested. The cohort has no timestamped food log, so there is no dietary exposure to
test and its absence is a stated limitation rather than a null result.

Stress is the device-derived index from the wearable, not the self-report scale.

in   processed/analysis_table_n1306_n14.csv
out  logs/behaviour_anthropometry_tests.{md,csv}
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

BASE = os.environ.get("AIREADI_DATA_ROOT", "")
assert BASE, "set AIREADI_DATA_ROOT to the data root"
BR = f'{BASE}/canonical_n14_rerun'
LOGD = f'{BR}/logs'
os.makedirs(LOGD, exist_ok=True)

PAT = {0: 'Spiker', 1: 'Stable', 2: 'Hypo-Prone'}
ORDER = ['Spiker', 'Stable', 'Hypo-Prone']

# (column, label, domain). Ten wearable features + two anthropometric.
FEATURES = [
    ('avg_daily_steps',      'Daily steps',            'Activity'),
    ('avg_active_min',       'Active min/day',         'Activity'),
    ('avg_sedentary_min',    'Sedentary min/day',      'Activity'),
    ('avg_total_sleep_hr',   'Sleep duration',         'Sleep'),
    ('sleep_efficiency_pct', 'Sleep efficiency',       'Sleep'),
    ('deep_sleep_pct',       'Deep sleep %',           'Sleep'),
    ('rem_sleep_pct',        'REM sleep %',            'Sleep'),
    ('light_sleep_pct',      'Light sleep %',          'Sleep'),
    ('avg_mean_stress',      'Stress, mean',           'Stress'),
    ('avg_max_stress',       'Stress, peak',           'Stress'),
    ('bmi',                  'BMI',                    'Body size'),
    ('whr',                  'Waist-to-hip ratio',     'Body size'),
]

RET = []
def rec(s=''):
    RET.append(s); print(s, flush=True)


def eta_squared(H, n, k):
    """Eta-squared for Kruskal-Wallis: (H - k + 1) / (n - k). Floored at 0."""
    return max(0.0, (H - k + 1) / (n - k))


def band(e):
    return 'large' if e >= 0.14 else 'medium' if e >= 0.06 else 'small' if e >= 0.01 else 'negligible'


# ---------------------------------------------------------------- load
df = pd.read_csv(f'{BR}/processed/analysis_table_n1306_n14.csv')
assert df.person_id.is_unique, 'duplicate person_id in the analysis table'
vc = df['hypo_k3_pruned'].value_counts().to_dict()
assert len(vc) == 3, 'expected three pattern labels'
df['pat'] = df['hypo_k3_pruned'].map(PAT)
missing = [c for c, _, _ in FEATURES if c not in df.columns]
assert not missing, f'features absent from the analysis table: {missing}'

rec(f'cohort n={len(df):,}  ·  ' + '  '.join(f'{PAT[g]} {vc[g]}' for g in (0, 1, 2)))
rec(f'testing {len(FEATURES)} features: '
    f'{sum(1 for _, _, d in FEATURES if d != "Body size")} wearable + '
    f'{sum(1 for _, _, d in FEATURES if d == "Body size")} anthropometric')
rec()

# ---------------------------------------------------------------- tests
# Complete-case per feature: a participant missing a wearable stream is dropped from that
# feature's test only, not from the others. Coverage differs by stream, so a listwise drop
# would silently test a different, smaller cohort than the one described.
rows = []
for col, label, dom in FEATURES:
    s = df.dropna(subset=[col])
    groups = [s.loc[s.pat == g, col].values for g in ORDER]
    ns = [len(x) for x in groups]
    assert min(ns) > 0, f'{col}: a pattern has no observations'

    H, p_kw = kruskal(*groups)
    e2 = eta_squared(H, len(s), 3)
    U, p_mwu = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
    # rank-biserial correlation: the Spiker-vs-Stable effect size on the same scale as the test
    rbc = 1 - (2 * U) / (ns[0] * ns[1])

    rows.append(dict(variable=col, label=label, domain=dom, n_total=len(s),
                     n_Spiker=ns[0], n_Stable=ns[1], n_HP=ns[2],
                     med_Spiker=float(np.median(groups[0])),
                     med_Stable=float(np.median(groups[1])),
                     med_HP=float(np.median(groups[2])),
                     kruskal_H=H, kruskal_p=p_kw, eta2=e2, effect_size=band(e2),
                     mwu_p=p_mwu, rank_biserial=rbc))

res = pd.DataFrame(rows)
res['kw_fdr_p'] = multipletests(res['kruskal_p'], method='fdr_bh')[1]
res['mwu_fdr_p'] = multipletests(res['mwu_p'], method='fdr_bh')[1]
res['kw_sig'] = np.where(res.kw_fdr_p < 0.05, '*', 'ns')
res['mwu_sig'] = np.where(res.mwu_fdr_p < 0.05, '*', 'ns')

# ---------------------------------------------------------------- report
pd.set_option('display.width', 250)
rec('=== Kruskal-Wallis across the three patterns (BH-FDR within these '
    f'{len(FEATURES)} features) ===')
rec('  ' + res[['label', 'domain', 'n_total', 'med_Spiker', 'med_Stable', 'med_HP',
                'eta2', 'effect_size', 'kw_fdr_p', 'kw_sig']]
    .round(4).to_string(index=False).replace('\n', '\n  '))
rec()
rec('=== Mann-Whitney, Spiker vs Stable (its own BH-FDR family) ===')
rec('  ' + res[['label', 'domain', 'n_Spiker', 'n_Stable', 'rank_biserial',
                'mwu_fdr_p', 'mwu_sig']]
    .round(4).to_string(index=False).replace('\n', '\n  '))
rec()

n_kw = int((res.kw_fdr_p < 0.05).sum())
n_mwu = int((res.mwu_fdr_p < 0.05).sum())
rec(f'SIGNIFICANT after BH-FDR — Kruskal-Wallis {n_kw}/{len(FEATURES)}  ·  '
    f'Mann-Whitney (Spiker vs Stable) {n_mwu}/{len(FEATURES)}')
if n_kw:
    rec('  KW: ' + ', '.join(res.loc[res.kw_fdr_p < 0.05, 'label']))
if n_mwu:
    rec('  MWU: ' + ', '.join(res.loc[res.mwu_fdr_p < 0.05, 'label']))
rec(f'largest effect size across all {len(FEATURES)} features: '
    f'eta2 = {res.eta2.max():.4f} ({res.loc[res.eta2.idxmax(), "label"]}, '
    f'{band(res.eta2.max())}); '
    f'largest |rank-biserial| = {res.rank_biserial.abs().max():.3f} '
    f'({res.loc[res.rank_biserial.abs().idxmax(), "label"]})')
rec('Smallest per-feature n is the binding constraint on power, not the cohort size: '
    f'n ranges {res.n_total.min():,}-{res.n_total.max():,} across features.')

# ---------------------------------------------------------------- outputs
res.to_csv(f'{LOGD}/behaviour_anthropometry_tests.csv', index=False)
with open(f'{LOGD}/behaviour_anthropometry_tests.md', 'w') as f:
    f.write('# Behaviour and body size across the three patterns\n\n')
    f.write(f'Device-derived activity, sleep and stress from the wearable plus two '
            f'anthropometric measures ({len(FEATURES)} features). Kruskal-Wallis across the '
            f'three patterns and Mann-Whitney between Spiker and Stable, each with its own '
            f'Benjamini-Hochberg family. Complete-case per feature. Diet is not tested: the '
            f'cohort has no timestamped food log.\n\n')
    f.write(res[['label', 'domain', 'n_total', 'n_Spiker', 'n_Stable', 'n_HP',
                 'med_Spiker', 'med_Stable', 'med_HP', 'eta2', 'effect_size',
                 'kw_fdr_p', 'kw_sig', 'rank_biserial', 'mwu_fdr_p', 'mwu_sig']]
            .round(4).to_markdown(index=False))
    f.write(f'\n\n**Kruskal-Wallis {n_kw}/{len(FEATURES)} significant after BH-FDR; '
            f'Mann-Whitney {n_mwu}/{len(FEATURES)}.**\n')

print(f'\nWROTE {LOGD}/behaviour_anthropometry_tests.{{md,csv}}')
