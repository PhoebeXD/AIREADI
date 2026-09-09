"""PCA of the clustering space, and the Hartigan dip test for multimodality.

Two questions about the same projection:

    1. What do the patterns look like in the space the labels were derived in? The 14 CGM
       features are median-filled, standardized and projected onto two components. PC1 is
       oriented to increase with mean_glucose and PC2 to increase with n_lows_70, so the
       hyperglycemia axis runs left-to-right and the hypoglycemia axis bottom-to-top. The
       orientation is a sign convention on the eigenvectors and changes nothing else.

    2. Is that projection actually two clusters, or one continuum? The Hartigan dip test is
       applied to PC1 over Spiker and Stable, the two poles of the hyperglycemia axis.
       Hypo-Prone is excluded because it is defined on the other axis and would enter the PC1
       distribution as a third, unrelated component. A dip test that does not reject is not
       proof of unimodality, so the log also reports the sample size and the separation
       between the two pattern medians on PC1, which is what the test would have to resolve.

The same projection also gives the diagnostic-stage composition of each pattern: the share of
each pattern that is study_group Healthy versus PreDM, against the cohort baseline. Every
pattern spans both categories, so the patterns are not a restatement of diagnostic stage.

The feature list is read from the fitted model's metadata rather than written out here, so the
projection cannot drift from the space the labels were fitted in.

in   processed/analysis_table_n1306_n14.csv, models/meta.json
out  logs/pca_dip_test.{md,csv}, logs/pca_dip_test_loadings.csv
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import diptest

SEED = 42
np.random.seed(SEED)
N_BOOT = 2000
BASE = os.environ.get("AIREADI_DATA_ROOT", "")
assert BASE, "set AIREADI_DATA_ROOT to the data root"
BR = f'{BASE}/canonical_n14_rerun'
LOGD = f'{BR}/logs'
os.makedirs(LOGD, exist_ok=True)

PAT = {0: 'Spiker', 1: 'Stable', 2: 'Hypo-Prone'}
FEATS = json.load(open(f'{BR}/models/meta.json'))['features']

RET = []
def rec(s=''):
    RET.append(s); print(s, flush=True)

# ---------------------------------------------------------------- load
df = pd.read_csv(f'{BR}/processed/analysis_table_n1306_n14.csv')
assert df.person_id.is_unique, 'duplicate person_id in the analysis table'
vc = df['hypo_k3_n14'].value_counts().to_dict()
assert len(vc) == 3, 'expected three pattern labels'
gid = df['hypo_k3_n14'].values
rec(f'cohort n={len(df):,}  ·  ' + '  '.join(f'{PAT[g]} {vc[g]}' for g in (0, 1, 2)))
rec(f'projection space: {len(FEATS)} CGM features (from models/meta.json)')
rec()

# ---------------------------------------------------------------- PCA
# Same preprocessing as the clustering fit: median fill then standardize. PCA is fitted on the
# whole cohort, not per pattern, so the components are a property of the cohort geometry.
X = df[FEATS].copy()
X = X.fillna(X.median())
Xs = StandardScaler().fit_transform(X.values)
pca = PCA(n_components=2, random_state=SEED).fit(Xs)
PC = pca.transform(Xs)
evr = pca.explained_variance_ratio_ * 100
ld = pca.components_.copy()

# Sign convention only: flip each component so it increases with the feature that names it.
for i, anchor in enumerate(['mean_glucose', 'n_lows_70']):
    if ld[i, FEATS.index(anchor)] < 0:
        PC[:, i] *= -1
        ld[i] *= -1

r_pc1 = np.corrcoef(PC[:, 0], X['mean_glucose'].values)[0, 1]
r_pc2 = np.corrcoef(PC[:, 1], X['n_lows_70'].values)[0, 1]
rec('=== PCA ===')
rec(f'  PC1 explained variance {evr[0]:.1f}%  ·  correlation with mean_glucose r={r_pc1:+.3f}')
rec(f'  PC2 explained variance {evr[1]:.1f}%  ·  correlation with n_lows_70   r={r_pc2:+.3f}')
rec(f'  PC1+PC2 = {evr.sum():.1f}% of the {len(FEATS)}-feature variance')

loadings = pd.DataFrame({'feature': FEATS, 'PC1': ld[0], 'PC2': ld[1]})
for i, pc in enumerate(['PC1', 'PC2']):
    top = loadings.reindex(loadings[pc].abs().sort_values(ascending=False).index).head(5)
    rec(f'  {pc} top loadings: ' + ', '.join(f'{r.feature} {r[pc]:+.2f}' for _, r in top.iterrows()))
rec()

# pattern positions on each axis, for the record
pos = []
for g in (0, 1, 2):
    m = gid == g
    pos.append(dict(pattern=PAT[g], n=int(m.sum()),
                    pc1_median=float(np.median(PC[m, 0])), pc1_iqr_lo=float(np.percentile(PC[m, 0], 25)),
                    pc1_iqr_hi=float(np.percentile(PC[m, 0], 75)),
                    pc2_median=float(np.median(PC[m, 1])), pc2_iqr_lo=float(np.percentile(PC[m, 1], 25)),
                    pc2_iqr_hi=float(np.percentile(PC[m, 1], 75))))
pos = pd.DataFrame(pos)
rec('=== pattern positions in the projection ===')
rec('  ' + pos.round(3).to_string(index=False).replace('\n', '\n  '))
rec()

# ---------------------------------------------------------------- Hartigan dip
# Spiker and Stable only: they are the two poles of the hyperglycemia axis, so if the pattern
# labels marked genuinely separate groups rather than regions of one distribution, PC1 would be
# bimodal over exactly these rows. Hypo-Prone is defined on the hypoglycemia axis and is not
# part of the question PC1 asks.
ss = np.isin(gid, [0, 1])
pc1_ss = PC[ss, 0]
D_ss, P_ss = diptest.diptest(pc1_ss, boot_pval=True, n_boot=N_BOOT, seed=SEED)
D_all, P_all = diptest.diptest(PC[:, 0], boot_pval=True, n_boot=N_BOOT, seed=SEED)

spk_med = float(np.median(PC[gid == 0, 0]))
stb_med = float(np.median(PC[gid == 1, 0]))
gap = spk_med - stb_med
pooled_sd = float(np.std(pc1_ss, ddof=1))

rec('=== Hartigan dip test on PC1 ===')
rec(f'  Spiker + Stable   n={int(ss.sum()):,}   D={D_ss:.4f}   p={P_ss:.3f}   '
    f'({N_BOOT} bootstrap replicates, seed {SEED})')
rec(f'  whole cohort      n={len(df):,}   D={D_all:.4f}   p={P_all:.3f}   (reference)')
rec(f'  pattern medians on PC1: Spiker {spk_med:+.3f} vs Stable {stb_med:+.3f}  '
    f'-> separation {gap:+.3f} = {gap / pooled_sd:.2f} pooled SD')
verdict = ('no evidence of separate modes' if P_ss >= 0.05 else 'multimodality detected')
rec(f'  verdict at alpha=0.05: {verdict}')
rec('  A non-rejection is evidence consistent with one continuous distribution, not proof of '
    'unimodality; read it together with the median separation above.')
rec()

# ---------------------------------------------------------------- diagnostic-stage composition
stage = (df['study_group'].astype(str).str.lower()
         .map(lambda s: 'Healthy' if 'health' in s else ('PreDM' if 'pre' in s else 'other')))
assert (stage == 'other').sum() == 0, 'untreated cohort must be all Healthy/PreDM'
base_healthy = 100 * (stage == 'Healthy').mean()

comp = []
for g in (0, 1, 2):
    m = gid == g
    n = int(m.sum()); nh = int(((stage == 'Healthy').values & m).sum())
    comp.append(dict(pattern=PAT[g], n=n, n_healthy=nh, n_predm=n - nh,
                     pct_healthy=100 * nh / n, pct_predm=100 * (n - nh) / n))
comp = pd.DataFrame(comp)
rec('=== diagnostic-stage composition per pattern ===')
rec('  ' + comp.round(1).to_string(index=False).replace('\n', '\n  '))
rec(f'  cohort baseline Healthy = {base_healthy:.1f}%')
rec(f'  every pattern contains both categories (min pattern share of either category = '
    f'{min(comp.pct_healthy.min(), comp.pct_predm.min()):.1f}%), so the patterns are not a '
    f'relabelling of diagnostic stage.')

# ---------------------------------------------------------------- outputs
summary = pd.DataFrame([
    dict(quantity='pc1_explained_variance_pct', value=evr[0]),
    dict(quantity='pc2_explained_variance_pct', value=evr[1]),
    dict(quantity='pc1_corr_mean_glucose', value=r_pc1),
    dict(quantity='pc2_corr_n_lows_70', value=r_pc2),
    dict(quantity='dip_D_spiker_stable', value=D_ss),
    dict(quantity='dip_p_spiker_stable', value=P_ss),
    dict(quantity='dip_n_spiker_stable', value=int(ss.sum())),
    dict(quantity='dip_D_whole_cohort', value=D_all),
    dict(quantity='dip_p_whole_cohort', value=P_all),
    dict(quantity='pc1_median_separation_sd', value=gap / pooled_sd),
    dict(quantity='baseline_pct_healthy', value=base_healthy),
])
summary.to_csv(f'{LOGD}/pca_dip_test.csv', index=False)
loadings.to_csv(f'{LOGD}/pca_dip_test_loadings.csv', index=False)
comp.to_csv(f'{LOGD}/pca_dip_test_stage_composition.csv', index=False)

with open(f'{LOGD}/pca_dip_test.md', 'w') as f:
    f.write('# PCA of the clustering space and the Hartigan dip test\n\n```\n'
            + '\n'.join(RET) + '\n```\n\n## Component loadings\n\n')
    f.write(loadings.round(3).to_markdown(index=False))
    f.write('\n\n## Pattern positions\n\n')
    f.write(pos.round(3).to_markdown(index=False))
    f.write('\n\n## Diagnostic-stage composition\n\n')
    f.write(comp.round(1).to_markdown(index=False))
    f.write('\n')

print(f'\nWROTE {LOGD}/pca_dip_test.{{md,csv}}')
print(f'WROTE {LOGD}/pca_dip_test_loadings.csv')
print(f'WROTE {LOGD}/pca_dip_test_stage_composition.csv')
