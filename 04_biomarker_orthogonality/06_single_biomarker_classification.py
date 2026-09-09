"""Single-biomarker classification of pattern membership.

For each routine marker on its own, and for the five-index insulin-resistance panel,
how well can pattern membership be assigned? Balanced accuracy and macro-OVR AUC over
stratified 5-fold out-of-fold predictions, LogisticRegression and HistGradientBoosting,
2000-resample bootstrap CI. The 14 CGM features are the positive control.

Fit-for-purpose sampling: HbA1c and the CGM control are fasting-independent and use the
full cohort; every insulin-resistance and lipid index uses the >= 12 h fasted subset.

Also reports the Spearman correlation of each marker with the spike-burden axis
(n_spikes_140) and high-IR prevalence by pattern, both on the fasted subset.

Run for K=3 (three patterns) and K=2 (Spiker vs Stable).

in   processed/analysis_table_n1306_n14.csv, clinical_data/observation.csv
out  logs/figure3[_k2]_panelA_balacc.csv, logs/figure3[_k2]_panelB_rho.csv
"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, fisher_exact
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, adjusted_rand_score

SEED = 42
np.random.seed(SEED)
N_BOOT = 2000
BASELINE = 1/3
FAST_MIN = 12          # self-report hours-since-last-ate threshold for "genuinely fasted"

BASE = os.environ.get("AIREADI_DATA_ROOT", "")

assert BASE, "set AIREADI_DATA_ROOT to the data root"
BR  = f'{BASE}/canonical_n14_rerun'
TABLE  = f'{BR}/processed/analysis_table_n1306_n14.csv'
OUTDIR = f'{BASE}/canonical_n14_rerun/plots'
LOGDIR = f'{BR}/logs'
os.makedirs(OUTDIR, exist_ok=True)

# locked palette (match Fig 1/2)
SPIKER, STABLE, HP = '#D55E00', '#0072B2', '#CC79A7'
PAT_COLOR = {0: SPIKER, 1: STABLE, 2: HP}
PAT_NAME  = {0: 'Spiker', 1: 'Stable', 2: 'Hypo-Prone'}
BIO_C    = '#7570B3'    # biomarker bars (purple, as Fig 1 person-level)
CGM_C    = '#1B9E77'    # CGM positive control (teal accent)
GBT_C    = '#B3A2D9'    # lighter purple for the 2nd model

# The CGM positive control must be built from the features the labels were derived from, so this
# is the clustering feature set and nothing else.
CGM14 = ['mean_glucose','glucose_sd','glucose_cv','mage','pct_above_140','pct_above_180',
         'pct_below_70','pct_below_54','n_lows_70','n_spikes_140',
         'avg_rise_rate','avg_fall_rate','day_night_diff','n_reactive_events']
BIO5  = ['hba1c','fasting_glucose','homa_ir_corrected','fasting_insulin','c_peptide']

# ---------------------------------------------------------------- load + verify
df = pd.read_csv(TABLE)
vc = df['hypo_k3_n14'].value_counts().to_dict()
print(f'Loaded n={len(df)}  labels (0/1/2 Spiker/Stable/HP) = '
      f'{vc.get(0)}/{vc.get(1)}/{vc.get(2)}')
assert len(vc) == 3, 'expected three pattern labels'
y_all = df['hypo_k3_n14'].values

# fasting duration — self-report (OMOP observation concept 2005200151, "hours since last ate";
# 1 response/person). Panel b uses only genuinely-fasted individuals (>=12 h); panels a & c full cohort.
_obs = pd.read_csv(f'{BASE}/clinical_data/observation.csv',
                   usecols=['person_id', 'observation_concept_id', 'value_as_number'])
_paate = (_obs[_obs['observation_concept_id'] == 2005200151][['person_id', 'value_as_number']]
          .rename(columns={'value_as_number': 'fast_h'}))
df['person_id'] = df['person_id'].astype(int)
_paate['person_id'] = _paate['person_id'].astype(int)
df = df.merge(_paate, on='person_id', how='left')
assert df.person_id.is_unique, 'fasting merge changed row count'

# ---------------------------------------------------------------- IR index construction
# Unit convention for the stored insulin columns. `insulin` and `fasting_insulin` are stored in
# ng/mL, so conversion to uU/mL is x28.70 (insulin MW 5808; 1 uU/mL = 6 pmol/L), and
# `homa_ir_corrected` -- computed with a x6 constant -- sits 4.783x below true HOMA-IR. The unit
# is established in Methods on three independent lines: the OMOP unit label, shared with
# C-peptide, whose own reference range fixes what that label means; the C-peptide-to-insulin
# molar ratio, which is physiological only under ng/mL; and the absolute insulin and HOMA-IR of
# fasted, lean, normoglycemic participants.
# The rescale is a positive scalar, so every rank-based result here -- Spearman, AUC, balanced
# accuracy, Kruskal-Wallis / Mann-Whitney p, FDR verdict -- is unchanged; only absolutes move.
# FPG/TG/HDL are mg/dL.
NGML_TO_UU = (1e6 / 5808.0) / 6.0     # 28.70 uU/mL per ng/mL
X6_TO_TRUE = NGML_TO_UU / 6.0         # 4.783 -> true HOMA-IR
INS_UU = df['fasting_insulin'] * NGML_TO_UU
FPG, TG, HDL = df['fasting_glucose'], df['triglycerides'], df['hdl']
df['IDX_hba1c']   = df['hba1c']
df['IDX_homa']    = df['homa_ir_corrected'] * X6_TO_TRUE
df['IDX_quicki']  = 1.0 / (np.log10(INS_UU) + np.log10(FPG))
df['IDX_finsulin']= INS_UU
df['IDX_tyg']     = np.log(TG * FPG / 2.0)
df['IDX_tghdl']   = df['tg_hdl']

INDIV = [('HbA1c','IDX_hba1c'), ('HOMA-IR','IDX_homa'), ('QUICKI','IDX_quicki'),
         ('Fasting insulin','IDX_finsulin'), ('TyG','IDX_tyg'), ('TG/HDL','IDX_tghdl')]
PANEL5 = ['IDX_homa','IDX_quicki','IDX_finsulin','IDX_tyg','IDX_tghdl']   # 5 IR indices
# Fasting-dependent indices: any set touching these is evaluated on the >=12 h fasted subset
# only (panels a & c), so a "fasting" input is genuinely a fasting value. HbA1c and the CGM
# control are NOT fasting-dependent -> full cohort.
FASTING_COLS = {'IDX_homa', 'IDX_quicki', 'IDX_finsulin', 'IDX_tyg', 'IDX_tghdl'}

# ---------------------------------------------------------------- CV / boot helpers
def oof_predict(X, y, model):
    """Stratified 5-fold OOF labels + macro-OVR probability matrix (classes sorted)."""
    classes = sorted(np.unique(y))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pred = np.empty(len(y), int)
    prob = np.zeros((len(y), len(classes)))
    for tr, te in skf.split(X, y):
        if model == 'lr':
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=5000, random_state=SEED).fit(sc.transform(X[tr]), y[tr])
            Xte = sc.transform(X[te])
        else:
            clf = HistGradientBoostingClassifier(random_state=SEED).fit(X[tr], y[tr])
            Xte = X[te]
        pred[te] = clf.predict(Xte)
        p = clf.predict_proba(Xte)
        for j, c in enumerate(clf.classes_):
            prob[te, classes.index(c)] = p[:, j]
    return pred, prob, classes

def boot_ci(y, pred, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED)
    n = len(y); accs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        accs[b] = balanced_accuracy_score(y[idx], pred[idx])
    return np.percentile(accs, 2.5), np.percentile(accs, 97.5)

def evaluate(cols, label, classes, fasted=True):
    # classes = K-group set ([0,1,2] for K=3 Spiker/Stable/HP; [0,1] for K=2 Spiker vs Stable).
    # All bars on the same >=12 h fasted sample (fasted=True); CGM control included (fasted=True too).
    base = df[df['fast_h'] >= FAST_MIN] if fasted else df
    base = base[base['hypo_k3_n14'].isin(classes)]
    sub = base.dropna(subset=cols)
    X = sub[cols].values.astype(float); y = sub['hypo_k3_n14'].values
    row = {'set': label, 'n': len(y), 'K': len(classes), 'fasted': fasted}
    for model in ['lr', 'gbt']:
        pred, prob, cls = oof_predict(X, y, model)
        bacc = balanced_accuracy_score(y, pred)
        lo, hi = boot_ci(y, pred)
        # K=2 -> binary AUC (positive class = Stable, col idx 1); K>=3 -> macro-OVR AUC
        auc = (roc_auc_score(y, prob[:, 1]) if len(cls) == 2
               else roc_auc_score(y, prob, multi_class='ovr', average='macro', labels=cls))
        row[f'{model}_bacc'] = bacc; row[f'{model}_lo'] = lo; row[f'{model}_hi'] = hi
        row[f'{model}_auc'] = auc
    return row

# ---------------------------------------------------------------- PC1 continuum axis (compute once)
# PC1 (the continuum axis, loads on mean_glucose) is a SECONDARY reference column for panel b(i);
# the plotted association axis is n_spikes_140 (spike-burden). Computed once on the full cohort.
Xc = df[CGM14].fillna(df[CGM14].median())
Xs = StandardScaler().fit_transform(Xc.values)
pca = PCA(2, random_state=42).fit(Xs); PC = pca.transform(Xs); ld = pca.components_
if ld[0, CGM14.index('mean_glucose')] < 0: PC[:, 0] *= -1
df['PC1'] = PC[:, 0]

HOMA_THRESH = 2.0
SHORT = {0: 'Spiker', 1: 'Stable', 2: 'HP'}


def build(K):
    """Render the orthogonality figure for K=3 (Spiker/Stable/HP) or K=2 (Spiker vs Stable).
    Panels a, b(i), c span all K groups on the same >=12 h fasted sample; b(ii)'s enrichment
    OR is always Spiker-vs-Stable. Writes figure3[_k2].{png,pdf}."""
    CLASSES = [0, 1, 2] if K == 3 else [0, 1]
    BASE_LINE = 1.0 / K
    sfx = '' if K == 3 else '_k2'
    ktag = f'K={K}'
    print(f'\n########################## FIGURE 3  {ktag}  (groups {CLASSES}) ##########################')

    # ---- PANEL A + AUC (panel c): balanced accuracy over the K classes ----
    # Fit-for-purpose sample per marker: HbA1c + CGM are fasting-independent -> FULL cohort
    # (max power for this negative result); every IR/lipid index -> >=12 h fasted subset.
    FASTED_MARKER = {'HbA1c': False}   # default True (fasted) for all other INDIV entries
    rowsA = [evaluate([col], label, CLASSES, fasted=FASTED_MARKER.get(label, True))
             for label, col in INDIV]
    rowsA.append(evaluate(PANEL5, '5 IR-index panel', CLASSES, fasted=True))
    rowsA.append(evaluate(CGM14, f'CGM {len(CGM14)}-feat (control)', CLASSES, fasted=False))
    row6 = evaluate(['IDX_hba1c'] + PANEL5, '6-biomarker panel (ref)', CLASSES)
    print(f'=== PANEL A ({ktag}): balanced accuracy (LR / GBT), chance {BASE_LINE:.3f} ===')
    for r in rowsA:
        print(f"  {r['set']:24s} n={r['n']:5d}  LR bacc={r['lr_bacc']:.3f} "
              f"[{r['lr_lo']:.3f},{r['lr_hi']:.3f}] AUC={r['lr_auc']:.3f}  | "
              f"GBT bacc={r['gbt_bacc']:.3f} AUC={r['gbt_auc']:.3f}")
    print(f"  {row6['set']:24s} n={row6['n']:5d}  LR bacc={row6['lr_bacc']:.3f}  GBT bacc={row6['gbt_bacc']:.3f}")

    # ---- FASTED subset for panel b (all K groups) ----
    df_f = df[(df['fast_h'] >= FAST_MIN) & (df['hypo_k3_n14'].isin(CLASSES))].copy()
    grp_str = ', '.join(f'{SHORT[g]} {int((df_f.hypo_k3_n14 == g).sum())}' for g in CLASSES)
    print(f'=== FASTED SUBSET panel b ({ktag}) n={len(df_f)} ({grp_str}) ===')

    # ---- PANEL B(i): Spearman rho of biomarkers vs spike-burden axis (n_spikes_140) ----
    rho_rows = []
    for label, col in INDIV:
        s = df_f[[col, 'n_spikes_140', 'PC1']].dropna()
        rho, p = spearmanr(s[col], s['n_spikes_140'])
        rpc = spearmanr(s[col], s['PC1'])[0]
        rho_rows.append({'biomarker': label, 'rho_nspikes140': rho, 'p_nspikes140': p,
                         'rho_PC1_ref': rpc, 'n': len(s)})
        print(f"  {label:16s} rho(n_spikes_140)={rho:+.3f} (p={p:.1e})   [PC1 ref {rpc:+.3f}]")
    maxA = max(abs(r['rho_nspikes140']) for r in rho_rows)

    # ---- PANEL B(ii): High-IR prevalence per group; OR/Fisher always Spiker vs Stable ----
    subIR = df_f.dropna(subset=['homa_ir_corrected']).copy()
    subIR['HighIR'] = (subIR['homa_ir_corrected'] > HOMA_THRESH).astype(int)
    n_irpanel = len(subIR)
    prev = subIR.groupby('hypo_k3_n14')['HighIR'].mean() * 100
    grp_n = subIR.groupby('hypo_k3_n14')['HighIR'].size()
    grp_hi = subIR.groupby('hypo_k3_n14')['HighIR'].sum()
    ss = subIR[subIR['hypo_k3_n14'].isin([0, 1])]          # Spiker+Stable enrichment contrast
    pooled_hi = ss['HighIR'].mean() * 100
    ct = pd.crosstab(ss['hypo_k3_n14'], ss['HighIR']).reindex(index=[0, 1], columns=[0, 1]).fillna(0)
    OR_ir, p_ir = fisher_exact([[ct.loc[0, 1], ct.loc[0, 0]], [ct.loc[1, 1], ct.loc[1, 0]]])
    print(f'=== PANEL B(ii) ({ktag}) High-IR (HOMA-IR>{HOMA_THRESH}) n={n_irpanel}  '
          f'Spiker+Stable pooled={pooled_hi:.1f}%  OR={OR_ir:.2f} p={p_ir:.3g} ===')
    for g in CLASSES:
        print(f'    {PAT_NAME[g]:12s} High-IR = {prev.get(g, float("nan")):.1f}%  '
              f'({int(grp_hi.get(g, 0))}/{int(grp_n.get(g, 0))})')

    # ---- logs ----
    pd.DataFrame([{'panel': 'a', **r} for r in rowsA + [row6]]).to_csv(
        f'{LOGDIR}/figure3{sfx}_panelA_balacc.csv', index=False)
    pd.DataFrame(rho_rows).to_csv(f'{LOGDIR}/figure3{sfx}_panelB_rho.csv', index=False)

    return rowsA, row6, rho_rows


build(3)   # three patterns
build(2)   # Spiker vs Stable