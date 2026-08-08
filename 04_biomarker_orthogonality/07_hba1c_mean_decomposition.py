"""HbA1c as a mean-glucose index: variance explained, axis correlations, matched-mean gap.

Three analyses behind the claim that routine biomarkers track average glycemia rather than
the glucose patterns.

    a  variance in each CGM feature explained by HbA1c (squared Pearson). glucose_sd and
       glucose_cv are reported together: removing the mean collapses the association.
    b  |Spearman rho| of each biomarker against a hyperglycemia axis and a hypoglycemia
       axis, pairwise complete-case. Reports the maximum on each axis and the maximum
       among non-glycemic labs.
    c  Spiker-Stable HbA1c gap, crude and adjusted for mean glucose (OLS), with the share
       of the crude gap attributable to mean glucose. Hypo-Prone is estimated in the same
       model and reported separately.
    d  HbA1c distribution by pattern, with ADA category shares.

in   processed/analysis_table_n1306_n14.csv
out  logs/figure4_panelA_r2.csv, figure4_panelB_rho.csv,
     figure4_panelC_matched_mean.csv, figure4_panelD_hba1c_by_pattern.csv
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_ind
import statsmodels.formula.api as smf

BASE = os.environ.get("AIREADI_DATA_ROOT", "")

assert BASE, "set AIREADI_DATA_ROOT to the data root"
PRUNED = os.path.join(BASE, "canonical_n14_rerun/processed/analysis_table_n1306_n14.csv")
OUTDIR = os.path.join(BASE, "canonical_n14_rerun/plots"); os.makedirs(OUTDIR, exist_ok=True)

# ---- locked pattern colors/names (identical across all npj figures) ----
NAME = {0: "Spiker", 1: "Stable", 2: "Hypo-Prone"}


def panel_tag(ax, s, y=1.075):
    ax.text(-0.16, y, s, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="bottom", ha="left")

# =====================================================================
# LOAD + verify frozen labels
# =====================================================================
pt = pd.read_csv(PRUNED)
vc = pt.hypo_k3_pruned.value_counts().to_dict()
print(f"[labels] PRUNED n={len(pt)}  Spiker:{vc.get(0)} Stable:{vc.get(1)} HP:{vc.get(2)}")
assert len(vc) == 3, 'expected three pattern labels'

# =====================================================================
# PANEL a — R² cliff:  R² = Pearson(HbA1c, feature)**2  (squared Pearson)
# =====================================================================
# the 10 FROZEN pruned clustering features (order = pruned_n10_rerun/scripts/00_refit_pruned_k3.py)
# which broke its argument two ways:
#   (1) pct_above_140 and pct_below_70 are n14 CLUSTERING features but were absent entirely;
#   (2) glucose_sd / glucose_cv were drawn as hatched "derived — not clustering inputs" contrast
#       bars, but on n14 they ARE clustering inputs.
# Panel a now plots all 14 n14 features and DERIVED is empty. The SD-vs-CV bracket is kept as an
# observation (SD carries the mean, CV removes it) but it is no longer an "added contrast" — both
# are genuine members of this feature set.
PRUNED10 = ["mean_glucose","glucose_sd","glucose_cv","mage","pct_above_140","pct_above_180",
            "pct_below_70","pct_below_54","n_lows_70","n_spikes_140",
            "avg_rise_rate","avg_fall_rate","day_night_diff","n_reactive_events"]   # = the n14 set
LBL = {"mean_glucose":"mean glucose","pct_above_180":"% time >180 mg/dL","n_spikes_140":"spike frequency",
       "n_lows_70":"lows <70 (count)","pct_below_54":"% time <54 mg/dL","mage":"MAGE (excursion amp.)",
       "avg_rise_rate":"rise rate","avg_fall_rate":"fall rate","n_reactive_events":"reactive-hypo events",
       "day_night_diff":"day-night difference",
       "pct_above_140":"% time >140 mg/dL","pct_below_70":"% time <70 mg/dL",
       "glucose_sd":"glucose SD","glucose_cv":"glucose CV"}
DERIVED = []   # nothing is "added" any more — all 14 are clustering inputs on this branch

def r2_pearson(df, feat):
    s = df[["hba1c", feat]].dropna()
    return pearsonr(s.hba1c, s[feat])[0] ** 2, len(s)

rows = []
for f in PRUNED10 + DERIVED:
    r2, n = r2_pearson(pt, f)
    rows.append(dict(feature=f, R2=r2, n=n, derived=f in DERIVED))
A = pd.DataFrame(rows)
na = int(A.n.iloc[0])
FOCAL = ["glucose_sd", "glucose_cv"]   # shown ONCE, in the focal block below the separator
ten = (A[~A.feature.isin(FOCAL)].sort_values("R2", ascending=False).reset_index(drop=True))
sd_r2 = float(A.loc[A.feature=="glucose_sd","R2"].iloc[0])
cv_r2 = float(A.loc[A.feature=="glucose_cv","R2"].iloc[0])
print(f"\n[panel a] R² = squared Pearson(HbA1c, feature); complete-case n={na}")
for _, r in ten.iterrows():
    print(f"   {r.feature:<20s} R2={r.R2:.4f}")
print(f"   glucose_sd (derived)  R2={sd_r2:.4f}  (expect ~0.21)")
print(f"   glucose_cv (derived)  R2={cv_r2:.4f}  (expect ~0.058)")
assert abs(sd_r2-0.209)<0.01 and abs(cv_r2-0.058)<0.01, "SD/CV R² did not reproduce"

# =====================================================================
# PANEL b — hyper vs hypo asymmetry: |Spearman rho| best per lab on each axis
# =====================================================================
# branch's own n14 table — the last read in any n14 figure that reached outside the branch.
# Verified before switching: the same person_ids, and all 29 columns this panel touches
# (8 hyper + 9 hypo + 12 labs) are identical to 1e-9, NaN masks included. The check mattered
# because the tz fix moved the night window, so noct_nadir / noct_tbr70 could have shifted —
# they did not. Panel b uses `cl` for correlations only; no cluster label is read from it,
# so the canonical-vs-n14 label difference is irrelevant here.
cl = pd.read_csv(PRUNED)
assert cl.person_id.is_unique, 'duplicate person_id'
HYPER = ["mean_glucose","pct_above_140","pct_above_180","n_spikes_140","avg_spike_peak",
         "mage","glucose_sd","glucose_cv"]
HYPO  = ["n_lows_70","pct_below_70","pct_below_54","avg_nadir_value","n_lows_54",
         "n_reactive_events","reactive_rate","noct_nadir","noct_tbr70"]
GLY   = {"hba1c"}   # fasting_glucose removed (fasting-dependent; cohort not genuinely fasted)
# Fasting-dependent labs EXCLUDED from panel b (labs carry no fasting flag; median 3 h since eating):
# homa_ir_corrected, fasting_glucose, fasting_insulin, hepatic_ir, c_peptide.
# Figure 5b. Triglycerides is the most fasting-sensitive lipid (eta2 0.008 -> 0.000 on fasted
# TG-derived by construction, so all three are postprandially contaminated on a cohort with no
# fasting flag (self-report only, median 3 h since eating).
#
# Restricting them to the >=12 h fasted subset was tried first and REJECTED: that subset is
# small and self-selected, and because best_abs() takes the MAX |rho| over 9 hypo metrics, the
# winning metric ran on only n~98 — where a max-over-9 is strongly upward biased (permutation
# null mean 0.11, p95 0.21). On that subset TG read hypo |rho| 0.249 and TG/HDL 0.242, i.e.
# ABOVE their own hyper coupling, which would have reversed this panel's claim on ~98 people
# with a selection-biased estimator. (TG/TG-HDL did survive a matched permutation null at
# p=0.020/0.012 and LDL did not, p=0.219 — so the fasted-lipid result is real enough to belong
# in a SUPPLEMENT, but not to carry a main-figure title.) Dropping is the same call Figure 5b
# made, and leaves both figures applying one standard.
#
# Fasting-dependent labs now EXCLUDED from panel b, in full: homa_ir_corrected,
# fasting_glucose, fasting_insulin, hepatic_ir, c_peptide, triglycerides, tg_hdl, ldl.
BIO   = ["hba1c","hdl","alt","whr","waist","ast",
         "wbc","crp","urine_albumin","creatinine","total_cholesterol","bmi"]
LABLBL = {"hba1c":"HbA1c","fasting_glucose":"fasting glucose","homa_ir_corrected":"HOMA-IR",
          "hepatic_ir":"hepatic IR","c_peptide":"C-peptide","tg_hdl":"TG/HDL","triglycerides":"triglycerides",
          "fasting_insulin":"fasting insulin","hdl":"HDL","alt":"ALT","whr":"waist-hip ratio",
          "waist":"waist","ast":"AST","wbc":"WBC","crp":"CRP","ldl":"LDL",
          "urine_albumin":"urine albumin","creatinine":"creatinine","total_cholesterol":"total chol.","bmi":"BMI"}
def bio_label(lab):
    """Row label. Every row is now the full untreated cohort — no mixed denominators."""
    return LABLBL[lab]

def best_abs(df, metrics, b):
    best = (np.nan, "")
    for m in metrics:
        s = df[[m, b]].dropna()
        if len(s) >= 30 and s[m].nunique() >= 3:
            rho = spearmanr(s[m], s[b])[0]
            if np.isnan(best[0]) or abs(rho) > abs(best[0]):
                best = (rho, m)
    return best

brows = []
for b in BIO:
    hr, hm = best_abs(cl, HYPER, b)
    or_, om = best_abs(cl, HYPO, b)
    brows.append(dict(lab=b, gly=b in GLY, hyper=abs(hr), hyper_m=hm, hypo=abs(or_), hypo_m=om,
                      n=int(cl[b].notna().sum())))
B = pd.DataFrame(brows).sort_values("hyper", ascending=False).reset_index(drop=True)
b_hyper_max = B.loc[B.hyper.idxmax()]
b_hypo_max  = B.loc[B.hypo.idxmax()]
B_ng = B[~B.gly]
b_hypo_ng_max = B_ng.loc[B_ng.hypo.idxmax()]
print(f"\n[panel b] |Spearman rho|, pairwise complete-case, n_max={len(cl)}, "
      f"{len(BIO)} non-fasting markers (TG family dropped — see note above)")
for _, _r in B.iterrows():
    print(f"   {bio_label(_r.lab):20s} hyper {_r.hyper:.3f} | hypo {_r.hypo:.3f} | n={_r.n}")
print(f"   MAX hyper |rho| = {b_hyper_max.hyper:.3f}  ({b_hyper_max.lab} x {b_hyper_max.hyper_m})")
print(f"   MAX hypo  |rho| (all)     = {b_hypo_max.hypo:.3f}  ({b_hypo_max.lab} x {b_hypo_max.hypo_m})")
print(f"   MAX hypo  |rho| (non-gly) = {b_hypo_ng_max.hypo:.3f}  ({b_hypo_ng_max.lab} x {b_hypo_ng_max.hypo_m})")
print(f"   non-glycemic labs exceeding 0.13 on hypo: {int((B_ng.hypo>0.13).sum())}")

# =====================================================================
# PANEL c — matched-mean HbA1c (Spiker vs Stable only)  [script 26 method]
# =====================================================================
d = pt.dropna(subset=["hba1c","mean_glucose"]).copy()
d["pattern"] = d.hypo_k3_pruned.map(NAME)
sp = d[d.pattern=="Spiker"].hba1c; st = d[d.pattern=="Stable"].hba1c
crude = sp.mean() - st.mean()
crude_se = np.sqrt(sp.var()/len(sp) + st.var()/len(st))
crude_p = ttest_ind(sp, st, equal_var=False).pvalue
d["pat"] = pd.Categorical(d.pattern, categories=["Stable","Spiker","Hypo-Prone"])  # ref=Stable
m1 = smf.ols("hba1c ~ mean_glucose + C(pat)", data=d).fit()
ci = m1.conf_int()
adj   = m1.params["C(pat)[T.Spiker]"]; adj_ci = ci.loc["C(pat)[T.Spiker]"]; adj_p = m1.pvalues["C(pat)[T.Spiker]"]
hp_b  = m1.params["C(pat)[T.Hypo-Prone]"]; hp_p = m1.pvalues["C(pat)[T.Hypo-Prone]"]
n_hp  = int((d.pattern=="Hypo-Prone").sum())
pct_mean = (1 - abs(adj)/abs(crude)) * 100
nc = len(d)
print(f"\n[panel c] matched-mean HbA1c (Spiker vs Stable), complete-case n={nc} "
      f"(Spiker {len(sp)} / Stable {len(st)} / HP {n_hp})")
print(f"   crude gap   = {crude:+.4f} %-pts (95% CI {crude-1.96*crude_se:+.3f},{crude+1.96*crude_se:+.3f}); p={crude_p:.2e}")
print(f"   adjusted    = {adj:+.4f} %-pts (95% CI {adj_ci[0]:+.3f},{adj_ci[1]:+.3f}); p={adj_p:.3f}")
print(f"   % of crude gap attributable to mean glucose = {pct_mean:.0f}%")
print(f"   [footnote] HP-Stable adjusted = {hp_b:+.3f} %-pts, p={hp_p:.4f}, n={n_hp}")

# =====================================================================
# =====================================================================
# Companion to panel c: c asks whether the Spiker-Stable GAP survives mean-adjustment;
# d shows the raw distributions the gap is drawn from, so a reader can see that the
# groups overlap almost completely and that HP is not separable from Stable at all.
ORDER_D = ["Spiker", "Stable", "Hypo-Prone"]
SD_ = {g: pt.loc[pt.hypo_k3_pruned.map(NAME) == g, "hba1c"].dropna().values for g in ORDER_D}
NLAB_D = {g: int((pt.hypo_k3_pruned.map(NAME) == g).sum()) for g in ORDER_D}
ADA = [(0.0, 5.7), (5.7, 6.5), (6.5, np.inf)]      # ADA: normal / prediabetes / diabetes-range
statsD = {}
print("\n[panel d] HbA1c by pattern")
for g in ORDER_D:
    v = SD_[g]
    cats = np.array([100 * ((v >= lo) & (v < hi)).mean() for lo, hi in ADA])
    statsD[g] = dict(n=len(v), mean=v.mean(), sd=v.std(ddof=1), med=np.median(v),
                     p5=np.percentile(v, 5), p95=np.percentile(v, 95),
                     q1=np.percentile(v, 25), q3=np.percentile(v, 75), cats=cats)
    s_ = statsD[g]
    print(f"   {g:11s} n={s_['n']:4d}  mean {s_['mean']:.3f}±{s_['sd']:.3f}  med {s_['med']:.2f}  "
          f"p5-p95 {s_['p5']:.2f}-{s_['p95']:.2f}  ADA {cats[0]:.1f}/{cats[1]:.1f}/{cats[2]:.1f}%")

# ---------------------------------------------------------------- persist panel values
# Every number drawn on this figure, written next to the artwork so the panels can be
# checked, cited and diffed without re-running the script.
LOGDIR = os.path.join(BASE, "canonical_n14_rerun/logs"); os.makedirs(LOGDIR, exist_ok=True)

A.to_csv(f"{LOGDIR}/figure4_panelA_r2.csv", index=False)

B.to_csv(f"{LOGDIR}/figure4_panelB_rho.csv", index=False)

pd.DataFrame([{
    "n": nc, "n_spiker": len(sp), "n_stable": len(st), "n_hp": n_hp,
    "crude_gap": crude, "crude_ci_lo": crude - 1.96 * crude_se,
    "crude_ci_hi": crude + 1.96 * crude_se, "crude_p": crude_p,
    "adjusted_gap": adj, "adjusted_ci_lo": adj_ci[0], "adjusted_ci_hi": adj_ci[1],
    "adjusted_p": adj_p, "pct_gap_from_mean_glucose": pct_mean,
    "hp_stable_adjusted": hp_b, "hp_stable_p": hp_p,
}]).to_csv(f"{LOGDIR}/figure4_panelC_matched_mean.csv", index=False)

pd.DataFrame([{
    "pattern": g, "n": s["n"], "mean": s["mean"], "sd": s["sd"], "median": s["med"],
    "q1": s["q1"], "q3": s["q3"], "p5": s["p5"], "p95": s["p95"],
    "pct_below_5.7": s["cats"][0], "pct_5.7_to_6.4": s["cats"][1],
    "pct_6.5_plus": s["cats"][2],
} for g, s in statsD.items()]).to_csv(
    f"{LOGDIR}/figure4_panelD_hba1c_by_pattern.csv", index=False)

print(f"\n[log] wrote panel values to {LOGDIR}/figure4_panel[A-D]_*.csv")
