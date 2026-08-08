"""Circadian ANCOVA: nocturnal, dawn and amplitude contrasts by pattern.

Builds per-participant circadian features from the cleaned CGM series on local time, then
fits pattern contrasts adjusted for mean glucose, so each estimate is a shape difference
rather than a level difference.

in   cgm_all_readings_clean_final.csv, analysis table, labels
out  processed/circadian_features_n14.csv, ANCOVA beta table, log
"""
import re
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

BASE = os.environ.get("AIREADI_DATA_ROOT", "")
assert BASE, "set AIREADI_DATA_ROOT to the data root"
RAW  = os.path.join(BASE, "cgm_all_readings_clean_final.csv")
LAB  = os.path.join(BASE, "canonical_n14_rerun/labels/hypo_clustering_n14_n1306.csv")
AT   = os.path.join(BASE, "canonical_n14_rerun/processed/analysis_table_n1306_n14.csv")
OUT_FEAT  = os.path.join(BASE, "canonical_n14_rerun/processed/circadian_features_n14.csv")
OUT_BETAS = os.path.join(BASE, "canonical_n14_rerun/logs/circadian_signature_n14_betas.csv")
OUT_MD    = os.path.join(BASE, "logs/circadian_signature_pruned.md")

UTC_OFFSET = {"pst": -8, "cst": -6}
NAME = {0: "Spiker", 1: "Stable", 2: "HP"}

# ---------------------------------------------------------------- labels + tz
lab = pd.read_csv(LAB, usecols=["person_id", "hypo_k3_pruned"])
meta = pd.read_csv(AT, usecols=["person_id", "timezone"]).merge(lab, on="person_id", how="right")
tz_map = dict(zip(meta.person_id, meta.timezone.fillna("pst").str.lower()))
pids = set(meta.person_id)
vc = meta.hypo_k3_pruned.value_counts().to_dict()
print(f"[load] PRUNED cohort n={len(meta)}  groups={{Spiker:{vc.get(0)}, Stable:{vc.get(1)}, HP:{vc.get(2)}}}")
assert len(vc) == 3, "expected three pattern labels"

# ---------------------------------------------------------------- stream raw CGM
print("[load] streaming clean raw CGM (filtered to cohort)...")
chunks = []
for ch in pd.read_csv(RAW, usecols=["participant_id", "start_datetime", "glucose_value_mg_dL"],
                      chunksize=1_000_000):
    chunks.append(ch[ch.participant_id.isin(pids)])
cgm = pd.concat(chunks, ignore_index=True).dropna(subset=["glucose_value_mg_dL"])
del chunks
cgm["t"] = pd.to_datetime(cgm.start_datetime, utc=True, format="ISO8601")
off = cgm.participant_id.map(tz_map).fillna("pst").map(UTC_OFFSET).fillna(-8).astype(int)
cgm["local_hour"] = (cgm.t.dt.hour + off) % 24
cgm["g"] = cgm.glucose_value_mg_dL.astype(float)
print(f"[load] {len(cgm):,} readings, {cgm.participant_id.nunique()} pids")

# ---------------------------------------------------------------- per-person features (== script 63)
rows = []
for pid, gdf in cgm.groupby("participant_id"):
    g = gdf.g.values; h = gdf.local_hour.values
    mean24 = g.mean()
    hm = np.array([g[h == k].mean() if (h == k).any() else np.nan for k in range(24)])
    night = g[(h >= 0) & (h < 6)]
    day   = g[(h >= 8) & (h < 22)]
    noct_mean = night.mean() if len(night) else np.nan
    day_mean  = day.mean()   if len(day)   else np.nan
    nadir_block = np.nanmin(hm[0:6]) if np.isfinite(hm[0:6]).any() else np.nan
    morning     = np.nanmean(hm[6:9])
    dawn_delta  = morning - nadir_block
    amp = np.nanmax(hm) - np.nanmin(hm)
    rows.append(dict(person_id=pid, mean24=mean24, noct_mean=noct_mean, day_mean=day_mean,
                     day_night_diff=day_mean - noct_mean, dawn_delta=dawn_delta, amp=amp,
                     n_read=len(g)))
feat = pd.DataFrame(rows).merge(meta, on="person_id", how="left")
feat["group"] = feat.hypo_k3_pruned.map(NAME)
feat.to_csv(OUT_FEAT, index=False)
print(f"[save] {OUT_FEAT}  ({len(feat)} rows)")

# ---------------------------------------------------------------- ANCOVA: beyond burden
two = feat[feat.hypo_k3_pruned.isin([0, 1])].copy()
two["is_spiker"] = (two.hypo_k3_pruned == 0).astype(int)
SP, ST, HP = (feat[feat.hypo_k3_pruned == k] for k in (0, 1, 2))

METRICS = [
    ("noct_mean", "Nocturnal mean 00-06 (mg/dL)"),
    ("dawn_delta", "Dawn rise (mg/dL)"),
    ("amp", "Circadian amplitude (mg/dL)"),
    ("day_mean", "Daytime mean 08-22 (mg/dL)"),
    ("day_night_diff", "Day-night diff (mg/dL)"),
]
# Multiple testing is corrected over these three contrasts and no others. Stating the family
# explicitly here is what fixes the correction: adding a metric to the list below changes every
# adjusted p-value in the output.
PRIMARY = ["noct_mean", "dawn_delta", "amp"]
anc_rows = []
for col, lab_ in METRICS:
    d = two[[col, "is_spiker", "mean24"]].dropna()
    m0 = smf.ols(f"{col} ~ is_spiker", data=d).fit()
    m1 = smf.ols(f"{col} ~ is_spiker + mean24", data=d).fit()
    anc_rows.append(dict(metric=col, label=lab_,
        spiker_mean=SP[col].mean(), stable_mean=ST[col].mean(), hp_mean=HP[col].mean(),
        beta_unadj=m0.params["is_spiker"], p_unadj=m0.pvalues["is_spiker"],
        beta_adj=m1.params["is_spiker"], p_adj=m1.pvalues["is_spiker"]))
anc = pd.DataFrame(anc_rows)
anc["family"] = np.where(anc.metric.isin(PRIMARY), "primary", "supplementary")

# TWO explicit families. The old single `p_adj_fdr` column is deliberately NOT written: its name
# did not say which family it belonged to, which is how the figure and this log came to disagree.
prim = anc.family == "primary"
anc["p_fdr_primary3"] = np.nan
anc.loc[prim, "p_fdr_primary3"] = multipletests(anc.loc[prim, "p_adj"], method="fdr_bh")[1]
anc["p_fdr_all5"] = multipletests(anc.p_adj, method="fdr_bh")[1]
anc.to_csv(OUT_BETAS, index=False)
print(f"[save] {OUT_BETAS}")

print("\n[FDR reconciliation] primary-3 = what figure 2 panel c and the README report;"
      " all-5 = exploratory, includes the two excluded metrics")
for _, r in anc.iterrows():
    p3 = "     —    " if np.isnan(r.p_fdr_primary3) else f"{r.p_fdr_primary3:.2e}"
    print(f"  {r.label:34s} {r.family:13s} p_fdr_primary3={p3}  p_fdr_all5={r.p_fdr_all5:.2e}")

with open(OUT_MD, "w") as f:
    f.write(f"# Circadian ANCOVA betas ({'/'.join(str(int(vc[i])) for i in sorted(vc.index))})\n\n")
    f.write("AI-READI v3.0.0; Spiker vs Stable, adjusted for 24h mean glucose; HP excluded "
            f"(n={len(HP)} descriptive). Mirrors scripts/63 on pruned labels.\n\n")
    f.write(f"Groups: Spiker {len(SP)} · Stable {len(ST)} · HP {len(HP)}.\n\n")
    f.write("FDR families are SEPARATE and both reported. `p_fdr_primary3` covers the three metrics "
            "figure 2 panel c plots (nocturnal mean, dawn rise, amplitude) and is the value to quote. "
            "`p_fdr_all5` additionally includes daytime mean and day-night diff, which are excluded "
            "from the primary family because day_night_diff is itself a clustering feature and the "
            "two means are components of the mean24 covariate. No verdict differs between families.\n\n")
    f.write("| metric | family | Spiker | Stable | HP | beta(unadj) | p(unadj) | **beta(adj)** | "
            "**p(adj)** | p_fdr_primary3 | p_fdr_all5 |\n")
    f.write("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n")
    for _, r in anc.iterrows():
        p3 = "—" if np.isnan(r.p_fdr_primary3) else f"{r.p_fdr_primary3:.2e}"
        f.write(f"| {r.label} | {r.family} | {r.spiker_mean:.2f} | {r.stable_mean:.2f} | "
                f"{r.hp_mean:.2f} | {r.beta_unadj:+.3f} | {r.p_unadj:.2e} | "
                f"**{r.beta_adj:+.3f}** | **{r.p_adj:.2e}** | {p3} | {r.p_fdr_all5:.2e} |\n")
    f.write("\nPanel-c headline (pruned): nocturnal beta_adj = "
            f"{anc.loc[anc.metric=='noct_mean','beta_adj'].iloc[0]:+.2f}, dawn = "
            f"{anc.loc[anc.metric=='dawn_delta','beta_adj'].iloc[0]:+.2f}, amplitude = "
            f"{anc.loc[anc.metric=='amp','beta_adj'].iloc[0]:+.2f}\n")
print(f"[save] {OUT_MD}")

print("\n[BETAS adjusted for 24h mean] (Spiker vs Stable):")
for _, r in anc.iterrows():
    print(f"  {r.label:34s} beta_adj={r.beta_adj:+7.3f}  p_adj={r.p_adj:.2e}")
print("\n[done]")
