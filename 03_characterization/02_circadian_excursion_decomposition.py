"""Circadian excursion-versus-baseline decomposition, by hour.

Separates the evening (17:00-21:00) glucose difference between Spikers and Stables into a
baseline component and an excursion component, to establish whether the evening pattern is
excursion-driven or a shift in baseline level.

Each participant's series is decomposed into a baseline (centred 3 h rolling median) and an
excursion residual. Both components are burden-adjusted by removing each person's own 24 h
mean, so the hourly contrast is shape, not overall level.

in   cgm_all_readings_clean_final.csv, analysis table, labels
out  logs/circadian_excursion_decomposition.{md,csv}, hourly plot
"""
import os, sys, numpy as np, pandas as pd
from scipy.signal import find_peaks
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = os.environ.get("AIREADI_DATA_ROOT", "")
assert BASE, "set AIREADI_DATA_ROOT to the data root"
sys.path.insert(0, os.path.join(BASE, "spiker_variability_decomp"))
from decomp_lib import to_grid, baseline_excursion

RAW = f"{BASE}/cgm_all_readings_clean_final.csv"
LAB = f"{BASE}/canonical_n14_rerun/labels/hypo_clustering_n14_n1306.csv"
AT  = f"{BASE}/processed/analysis_table_n1306_clean.csv"
LOGMD = f"{BASE}/canonical_n14_rerun/logs/circadian_excursion_decomposition.md"
LOGCSV = f"{BASE}/canonical_n14_rerun/logs/circadian_excursion_decomposition.csv"
PLOTDIR = f"{BASE}/plots/circadian"; os.makedirs(PLOTDIR, exist_ok=True)
CACHE = f"{BASE}/canonical_n14_rerun/processed/_decomp_cache_n14.npz"
UTC_OFFSET = {"pst": -8, "cst": -6}
NAME = {0: "Spiker", 1: "Stable", 2: "HP"}
EVENING = [17, 18, 19, 20]   # 17:00-21:00 local

# ---------------------------------------------------------------- load / compute
lab = pd.read_csv(LAB, usecols=["person_id", "hypo_k3_n14"]).rename(columns={"hypo_k3_n14": "k3"})
tz = pd.read_csv(AT, usecols=["person_id", "timezone"])
meta = lab.merge(tz, on="person_id", how="left")
meta["timezone"] = meta.timezone.fillna("pst").str.lower()
tz_map = dict(zip(meta.person_id, meta.timezone)); g3 = dict(zip(meta.person_id, meta.k3))
pids = set(meta.person_id)
print(f"[load] cohort n={len(meta)}  {meta.k3.map(NAME).value_counts().to_dict()}")

if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    adjbase, exc, rate, size, grp = z["adjbase"], z["exc"], z["rate"], z["size"], z["grp"]
    print(f"[cache] loaded {len(grp)} persons")
else:
    chunks = []
    for ch in pd.read_csv(RAW, usecols=["participant_id", "start_datetime", "glucose_value_mg_dL"],
                          chunksize=1_000_000):
        chunks.append(ch[ch.participant_id.isin(pids)])
    cgm = pd.concat(chunks, ignore_index=True).dropna(subset=["glucose_value_mg_dL"])
    cgm["t"] = pd.to_datetime(cgm.start_datetime, utc=True, format="ISO8601")
    off = cgm.participant_id.map(tz_map).fillna("pst").map(UTC_OFFSET).fillna(-8).astype(int)
    cgm["local_ts"] = cgm.t + pd.to_timedelta(off, unit="h")
    cgm["g"] = cgm.glucose_value_mg_dL.astype(float)
    print(f"[load] {len(cgm):,} readings, {cgm.participant_id.nunique()} pids")

    adjbase, exc, rate, size, grp = [], [], [], [], []
    for pid, gdf in cgm.groupby("participant_id"):
        gdf = gdf.sort_values("local_ts")
        # ---- continuous decomposition on a regular 5-min local grid ----
        s = pd.Series(gdf.g.values, index=gdf.local_ts.values)
        gg = to_grid(s)                                   # 5-min grid
        base, ex = baseline_excursion(gg)
        hr = gg.index.hour
        pmean = np.nanmean(gg.values)                     # person 24h mean glucose
        bm = np.array([np.nanmean(base.values[hr == k]) if (hr == k).any() else np.nan for k in range(24)])
        em = np.array([np.nanmean(ex.values[hr == k]) if (hr == k).any() else np.nan for k in range(24)])
        # burden-adjust: remove person 24h mean of each component
        adjbase.append(bm - np.nanmean(base.values))
        exc.append(em - np.nanmean(ex.values))
        # ---- discrete excursions on raw local-sorted series (existing event def) ----
        g = gdf.g.values; h = gdf.local_hour.values if "local_hour" in gdf else gdf.local_ts.dt.hour.values
        ndays = max(pd.Series(gdf.local_ts.values).dt.normalize().nunique(), 1)
        pk, props = find_peaks(g, height=140, prominence=30)
        ph = h[pk]; prom = props["prominences"]
        rate.append(np.array([(ph == k).sum() for k in range(24)]) / ndays)
        size.append(np.array([prom[ph == k].mean() if (ph == k).any() else np.nan for k in range(24)]))
        grp.append(NAME[g3[pid]])
    adjbase = np.array(adjbase); exc = np.array(exc); rate = np.array(rate); size = np.array(size); grp = np.array(grp)
    np.savez(CACHE, adjbase=adjbase, exc=exc, rate=rate, size=size, grp=grp)
    print(f"[cache] saved {CACHE}")

mask = {g: grp == g for g in ["Spiker", "Stable", "HP"]}
ns = {g: int(mask[g].sum()) for g in mask}

# ---------------------------------------------------------------- per-hour gaps + FDR
def gap_and_p(mat, hr):
    a, b = mat[mask["Spiker"], hr], mat[mask["Stable"], hr]
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    g = np.nanmean(mat[mask["Spiker"], hr]) - np.nanmean(mat[mask["Stable"], hr])
    p = stats.mannwhitneyu(a, b).pvalue if len(a) > 5 and len(b) > 5 else np.nan
    return g, p

rows = []
for k in range(24):
    bg, bp = gap_and_p(adjbase, k)
    eg, ep = gap_and_p(exc, k)
    rows.append({"hour": k,
                 "baseline_Spiker": np.nanmean(adjbase[mask["Spiker"], k]),
                 "baseline_Stable": np.nanmean(adjbase[mask["Stable"], k]),
                 "excursion_Spiker": np.nanmean(exc[mask["Spiker"], k]),
                 "excursion_Stable": np.nanmean(exc[mask["Stable"], k]),
                 "exc_count_Spiker": np.nanmean(rate[mask["Spiker"], k]),
                 "exc_count_Stable": np.nanmean(rate[mask["Stable"], k]),
                 "exc_size_Spiker": np.nanmean(size[mask["Spiker"], k]),
                 "exc_size_Stable": np.nanmean(size[mask["Stable"], k]),
                 "baseline_gap": bg, "baseline_p": bp,
                 "excursion_gap": eg, "excursion_p": ep})
T = pd.DataFrame(rows)
T["baseline_q"] = multipletests(T.baseline_p.fillna(1), method="fdr_bh")[1]
T["excursion_q"] = multipletests(T.excursion_p.fillna(1), method="fdr_bh")[1]
T.to_csv(LOGCSV, index=False)

# ---------------------------------------------------------------- evening attribution
ev = T[T.hour.isin(EVENING)]
# window-level per-person values for MWU
def win(mat):
    return np.nanmean(mat[:, EVENING], axis=1)
ev_base_sp, ev_base_st = win(adjbase)[mask["Spiker"]], win(adjbase)[mask["Stable"]]
ev_exc_sp, ev_exc_st = win(exc)[mask["Spiker"]], win(exc)[mask["Stable"]]
base_gap = np.nanmean(ev_base_sp) - np.nanmean(ev_base_st)
exc_gap = np.nanmean(ev_exc_sp) - np.nanmean(ev_exc_st)
total_gap = base_gap + exc_gap
base_p = stats.mannwhitneyu(ev_base_sp[~np.isnan(ev_base_sp)], ev_base_st[~np.isnan(ev_base_st)]).pvalue
exc_p = stats.mannwhitneyu(ev_exc_sp[~np.isnan(ev_exc_sp)], ev_exc_st[~np.isnan(ev_exc_st)]).pvalue
exc_frac = exc_gap / total_gap if total_gap != 0 else np.nan
base_frac = base_gap / total_gap if total_gap != 0 else np.nan
exc_fdr_sig_evening = bool((ev.excursion_q < 0.05).all() and (ev.excursion_gap > 0).all())

excursion_dominant = abs(exc_gap) > abs(base_gap)
VALIDATED = bool(excursion_dominant and exc_fdr_sig_evening)
verdict = ('VALIDATED ("swing/spiking") — evening Spiker>Stable gap is EXCURSION-driven'
           if VALIDATED else
           'RELABEL ("evening elevation") — evening gap is BASELINE-drift dominated; drop "spiking"')

# ---------------------------------------------------------------- write md
hdr = __doc__
with open(LOGMD, "w") as f:
    f.write(f"# Circadian excursion-vs-baseline decomposition — VERDICT: {verdict}\n\n")
    f.write("```\n" + hdr.strip() + "\n```\n\n")
    f.write(f"Cohort (n14 labels): Spiker {ns['Spiker']} / Stable {ns['Stable']} "
            f"(test); HP {ns['HP']} (caption only).\n\n")
    f.write("## Evening-window (17:00-21:00) attribution of the de-meaned Spiker-Stable gap\n\n")
    f.write(f"- baseline-component gap = **{base_gap:+.2f} mg/dL** (MWU p={base_p:.2e})\n")
    f.write(f"- excursion-component gap = **{exc_gap:+.2f} mg/dL** (MWU p={exc_p:.2e})\n")
    f.write(f"- total de-meaned gap = {total_gap:+.2f} mg/dL  "
            f"-> excursion fraction **{exc_frac:.2f}**, baseline fraction **{base_frac:.2f}**\n")
    f.write(f"- excursion gap FDR-sig & positive across all evening hours: **{exc_fdr_sig_evening}**\n")
    f.write(f"- excursion-dominant (|exc|>|base|): **{excursion_dominant}**\n\n")
    f.write(f"**VERDICT: {verdict}**\n\n")
    f.write("## Hourly table (de-meaned components, mg/dL; discrete excursions/day; size=mean prominence)\n\n")
    f.write(T.round(3).to_string(index=False) + "\n\n")
    f.write("## HP (caption only, not in test)\n")
    for k in EVENING:
        f.write(f"- hour {k}: HP baseline {np.nanmean(adjbase[mask['HP'],k]):+.2f}, "
                f"excursion {np.nanmean(exc[mask['HP'],k]):+.2f}, "
                f"exc/day {np.nanmean(rate[mask['HP'],k]):.2f}\n")
    f.write("\n## Caveats\n- AI-READI has NO timestamped food log; evening eating window is CGM-proxied "
            "(inferred post-dinner), not event-anchored.\n- Local-time alignment (pst-8/cst-6, DST ignored, "
            "matches pipeline).\n- Excursion component = residual from the 3h centered rolling median: "
            "sustained (>~1.5h) evening elevation is absorbed into BASELINE; only sharp sub-3h swings score "
            "as EXCURSION — so the test genuinely separates plateau-drift from spiking.\n")

# ---------------------------------------------------------------- figure
COL = {"Spiker": "#d62728", "Stable": "#1f77b4", "HP": "#2ca02c"}
def band(mat, g):
    sub = mat[mask[g]]; m = np.nanmean(sub, 0)
    n = np.sum(~np.isnan(sub), 0); se = np.nanstd(sub, 0) / np.sqrt(np.maximum(n, 1))
    return m, m - 1.96 * se, m + 1.96 * se
hh = np.arange(24)
fig, (axB, axE) = plt.subplots(1, 2, figsize=(13.5, 5.2))
for ax, mat, ttl, ylab in [(axB, adjbase, "Baseline component (de-meaned)", "baseline − personal 24h mean (mg/dL)"),
                           (axE, exc, "Excursion component (de-meaned)", "mean residual = swing above baseline (mg/dL)")]:
    ax.axvspan(17, 21, color="#fff1c9", zorder=0, label="evening 17–21")
    ax.axvspan(0, 6, color="#f0f0f0", zorder=0)
    for g in ["Spiker", "Stable", "HP"]:
        m, lo, hi = band(mat, g)
        ax.fill_between(hh, lo, hi, color=COL[g], alpha=0.15, lw=0)
        ax.plot(hh, m, "-", color=COL[g], lw=2.3, label=f"{g} (n={ns[g]})")
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xticks([0, 6, 12, 18, 24]); ax.set_xlim(0, 23.5)
    ax.set_xlabel("local hour"); ax.set_ylabel(ylab); ax.set_title(ttl, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle(f"Circadian baseline vs excursion, Spiker vs Stable (n14 labels) — "
             f"evening gap: excursion {exc_gap:+.1f} vs baseline {base_gap:+.1f} mg/dL  →  "
             f"{'SWING validated' if VALIDATED else 'RELABEL: evening elevation'}",
             fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{PLOTDIR}/excursion_by_hour_spiker_vs_stable.png", dpi=140)
fig.savefig(f"{PLOTDIR}/excursion_by_hour_spiker_vs_stable.pdf")
print(f"\nVERDICT: {verdict}")
print(f"evening excursion_gap {exc_gap:+.2f} (p={exc_p:.1e})  baseline_gap {base_gap:+.2f} (p={base_p:.1e})")
print(f"wrote {LOGMD}, {LOGCSV}, plots")
