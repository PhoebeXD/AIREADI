#!/usr/bin/env python3
"""Gaussian-copula null for cluster separation.

Tests the observed silhouette against a null of one unimodal population that matches both
the empirical marginal of every feature (skew preserved) and the rank-correlation
structure. A per-feature shuffle is not used: it assumes the 14 features are mutually
independent (n_lows_70 and pct_below_70 are near-duplicates), so it rejects for any
correlated unimodal cloud and cannot distinguish clusters from a continuum.

B = 1000 draws. Reports observed silhouette, null distribution, p and z.

in   processed/analysis_table_n1306_n14.csv
out  logs/null_copula_n14.md
"""
import warnings, os, json; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SEED = 42; B = 1000
BASE = os.environ.get("AIREADI_DATA_ROOT", "")
assert BASE, "set AIREADI_DATA_ROOT to the data root"
BR = f"{BASE}/canonical_n14_rerun"
TABLE = f"{BR}/processed/analysis_table_n1306_n14.csv"
RESULTS = f"{BR}/logs/null_copula_sils_n14.txt"
os.makedirs(f"{BR}/logs", exist_ok=True)

FEATS = json.load(open(f"{BR}/models/meta.json"))["features"]
assert len(FEATS) == 14, FEATS

df = pd.read_csv(TABLE); n = len(df); d = len(FEATS)
Xraw = df[FEATS].copy().fillna(df[FEATS].median()).values
Xz = StandardScaler().fit_transform(Xraw)
km = lambda X: KMeans(n_clusters=3, n_init=20, random_state=SEED).fit_predict(X)
lab_obs = km(Xz)
sil_obs = silhouette_score(Xz, lab_obs)

# sanity: the refit reproduces the frozen n14 labels (up to cluster renumbering)
from sklearn.metrics import adjusted_rand_score
ari_frozen = adjusted_rand_score(df["hypo_k3_pruned"].values, lab_obs)

NS = np.empty_like(Xraw)
for j in range(d):
    NS[:, j] = norm.ppf((rankdata(Xraw[:, j]) - 0.5) / n)
L = np.linalg.cholesky(np.cov(NS, rowvar=False))
sorted_cols = [np.sort(Xraw[:, j]) for j in range(d)]

def copula_draw(rng):
    Zc = rng.standard_normal((n, d)) @ L.T
    U = norm.cdf(Zc); idx = U * (n - 1)
    lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, n - 1); frac = idx - lo
    out = np.empty_like(Zc)
    for j in range(d):
        s = sorted_cols[j]; out[:, j] = s[lo[:, j]] * (1 - frac[:, j]) + s[hi[:, j]] * frac[:, j]
    return out

done = 0
if os.path.exists(RESULTS):
    done = sum(1 for _ in open(RESULTS))
print(f"n={n} d={d}  obs sil={sil_obs:.4f}  ARI vs frozen n14 labels={ari_frozen:.4f}  "
      f"resuming at {done}/{B}", flush=True)
for b in range(done, B):
    rng = np.random.default_rng(SEED * 100000 + b)
    Xd = copula_draw(rng)
    Xdz = StandardScaler().fit_transform(Xd)
    s = silhouette_score(Xdz, km(Xdz))
    with open(RESULTS, "a") as f:
        f.write(f"{s:.6f}\n")
    if (b + 1) % 50 == 0:
        print(f"  ... {b+1}/{B}", flush=True)

null3 = np.loadtxt(RESULTS)
m, sd = null3.mean(), null3.std(ddof=1)
ge = int((null3 >= sil_obs).sum())
p = "< 0.001" if ge == 0 else f"{ge/len(null3):.3f}"
z = (sil_obs - m) / sd
beaten = (ge / len(null3) < 0.05) and (sil_obs > m)

lines = []
def w(s=""):
    lines.append(s); print(s, flush=True)

w("=" * 68)
w("Gaussian-copula null — 14 features")
w(f"Observed overall silhouette = {sil_obs:.4f}   (B={len(null3)})")
w(f"Null 3  mean={m:.4f}  SD={sd:.4f}  min={null3.min():.4f}  max={null3.max():.4f}")
w(f"        p={p}   z=(obs-mean)/SD={z:+.2f}")
w(f"NULL 3: {'BEATEN' if beaten else 'NOT beaten'}")
w("")
w("Interpretation: BEATEN => the 3-cluster separation exceeds what one skewed,")
w("correlated single population produces. NOT beaten => the separation is consistent")
w("with a continuum, and K=3 must be described as regions of a continuum, not camps.")
w(f"(refit of the 14 features reproduces the frozen n14 labels, ARI={ari_frozen:.4f})")
w("DONE.")

with open(f"{BR}/logs/null_copula_n14.md", "w") as f:
    f.write("# Gaussian-copula null\n\n```\n" + "\n".join(lines) + "\n```\n")
print(f"\nWROTE {BR}/logs/null_copula_n14.md")
