"""Validation cascade for the K=3 solution.

Six tests on the 14-feature solution: permutation (1000x), seed stability (20 seeds),
bootstrap (200 resamples), split-half (50 splits), random-forest OOB recovery (500 trees),
and silhouette decomposition by cluster.

in   processed/analysis_table_n1306_n14.csv
out  validation cascade table and plot
"""
import os, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    silhouette_score, silhouette_samples, adjusted_rand_score, confusion_matrix,
)

BASE = os.environ.get("AIREADI_DATA_ROOT", "")
assert BASE, "set AIREADI_DATA_ROOT to the data root"
NEW  = f'{BASE}/canonical_n14_rerun'
TABLE  = f'{NEW}/processed/analysis_table_n1306_n14.csv'
OUT_CSV = f'{NEW}/logs/validation_cascade_n14_n1306.csv'
OUT_PLOT = f'{NEW}/plots/validation_summary_n14_n1306.png'
os.makedirs(f'{NEW}/plots', exist_ok=True)

SEED = 42; N_PERM = 1000; N_SEEDS = 20; N_BOOT = 200; N_SPLITS = 50

FEATURES = ['mean_glucose','glucose_sd','glucose_cv','mage','pct_above_140','pct_above_180',
            'pct_below_70','pct_below_54','n_lows_70','n_spikes_140',
            'avg_rise_rate','avg_fall_rate','day_night_diff','n_reactive_events']  # n14

print('='*60); print('VALIDATION CASCADE'); print('='*60)

df = pd.read_csv(TABLE)
assert len(df) == len(df.person_id.unique()), 'duplicate person_id'
X_raw = df[FEATURES].copy()
X = StandardScaler().fit(X_raw.fillna(X_raw.median()).values).transform(X_raw.fillna(X_raw.median()).values)
y = df['hypo_k3_n14'].values.astype(int)
labname = {0:'Spiker',1:'Stable',2:'Hypo-Prone'}
print(f'\nLoaded n={len(df)}, label counts={dict(zip(*np.unique(y,return_counts=True)))}')

rows = []; t0 = time.time()

# TEST 1 — permutation
print(f'\n[TEST 1] Permutation — {N_PERM} shuffles')
obs_sil = silhouette_score(X, y); rng = np.random.default_rng(SEED)
perm = np.array([silhouette_score(X, rng.permutation(y)) for _ in range(N_PERM)])
p_perm = (np.sum(perm >= obs_sil) + 1) / (N_PERM + 1)
print(f'  observed {obs_sil:.4f}; null mean {perm.mean():.4f} max {perm.max():.4f}; p={p_perm:.4f}')
rows += [{'test':'permutation','metric':'observed_silhouette','value':float(obs_sil)},
         {'test':'permutation','metric':'null_mean','value':float(perm.mean())},
         {'test':'permutation','metric':'null_max','value':float(perm.max())},
         {'test':'permutation','metric':'p_value','value':float(p_perm)}]

# TEST 2 — seed stability
print(f'\n[TEST 2] Seed stability — {N_SEEDS} seeds')
seed_aris = np.array([adjusted_rand_score(y, KMeans(n_clusters=3,n_init=20,random_state=s).fit(X).labels_) for s in range(N_SEEDS)])
print(f'  ARI mean {seed_aris.mean():.4f} ± {seed_aris.std():.4f} [{seed_aris.min():.4f},{seed_aris.max():.4f}]')
rows += [{'test':'seed_stability','metric':'mean_ari','value':float(seed_aris.mean())},
         {'test':'seed_stability','metric':'sd_ari','value':float(seed_aris.std())},
         {'test':'seed_stability','metric':'min_ari','value':float(seed_aris.min())},
         {'test':'seed_stability','metric':'max_ari','value':float(seed_aris.max())}]

# TEST 3 — bootstrap
print(f'\n[TEST 3] Bootstrap — {N_BOOT} samples')
n = len(X); rng = np.random.default_rng(SEED); boot = np.empty(N_BOOT)
for b in range(N_BOOT):
    idx = rng.integers(0, n, size=n)
    km = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(X[idx])
    boot[b] = adjusted_rand_score(y, km.predict(X))
    if (b+1) % 50 == 0: print(f'  {b+1}/{N_BOOT}')
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
print(f'  ARI mean {boot.mean():.4f} CI95 [{ci_lo:.4f},{ci_hi:.4f}] min {boot.min():.4f}')
rows += [{'test':'bootstrap','metric':'mean_ari','value':float(boot.mean())},
         {'test':'bootstrap','metric':'ci95_lo','value':float(ci_lo)},
         {'test':'bootstrap','metric':'ci95_hi','value':float(ci_hi)},
         {'test':'bootstrap','metric':'min_ari','value':float(boot.min())}]

# TEST 4 — split-half
print(f'\n[TEST 4] Split-half — {N_SPLITS} splits')
rng = np.random.default_rng(SEED); sh = np.empty(N_SPLITS)
for k in range(N_SPLITS):
    perm_i = rng.permutation(n); half = n//2
    km = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(X[perm_i[:half]])
    sh[k] = adjusted_rand_score(y[perm_i[half:]], km.predict(X[perm_i[half:]]))
print(f'  ARI mean {sh.mean():.4f} ± {sh.std():.4f} [{sh.min():.4f},{sh.max():.4f}]')
rows += [{'test':'split_half','metric':'mean_ari','value':float(sh.mean())},
         {'test':'split_half','metric':'sd_ari','value':float(sh.std())},
         {'test':'split_half','metric':'min_ari','value':float(sh.min())},
         {'test':'split_half','metric':'max_ari','value':float(sh.max())}]

# TEST 5 — RF OOB
print(f'\n[TEST 5] RF OOB — 500 trees, balanced')
rf = RandomForestClassifier(n_estimators=500, oob_score=True, class_weight='balanced',
                            random_state=SEED, n_jobs=-1).fit(X, y)
oob = rf.oob_score_; oob_pred = np.argmax(rf.oob_decision_function_, axis=1)
recall = {c: float((oob_pred[y==c]==c).mean()) for c in (0,1,2)}
print(f'  OOB acc {oob:.4f}; recall Spk {recall[0]:.3f} Stb {recall[1]:.3f} HP {recall[2]:.3f}')
cm = confusion_matrix(y, oob_pred, labels=[0,1,2])
print(pd.DataFrame(cm, index=[f'true_{labname[c]}' for c in (0,1,2)],
                   columns=[f'pred_{labname[c]}' for c in (0,1,2)]).to_string())
rows += [{'test':'rf_oob','metric':'oob_accuracy','value':float(oob)},
         {'test':'rf_oob','metric':'recall_Spiker','value':recall[0]},
         {'test':'rf_oob','metric':'recall_Stable','value':recall[1]},
         {'test':'rf_oob','metric':'recall_HP','value':recall[2]}]

# TEST 6 — silhouette decomposition
print(f'\n[TEST 6] Silhouette decomposition')
ss = silhouette_samples(X, y); overall = ss.mean()
per = {c: float(ss[y==c].mean()) for c in (0,1,2)}
print(f'  overall {overall:.4f}; Spk {per[0]:.4f} Stb {per[1]:.4f} HP {per[2]:.4f}')
rows += [{'test':'silhouette','metric':'overall','value':float(overall)},
         {'test':'silhouette','metric':'Spiker','value':per[0]},
         {'test':'silhouette','metric':'Stable','value':per[1]},
         {'test':'silhouette','metric':'HP','value':per[2]}]

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(f'\nSaved: {OUT_CSV}')

# plot (6-panel, mirrors canonical)
fig, ax = plt.subplots(2, 3, figsize=(15, 9), dpi=300)
ax[0,0].hist(perm, bins=40, color='lightgray', edgecolor='black'); ax[0,0].axvline(obs_sil, color='red', lw=2, label=f'obs={obs_sil:.3f}')
ax[0,0].set_title(f'Test1 Permutation null (1000x)\np={p_perm:.4f}'); ax[0,0].legend()
ax[0,1].hist(seed_aris, bins=10, color='steelblue', edgecolor='black'); ax[0,1].axvline(seed_aris.mean(), color='red', lw=2)
ax[0,1].set_title(f'Test2 Seed stability\n{seed_aris.mean():.3f}±{seed_aris.std():.3f}')
ax[0,2].hist(boot, bins=30, color='seagreen', edgecolor='black'); ax[0,2].axvline(boot.mean(), color='red', lw=2)
ax[0,2].axvline(ci_lo, color='red', ls='--'); ax[0,2].axvline(ci_hi, color='red', ls='--')
ax[0,2].set_title(f'Test3 Bootstrap (200x)\n{boot.mean():.3f} [{ci_lo:.2f},{ci_hi:.2f}]')
ax[1,0].hist(sh, bins=15, color='goldenrod', edgecolor='black'); ax[1,0].axvline(sh.mean(), color='red', lw=2)
ax[1,0].set_title(f'Test4 Split-half (50)\n{sh.mean():.3f}±{sh.std():.3f}')
cls = ['Spiker','Stable','HP']
b1 = ax[1,1].bar(cls, [recall[0],recall[1],recall[2]], color=['#d95f02','#7570b3','#1b9e77'])
for b,v in zip(b1,[recall[0],recall[1],recall[2]]): ax[1,1].text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.3f}', ha='center')
ax[1,1].axhline(oob, color='red', ls='--', label=f'OOB={oob:.3f}'); ax[1,1].set_ylim(0,1.05); ax[1,1].set_title('Test5 RF OOB recall'); ax[1,1].legend()
b2 = ax[1,2].bar(cls, [per[0],per[1],per[2]], color=['#d95f02','#7570b3','#1b9e77'])
for b,v in zip(b2,[per[0],per[1],per[2]]): ax[1,2].text(b.get_x()+b.get_width()/2, v+0.005, f'{v:.3f}', ha='center')
ax[1,2].axhline(overall, color='red', ls='--', label=f'overall={overall:.3f}'); ax[1,2].axhline(0, color='black', lw=0.5)
ax[1,2].set_title('Test6 Per-phenotype silhouette'); ax[1,2].legend()
plt.suptitle('Validation Cascade', fontsize=14, y=1.0)
plt.tight_layout(); plt.savefig(OUT_PLOT, dpi=300, bbox_inches='tight'); plt.close()
print(f'Saved: {OUT_PLOT}')

print('\n'+'='*60); print('SUMMARY'); print('='*60)
for _name, _val in [('Permutation p', p_perm), ('Seed ARI', seed_aris.mean()),
                    ('Bootstrap ARI', boot.mean()), ('Split-half ARI', sh.mean()),
                    ('RF OOB acc', oob), ('RF HP recall', recall[2]),
                    ('Silhouette overall', overall), ('  HP silhouette', per[2])]:
    print(f'  {_name:28} {_val:.4f}')
print(f'\nTotal time: {time.time()-t0:.1f}s')
