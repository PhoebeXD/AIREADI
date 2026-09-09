"""Project the patterns onto the treated cohorts.

Assigns oral-medication (n = 681) and insulin-treated (n = 250) participants to the three
patterns to test how prevalence shifts with treatment intensity.

    1. recompute the clustering features from raw CGM with the same extractor
    2. verify the extractor reproduces the untreated feature table before projecting
    3. apply the fitted scaler and KMeans (nearest centroid) and report prevalence

The scaler and KMeans are refit from the frozen labels rather than loaded from stored
pickles, so the projection always matches the current feature definitions.

in   participants.tsv, raw medicated CGM, analysis table, labels
out  processed/medicated_features_n14.csv, logs/medicated_projection_n14.csv,
     extractor verification CSV
"""
import os
import warnings, json, pickle; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.signal import argrelextrema

BASE = os.environ.get("AIREADI_DATA_ROOT", "")
assert BASE, "set AIREADI_DATA_ROOT to the data root"
NEW=f'{BASE}/canonical_n14_rerun'
F14=['mean_glucose','glucose_sd','glucose_cv','mage','pct_above_140','pct_above_180',
        'pct_below_70','pct_below_54','n_lows_70','n_spikes_140',
        'avg_rise_rate','avg_fall_rate','day_night_diff','n_reactive_events']  # n14
NAME={0:'Spiker',1:'Stable',2:'Hypo-Prone'}

# UAB=cst(-6), UW/UCSD=pst(-8), non-DST.
# Covers all participants (the old `timezone` column covered the untreated cohort only).
_SITE_TZ={'UAB':-6,'UW':-8,'UCSD':-8}
_part=pd.read_csv(f'{BASE}/participants/participants.tsv', sep='\t',
                  usecols=['person_id','clinical_site'])
TZOFF=dict(zip(_part.person_id.astype(int), _part.clinical_site.map(_SITE_TZ)))
assert not any(v!=v for v in TZOFF.values()), 'unmapped clinical_site'

# ---- canonical extractor; see 00_feature_extraction/ for the same features ----
def compute_dynamics(grp):
    grp=grp.sort_values('datetime')
    g=grp['glucose_value_mg_dL'].dropna().values
    if len(g)<100: return None
    result={'mean_glucose':np.mean(g),'glucose_sd':np.std(g),
            'glucose_cv':100*np.std(g)/(np.mean(g)+1e-10),
            'pct_above_140':100*np.mean(g>140),'pct_above_180':100*np.mean(g>180),
            'pct_below_70':100*np.mean(g<70),'pct_below_54':100*np.mean(g<54),
            'n_lows_70':int((g<70).sum()),'n_lows_54':int((g<54).sum())}
    n_spikes=0
    for i in range(len(g)):
        if g[i]>140 and (i==0 or g[i-1]<=140): n_spikes+=1
    result['n_spikes_140']=n_spikes
    diffs=np.diff(g)/5
    result['avg_rise_rate']=np.mean(diffs[diffs>0]) if np.any(diffs>0) else 0
    result['avg_fall_rate']=np.mean(np.abs(diffs[diffs<0])) if np.any(diffs<0) else 0
    if len(g)>20:
        peaks=argrelextrema(g,np.greater,order=6)[0]; troughs=argrelextrema(g,np.less,order=6)[0]
        if len(peaks)>0 and len(troughs)>0:
            extrema=sorted(list(peaks)+list(troughs))
            exc=[abs(g[extrema[i+1]]-g[extrema[i]]) for i in range(len(extrema)-1)]
            large=[e for e in exc if e>np.std(g)]
            result['mage']=np.mean(large) if large else 0
        else: result['mage']=0
    else: result['mage']=0
    # windowing, else "day" = local 00-12 and "night" = local 16-22 for a pst/cst cohort.
    hours=((grp['datetime'].dt.hour.values[:len(g)] + grp['_tzoff'].values[:len(g)]) % 24)
    day=(hours>=8)&(hours<20); night=(hours>=0)&(hours<6)
    result['day_night_diff']=(np.mean(g[day])-np.mean(g[night])) if day.sum()>10 and night.sum()>10 else 0
    window=36; n_reactive=0; i=0
    while i<len(g):
        if g[i]>140:
            while i<len(g)-1 and g[i+1]>=g[i]: i+=1
            end=min(i+window,len(g))
            for j in range(i,end):
                if g[j]<70: n_reactive+=1; i=j; break
        i+=1
    result['n_reactive_events']=n_reactive
    return result

def extract(raw_path, pids=None):
    cols=['participant_id','start_datetime','glucose_value_mg_dL']
    cgm=pd.read_csv(raw_path, usecols=cols)
    if pids is not None: cgm=cgm[cgm['participant_id'].isin(pids)]
    cgm['datetime']=pd.to_datetime(cgm['start_datetime'], utc=True, errors='coerce')
    cgm['_tzoff']=cgm['participant_id'].astype(int).map(TZOFF).fillna(-8).astype(int)
    out=[]
    for pid,grp in cgm.groupby('participant_id'):
        r=compute_dynamics(grp)
        if r is not None: r['person_id']=pid; out.append(r)
    return pd.DataFrame(out)

# ---- 1+2. VERIFY extractor on unmedicated ----
print('[verify] recomputing unmed features from cleaned raw...')
canon=pd.read_csv(f'{BASE}/processed/analysis_table_n1306_clean.csv')
canon_pids=set(canon['person_id'])
unmed_feat=extract(f'{BASE}/cgm_all_readings_clean_final.csv', pids=canon_pids)
m=canon[['person_id']+F14].merge(unmed_feat[['person_id']+F14], on='person_id', suffixes=('_canon','_recomp'))
print(f'  matched {len(m)} unmed PIDs')
vrows=[]
for f in F14:
    r=np.corrcoef(m[f+'_canon'], m[f+'_recomp'])[0,1]
    md=(m[f+'_recomp']-m[f+'_canon']).abs().median()
    vrows.append({'feature':f,'pearson_r':r,'median_abs_diff':md})
    print(f'  {f:18} r={r:.4f}  med|Δ|={md:.4f}')
pd.DataFrame(vrows).to_csv(f'{NEW}/logs/extractor_verification.csv', index=False)
minr=min(v['pearson_r'] for v in vrows)
print(f'  min feature correlation vs canonical: {minr:.4f}')

# ---- 3. extract medicated + project ----
print('\n[project] extracting medicated features...')
allgrp=pd.read_csv(f'{BASE}/output/hypo_clustering_all_n2237.csv')[['person_id','study_group']]
def cohort(s):
    s=str(s).lower()
    return 'Oral' if s.startswith('oral') else ('Insulin' if 'insulin' in s else ('Unmed' if ('health' in s or 'pre' in s) else 'other'))
allgrp['cohort']=allgrp['study_group'].map(cohort)
med_pids=set(allgrp.loc[allgrp['cohort'].isin(['Oral','Insulin']),'person_id'])
med_feat=extract(f'{BASE}/cgm_all_readings_clean_final_medicated.csv', pids=med_pids)
med_feat=med_feat.merge(allgrp[['person_id','cohort']], on='person_id', how='left')
print(f'  extracted {len(med_feat)} medicated PIDs: {med_feat.cohort.value_counts().to_dict()}')
med_feat.to_csv(f'{NEW}/processed/medicated_features_n14.csv', index=False)

# load the n14 model - deliberately NOT the pickles in pruned_n10_rerun/models/,
# which were fit on the UTC-buggy day_night_diff.
N14=f'{BASE}/canonical_n14_rerun'
scaler=pickle.load(open(f'{N14}/models/scaler_n14.pkl','rb'))
km=pickle.load(open(f'{N14}/models/kmeans_n14.pkl','rb'))
remap=json.load(open(f'{N14}/models/meta.json'))['raw_to_canonical']
remap={int(k):int(v) for k,v in remap.items()}

def project(feat_df):
    X=feat_df[F14].fillna(feat_df[F14].median()).values
    raw=km.predict(scaler.transform(X))
    return np.array([remap[r] for r in raw])

# unmed prevalence = the refit itself (sanity)
rows=[]
for coh, sub in [('Oral', med_feat[med_feat.cohort=='Oral']),
                 ('Insulin', med_feat[med_feat.cohort=='Insulin'])]:
    lab=project(sub); n=len(sub)
    cnt={NAME[i]:int((lab==i).sum()) for i in (0,1,2)}
    rows.append({'cohort':coh,'n':n,**{f'{k}_n':v for k,v in cnt.items()},
                 **{f'{k}_pct':round(v/n*100,1) for k,v in cnt.items()}})
# unmed from the n14 refit
pr=pd.read_csv(f'{NEW}/labels/hypo_clustering_n14_n1306.csv')
u=pr['hypo_k3_n14']; n=len(u)
rows.insert(0, {'cohort':'Unmed','n':n,
                **{f'{NAME[i]}_n':int((u==i).sum()) for i in (0,1,2)},
                **{f'{NAME[i]}_pct':round((u==i).mean()*100,1) for i in (0,1,2)}})
res=pd.DataFrame(rows)
res.to_csv(f'{NEW}/logs/medicated_projection_n14.csv', index=False)
print('\n=== F14 K=3 prevalence across medication status ===')
print(res[['cohort','n','Spiker_pct','Stable_pct','Hypo-Prone_pct']].to_string(index=False))
print('\nReference: Spiker 25.8 -> 67.5 -> 85.6% ; HP 3.1/2.6/3.2% (invariant)')
print(f'Saved: {NEW}/logs/medicated_projection_n14.csv, processed/medicated_features_n14.csv')
