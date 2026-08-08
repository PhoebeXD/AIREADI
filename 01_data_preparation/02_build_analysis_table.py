"""Build the per-participant analysis table.

Merges phenotype labels, per-person wearable and survey aggregates, and the cleaned
clinical table on person_id.

in   output/hypo_clustering_unmedicated.csv, CSV/lifestyle_all.csv,
     CSV/clinical_merged_clean.csv
out  processed/analysis_table.csv
"""
import os
import pandas as pd

BASE = os.environ.get("AIREADI_DATA_ROOT", "")
assert BASE, "set AIREADI_DATA_ROOT to the data root"
OUT  = f"{BASE}/processed"
os.makedirs(OUT, exist_ok=True)

# 1. Clustered dataset --------------------------------------------------------
clusters = pd.read_csv(f"{BASE}/output/hypo_clustering_unmedicated.csv")
print(f"[1] clusters: {clusters.shape}")
print("    hypo_k3 value counts:")
print(clusters["hypo_k3"].value_counts().sort_index().to_string())

# hypo_cluster_k3 is a DIFFERENT clustering — drop it to prevent confusion
if "hypo_cluster_k3" in clusters.columns:
    n_diff = (clusters["hypo_k3"] != clusters["hypo_cluster_k3"]).sum()
    print(f"    hypo_k3 vs hypo_cluster_k3: {n_diff} rows differ — dropping hypo_cluster_k3")
    clusters = clusters.drop(columns=["hypo_cluster_k3"])
if "hypo_cluster_k2" in clusters.columns:
    clusters = clusters.drop(columns=["hypo_cluster_k2"])

# 2. Lifestyle aggregates -----------------------------------------------------
lifestyle = pd.read_csv(f"{BASE}/CSV/lifestyle_all.csv")
print(f"\n[2] lifestyle_all: {lifestyle.shape}")

# 3. Clinical / study_group ---------------------------------------------------
clinical = pd.read_csv(f"{BASE}/CSV/clinical_merged_clean.csv",
                       usecols=["person_id", "study_group"])
print(f"\n[3] clinical_merged_clean (study_group only): {clinical.shape}")
print("    study_group value counts:")
print(clinical["study_group"].value_counts().to_string())

# 4. Merge --------------------------------------------------------------------
# lifestyle_all already contains the survey fields from lifestyle_survey.csv,
# so we don't need to pull lifestyle_survey.csv again. Avoid column collisions
# on any variable that may also exist in the cluster file (e.g. study_group).
lifestyle_cols_to_keep = [c for c in lifestyle.columns if c not in clusters.columns or c == "person_id"]
df = clusters.merge(lifestyle[lifestyle_cols_to_keep], on="person_id", how="left")
df = df.merge(clinical, on="person_id", how="left", suffixes=("", "_clinical"))

# If clusters already had a study_group column, prefer clinical_merged_clean
if "study_group_clinical" in df.columns:
    df["study_group"] = df["study_group_clinical"]
    df = df.drop(columns=["study_group_clinical"])

print(f"\n[4] merged table: {df.shape}")
print(f"    missing avg_total_sleep_hr (no Garmin sleep): {df['avg_total_sleep_hr'].isna().sum()}")
print(f"    missing avg_mean_stress    (no Garmin stress): {df['avg_mean_stress'].isna().sum()}")
print(f"    missing avg_daily_steps    (no Garmin activity): {df['avg_daily_steps'].isna().sum()}")
print(f"    missing alcohol_ever       (no survey)       : {df['alcohol_ever'].isna().sum()}")
print(f"    missing study_group                            : {df['study_group'].isna().sum()}")

# 5. Cluster label ------------------------------------------------------------
df["cluster_name"] = df["hypo_k3"].map({0: "Spiker", 1: "Stable", 2: "Hypo-Prone"})
print("\n[5] cluster_name value counts:")
print(df["cluster_name"].value_counts().to_string())

# 6. Short study_group labels -------------------------------------------------
sg_map = {
    "healthy":                                                                 "Healthy",
    "pre_diabetes_lifestyle_controlled":                                       "Pre-diabetes",
    "oral_medication_and_or_non_insulin_injectable_medication_controlled":     "Oral medication",
    "insulin_dependent":                                                       "Insulin-dependent",
}
df["study_group_short"] = df["study_group"].map(sg_map)
print("\n[6] study_group_short x cluster_name (row %):")
print(pd.crosstab(df["cluster_name"], df["study_group_short"],
                  normalize="index").round(3).to_string())

# 7. Save ---------------------------------------------------------------------
out_path = f"{OUT}/analysis_table.csv"
df.to_csv(out_path, index=False)
print(f"\n[7] Saved -> {out_path}  ({df.shape[0]} rows x {df.shape[1]} cols)")
