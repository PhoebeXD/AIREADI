"""Rate-of-change sensor-artifact flagging for the CGM series.

Flags consecutive-reading deltas > 25 mg/dL (> 5 mg/dL/min at 5-minute sampling) as
physiologically implausible. Readings are flagged, not removed; downstream cleaning
consumes the flag list.

in   cgm_all_readings_deduped.csv, output/hypo_clustering_unmedicated.csv
out  logs/roc_outlier_readings_n1306.csv
"""

import os
import numpy as np
import pandas as pd

BASE = os.environ.get("AIREADI_DATA_ROOT", "")
assert BASE, "set AIREADI_DATA_ROOT to the data root"
CGM_PATH = f"{BASE}/cgm_all_readings_deduped.csv"
LABELS_PATH = f"{BASE}/output/hypo_clustering_unmedicated.csv"
OUT_PATH = f"{BASE}/logs/roc_outlier_readings_n1306.csv"

DELTA_THRESHOLD = 25.0  # mg/dL between consecutive readings


def main():
    print("=" * 60)
    print("RATE-OF-CHANGE OUTLIER FLAGGING")
    print("=" * 60)

    # Phenotype labels (defines the unmedicated cohort and HP membership).
    labels = pd.read_csv(LABELS_PATH, usecols=["person_id", "hypo_k3"])
    cohort_ids = set(labels["person_id"].astype(int))
    hp_ids = set(labels.loc[labels["hypo_k3"] == 2, "person_id"].astype(int))
    print(f"Cohort participants (unmedicated, labelled): {len(cohort_ids)}")
    print(f"  of which Hypo-Prone (hypo_k3==2):          {len(hp_ids)}")

    # Stream the CGM file. We only need 3 columns; cast types up front to
    # keep peak memory manageable on the 451 MB file.
    print(f"\nLoading {CGM_PATH} ...")
    cgm = pd.read_csv(
        CGM_PATH,
        usecols=["participant_id", "start_datetime", "glucose_value_mg_dL"],
        dtype={"participant_id": "int32", "glucose_value_mg_dL": "float32"},
    )
    print(f"  total readings (all participants): {len(cgm):,}")

    # Restrict to the unmedicated cohort up front — we only care about flags
    # in this cohort and it cuts row count substantially before sort/diff.
    cgm = cgm[cgm["participant_id"].isin(cohort_ids)].copy()
    print(f"  readings within unmedicated cohort: {len(cgm):,}")

    # Parse timestamps once; ISO-8601 with 'Z' suffix.
    cgm["ts"] = pd.to_datetime(cgm["start_datetime"], utc=True)

    # Sort by participant then timestamp so groupby().diff() respects time order.
    cgm = cgm.sort_values(["participant_id", "ts"], kind="mergesort")
    cgm = cgm.reset_index(drop=True)

    # Vectorised within-participant delta. The first reading per participant
    # gets NaN (no predecessor) and is correctly excluded by the > threshold
    # comparison (NaN > x → False).
    cgm["delta"] = (
        cgm.groupby("participant_id", sort=False)["glucose_value_mg_dL"]
           .diff()
           .abs()
    )
    cgm["flagged"] = cgm["delta"] > DELTA_THRESHOLD

    # ---- Reporting ------------------------------------------------------
    n_flagged = int(cgm["flagged"].sum())
    n_total = len(cgm)
    pct = 100.0 * n_flagged / n_total if n_total else 0.0

    affected_ids = set(
        cgm.loc[cgm["flagged"], "participant_id"].astype(int).unique().tolist()
    )
    n_affected = len(affected_ids)
    n_hp_affected = len(affected_ids & hp_ids)

    print("\n--- RESULTS ---")
    print(f"Threshold:                |Δ| > {DELTA_THRESHOLD:.0f} mg/dL between consecutive readings")
    print(f"Readings flagged:         {n_flagged:,} / {n_total:,}  ({pct:.4f}%)")
    print(f"Participants affected:    {n_affected} / {len(cohort_ids)}")
    print(f"HP participants affected: {n_hp_affected} / {len(hp_ids)}")

    # Per-participant counts as a quick sanity scan (top 10 worst).
    if n_flagged:
        per_pid = (
            cgm.loc[cgm["flagged"]]
               .groupby("participant_id")
               .size()
               .sort_values(ascending=False)
        )
        print("\nTop 10 participants by flagged-reading count:")
        for pid, n in per_pid.head(10).items():
            tag = " (HP)" if int(pid) in hp_ids else ""
            print(f"  {pid}: {n} flagged{tag}")

    # ---- Save flagged reading rows --------------------------------------
    # We persist enough fields to re-locate each reading later: participant,
    # timestamp, glucose value, the computed delta, and an HP indicator.
    out = cgm.loc[cgm["flagged"], [
        "participant_id", "start_datetime", "glucose_value_mg_dL", "delta"
    ]].copy()
    out["is_hp"] = out["participant_id"].astype(int).isin(hp_ids)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved flagged reading IDs: {OUT_PATH}")
    print(f"  rows: {len(out):,}")


if __name__ == "__main__":
    main()
