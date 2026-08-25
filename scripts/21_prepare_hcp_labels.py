#!/usr/bin/env python3
"""
Prepare HCP labels (Fluid Intelligence and Working Memory) for packing.
Extracts PMAT24_A_CR (mapped to 'label' for AAAI compatibility) and ListSort_Unadj.
"""
import pandas as pd
from pathlib import Path

BEHAVIOR_PATH = "data/hcp/behavior/unrestricted_behavioral.csv"
OUT_PATH = "data/hcp/processed/labels.csv"

# Internal column names expected by the updated 24_pack_hcp_arrays.py
INTERNAL_TARGETS = ["label", "listsort_unadj"]
# Actual HCP column names in the unrestricted CSV
HCP_COL_MAP = {
    "label": "PMAT24_A_CR",
    "listsort_unadj": "ListSort_Unadj"
}

def main():
    print(f"Loading behavior data from {BEHAVIOR_PATH}...")
    # low_memory=False prevents the DtypeWarning; on_bad_lines='skip' handles messy rows
    beh = pd.read_csv(BEHAVIOR_PATH, low_memory=False, on_bad_lines='skip')
    
    # Find the subject column (case-insensitive)
    subj_col = next((c for c in beh.columns if c.lower() == "subject"), None)
    if not subj_col:
        raise ValueError("Could not find 'Subject' column in behavior CSV.")
        
    beh = beh.set_index(subj_col)
    
    records = []
    dropped_nan = 0
    
    print("Extracting dual targets and applying intersection QC...")
    for sub in beh.index:
        row = {"subject": sub}
        valid = True
        for int_name, hcp_name in HCP_COL_MAP.items():
            if hcp_name not in beh.columns:
                raise KeyError(f"HCP column '{hcp_name}' not found in CSV.")
            
            val = beh.loc[sub, hcp_name]
            
            # Drop if either target is NaN
            if pd.isna(val):
                valid = False
                break
            
            try:
                row[int_name] = float(val)
            except (ValueError, TypeError):
                valid = False
                break
                
        if not valid:
            dropped_nan += 1
            continue
            
        records.append(row)
        
    df_out = pd.DataFrame(records)
    print(f"Processed {len(beh)} subjects from behavior CSV.")
    print(f"  - Dropped {dropped_nan} subjects due to missing/invalid PMAT24_A_CR or ListSort_Unadj.")
    print(f"  - Retained {len(df_out)} subjects with complete dual targets.")
    
    out_dir = Path(OUT_PATH).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)
    print(f"✅ Saved dual-target labels to {OUT_PATH}")

if __name__ == "__main__":
    main()