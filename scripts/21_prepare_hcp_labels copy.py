"""Extract HCP behavioral labels (dual-target) for the packing step.

Extracts BOTH ICLR 2027 targets from the unrestricted behavior table:

- ``PMAT24_A_CR``    -> column ``label``          (fluid intelligence)
- ``ListSort_Unadj`` -> column ``listsort_unadj`` (working memory)

Subjects missing either target are dropped HERE with an explicit per-reason
count, so downstream packing never sees NaN labels.
"""
from pathlib import Path
import pandas as pd

subjects_path = Path("data/hcp/manifest/machine_big2_subjects.txt")
behavior_path = Path("data/hcp/behavior/unrestricted_behavioral.csv")
out_path = Path("data/hcp/processed/labels.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

TARGETS = {"label": "PMAT24_A_CR", "listsort_unadj": "ListSort_Unadj"}

subjects = [s.strip() for s in subjects_path.read_text().splitlines() if s.strip()]
beh = pd.read_csv(behavior_path, on_bad_lines='skip')
beh["Subject"] = beh["Subject"].astype(str)
beh = beh.set_index("Subject")

for col in TARGETS.values():
    if col not in beh.columns:
        raise ValueError(f"Column '{col}' not found in {behavior_path}")

rows = []
n_missing_row = n_nan = 0
for sub in subjects:
    if sub not in beh.index:
        print(f"[SKIP] missing behavior row: {sub}")
        n_missing_row += 1
        continue
    values = {col: beh.loc[sub, col] for col in TARGETS}
    if any(pd.isna(v) for v in values.values()):
        reasons = [name for name, col in TARGETS.items() if pd.isna(values[col])]
        print(f"[SKIP] missing behavioral target(s) {reasons}: {sub}")
        n_nan += 1
        continue
    rows.append({"subject": sub, **{name: float(values[col])
                                    for name, col in TARGETS.items()}})

df = pd.DataFrame(rows)
df.to_csv(out_path, index=False)

print(df.head())
print("Saved:", out_path)
print(f"N kept: {len(df)} / {len(subjects)} input subjects "
      f"(dropped: {n_missing_row} without a behavior row, "
      f"{n_nan} with NaN targets)")
print("Label ranges:", {c: (df[c].min(), df[c].max()) for c in TARGETS})
