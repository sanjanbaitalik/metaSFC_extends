from pathlib import Path
import pandas as pd

subjects_path = Path("data/hcp/manifest/machine_big2_subjects.txt")
behavior_path = Path("data/hcp/behavior/unrestricted_behavioral.csv")
out_path = Path("data/hcp/processed/labels.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

target_col = "PMAT24_A_CR"

subjects = [s.strip() for s in subjects_path.read_text().splitlines() if s.strip()]
beh = pd.read_csv(behavior_path, on_bad_lines='skip')
beh["Subject"] = beh["Subject"].astype(str)
beh = beh.set_index("Subject")

rows = []
for sub in subjects:
    if sub not in beh.index:
        print(f"[SKIP] missing behavior row: {sub}")
        continue

    value = beh.loc[sub, target_col]
    if pd.isna(value):
        print(f"[SKIP] missing {target_col}: {sub}")
        continue

    rows.append({"subject": sub, "label": float(value)})

df = pd.DataFrame(rows)
df.to_csv(out_path, index=False)

print(df.head())
print("Saved:", out_path)
print("N:", len(df))
print("Label range:", df["label"].min(), df["label"].max())
