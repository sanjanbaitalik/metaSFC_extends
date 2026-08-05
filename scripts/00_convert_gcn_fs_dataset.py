from pathlib import Path
import numpy as np


def find_first(patterns):
    for p in patterns:
        matches = list(Path(".").glob(p))
        if matches:
            return matches[0]
    return None


def load_array(path):
    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr, dtype=np.float32)
    return arr


def main():
    # Change these two paths to where you cloned/copied GCN-FS-fusion
    source_root = Path("../GCN-FS-fusion")
    fc_root = source_root / "dataset_FC"
    sc_root = source_root / "dataset_SC"

    if not fc_root.exists() or not sc_root.exists():
        raise FileNotFoundError(
            "Could not find dataset_FC/dataset_SC. "
            "Edit source_root in this script."
        )

    fc_files = sorted(fc_root.glob("**/X.npy"))
    if not fc_files:
        raise FileNotFoundError(f"No X.npy files found under {fc_root}")

    all_fc, all_sc, all_y = [], [], []

    for fc_file in fc_files:
        rel = fc_file.relative_to(fc_root)
        sc_file = sc_root / rel
        y_file = sc_file.parent / "Y.npy"

        if not sc_file.exists():
            print(f"[SKIP] Missing SC file for {fc_file}: {sc_file}")
            continue
        if not y_file.exists():
            print(f"[SKIP] Missing label file for {fc_file}: {y_file}")
            continue

        fc = load_array(fc_file)
        sc = load_array(sc_file)
        y = np.load(y_file, allow_pickle=True)

        fc = np.asarray(fc, dtype=np.float32)
        sc = np.asarray(sc, dtype=np.float32)
        y = np.asarray(y)

        print("Loaded:", fc_file)
        print("  FC:", fc.shape, "SC:", sc.shape, "Y:", y.shape)

        if fc.ndim == 2:
            fc = fc[None, :, :]
        if sc.ndim == 2:
            sc = sc[None, :, :]
        if y.ndim == 0:
            y = y[None]

        all_fc.append(fc)
        all_sc.append(sc)
        all_y.append(y)

    FC_all = np.concatenate(all_fc, axis=0)
    SC_all = np.concatenate(all_sc, axis=0)
    label_all = np.concatenate(all_y, axis=0)

    if FC_all.shape != SC_all.shape:
        raise ValueError(f"FC/SC shape mismatch: {FC_all.shape} vs {SC_all.shape}")
    if FC_all.shape[1:] != (116, 116):
        print(f"[WARN] Expected 116x116 connectomes, got {FC_all.shape[1:]}")
    if len(label_all) != FC_all.shape[0]:
        raise ValueError("Number of labels does not match number of subjects.")

    Path("inputs/dataset_FC").mkdir(parents=True, exist_ok=True)
    Path("inputs/dataset_SC").mkdir(parents=True, exist_ok=True)

    np.save("inputs/dataset_FC/FC_all.npy", FC_all.astype(np.float32))
    np.save("inputs/dataset_SC/SC_all.npy", SC_all.astype(np.float32))
    np.save("inputs/dataset_SC/label_all.npy", label_all)

    print("\nSaved:")
    print("  inputs/dataset_FC/FC_all.npy", FC_all.shape)
    print("  inputs/dataset_SC/SC_all.npy", SC_all.shape)
    print("  inputs/dataset_SC/label_all.npy", label_all.shape)


if __name__ == "__main__":
    main()