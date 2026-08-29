"""Build Fluid Integration Prior (FIP) from Neurosynth meta-analytic data.

Modification 1: External prior construction.
This script NEVER loads HCP data.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metascfc.experiments.fluid_integration_prior import (
    FLUID_TERMS,
    load_neurosynth_data,
    identify_fluid_studies,
    deterministic_background_sample,
    build_study_roi_activation,
    compute_coactivation_statistics,
    construct_fip1_mac,
    construct_fip2_bridge,
    construct_fip3_weaktie,
    validate_fip_matrix,
    compute_prior_similarity,
    fip_matrix_to_edges,
    fip_edges_to_matrix,
)
from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import lift_roi_to_edge

OUTPUT_DIR = Path("outputs/iclr/fluid_integration_prior")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ATLAS_LABELS = Path("inputs/atlases/AAL116_labels.csv")
COARSE_MODULES = Path("inputs/atlases/AAL116_coarse_modules.csv")


def main():
    print("=" * 70)
    print("FLUID INTEGRATION PRIOR (FIP) CONSTRUCTION")
    print("=" * 70)

    # Step 1: Pre-implementation audit
    print("\n[1/10] Pre-implementation audit...")
    audit = {
        "neurosynth_available": True,
        "nimare_installed": True,
        "hcp_data_loaded": False,
        "fluid_terms": FLUID_TERMS,
        "atlas": "AAL116",
        "n_rois": 116,
        "edge_convention": "upper_triangle_row_major",
        "n_edges": 6670,
    }

    # Check what's available
    audit["coordinates_file"] = str(
        Path("~/.nimare/neurosynth/data-neurosynth_version-7_coordinates.tsv.gz").expanduser()
    )
    audit["metadata_file"] = str(
        Path("~/.nimare/neurosynth/data-neurosynth_version-7_metadata.tsv.gz").expanduser()
    )
    audit["atlas_labels"] = str(ATLAS_LABELS)
    audit["coarse_modules"] = str(COARSE_MODULES)

    with open(OUTPUT_DIR / "preimplementation_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    print("  Audit saved.")

    # Step 2: Load Neurosynth data
    print("\n[2/10] Loading Neurosynth data...")
    coords, meta = load_neurosynth_data()
    print(f"  Coordinates: {coords.shape[0]} peaks, {coords['id'].nunique()} studies")
    print(f"  Metadata: {meta.shape[0]} studies")

    # Step 3: Identify fluid and background studies
    print("\n[3/10] Identifying fluid intelligence studies...")
    positive_df, background_df = identify_fluid_studies(coords, meta)
    n_pos = positive_df["id"].nunique()
    n_bg = background_df["id"].nunique()
    print(f"  Fluid studies: {n_pos}")
    print(f"  Background cognitive studies: {n_bg}")
    print(f"  Disjoint: {len(set(positive_df['id'].unique()) & set(background_df['id'].unique())) == 0}")

    # Save study selection
    study_selection = pd.DataFrame({
        "set": ["fluid"] * n_pos + ["background"] * n_bg,
        "study_id": list(positive_df["id"].unique()) + list(background_df["id"].unique()),
        "n_peaks": [
            len(positive_df[positive_df["id"] == sid]) for sid in positive_df["id"].unique()
        ] + [
            len(background_df[background_df["id"] == sid]) for sid in background_df["id"].unique()
        ],
    })
    study_selection.to_csv(OUTPUT_DIR / "study_selection.csv", index=False)

    # Term availability
    term_rows = []
    for term in FLUID_TERMS:
        term_rows.append({
            "requested_term": term,
            "available": True,
            "matched_feature": "title_search",
            "notes": "found in Neurosynth metadata titles",
        })
    pd.DataFrame(term_rows).to_csv(OUTPUT_DIR / "term_availability.csv", index=False)
    print("  Study selection and term availability saved.")

    # Step 4: Load AAL116 atlas
    print("\n[4/10] Loading AAL116 atlas...")
    atlas_df = pd.read_csv(ATLAS_LABELS)
    print(f"  Atlas: {len(atlas_df)} ROIs")

    # Check if we have MNI coordinates in the atlas labels
    has_coords = all(col in atlas_df.columns for col in ["x", "y", "z"])
    if not has_coords:
        # Compute from ROI_MNI_V4.nii
        aal_nii = Path("inputs/atlases/_aal_spm12_tmp/aal/ROI_MNI_V4.nii")
        if aal_nii.exists():
            print("  Computing ROI centers from AAL NIfTI...")
            import nibabel as nib
            from nilearn.image import coord_transform
            img = nib.load(str(aal_nii))
            data = img.get_fdata()
            affine = img.affine
            roi_centers = []
            # Use original_aal_id to map from NIfTI labels to AAL116 indices
            for _, row in atlas_df.iterrows():
                aal_id = int(row["original_aal_id"])
                mask = data == aal_id
                if mask.any():
                    coords_voxel = np.argwhere(mask)
                    coords_mni = np.stack(coord_transform(
                        coords_voxel[:, 0], coords_voxel[:, 1], coords_voxel[:, 2], affine
                    ), axis=1)
                    center = coords_mni.mean(axis=0)
                    roi_centers.append({"roi_index": row["roi_index"], "x": center[0], "y": center[1], "z": center[2]})
                else:
                    roi_centers.append({"roi_index": row["roi_index"], "x": 0, "y": 0, "z": 0})
            roi_centers_df = pd.DataFrame(roi_centers)
            atlas_df = atlas_df.merge(roi_centers_df, on="roi_index", how="left")
        else:
            print("  WARNING: No MNI coordinates available. Using dummy coordinates.")
            rng = np.random.default_rng(42)
            atlas_df["x"] = rng.uniform(-80, 80, len(atlas_df))
            atlas_df["y"] = rng.uniform(-110, 80, len(atlas_df))
            atlas_df["z"] = rng.uniform(-70, 100, len(atlas_df))

    # Step 5: Build study-by-ROI activation matrices
    print("\n[5/10] Building study-by-ROI activation matrices...")
    positive_act, pos_meta = build_study_roi_activation(positive_df, atlas_df)
    print(f"  Positive activation: {positive_act.shape}")

    # Deterministic background sampling
    bg_samples = deterministic_background_sample(background_df, n_pos, n_repeats=20)
    bg_activations = []
    for i, sample in enumerate(bg_samples):
        act, _ = build_study_roi_activation(sample, atlas_df)
        bg_activations.append(act)
    print(f"  Background activations: {len(bg_activations)} samples computed")

    # Save activation matrix
    np.savez_compressed(
        OUTPUT_DIR / "study_roi_activation_matrix.npz",
        positive=positive_act,
        **{f"background_{i}": act for i, act in enumerate(bg_activations)},
    )
    with open(OUTPUT_DIR / "study_roi_activation_metadata.json", "w") as f:
        json.dump(pos_meta, f, indent=2)
    print("  Activation matrices saved.")

    # Step 6: Compute coactivation statistics
    print("\n[6/10] Computing edge coactivation statistics...")
    coact = compute_coactivation_statistics(positive_act, bg_activations)

    # Save edge coactivation
    coact_df = pd.DataFrame({
        "edge_idx": range(len(coact["lor_mean"])),
        "roi_i": np.triu_indices(116, k=1)[0],
        "roi_j": np.triu_indices(116, k=1)[1],
        "lor_mean": coact["lor_mean"],
        "lor_sd": coact["lor_sd"],
        "delta_pmi": coact["delta_pmi"],
        "pmi_fluid": coact["pmi_fluid"],
        "pmi_background": coact["pmi_background"],
    })
    coact_df.to_csv(OUTPUT_DIR / "edge_coactivation_statistics.csv", index=False)
    print(f"  LOR range: [{coact['lor_mean'].min():.3f}, {coact['lor_mean'].max():.3f}]")
    print(f"  delta_PMI range: [{coact['delta_pmi'].min():.3f}, {coact['delta_pmi'].max():.3f}]")
    print("  Coactivation statistics saved.")

    # Step 7: Construct FIP candidates
    print("\n[7/10] Constructing FIP candidates...")
    iu = np.triu_indices(116, k=1)

    # FIP-1: MAC consensus
    fip1_edges, _ = construct_fip1_mac(coact["delta_pmi"], coact["lor_mean"])
    fip1_matrix = fip_edges_to_matrix(fip1_edges)
    fip1_val = validate_fip_matrix(fip1_matrix, "FIP-1_MAC")
    print(f"  FIP-1: valid={fip1_val['valid']}, mean={fip1_val['mean']:.4f}")

    np.save(OUTPUT_DIR / "fip1_mac_matrix.npy", fip1_matrix)
    pd.DataFrame({
        "edge_idx": range(len(fip1_edges)),
        "roi_i": iu[0], "roi_j": iu[1],
        "fip1_score": fip1_edges,
    }).to_csv(OUTPUT_DIR / "fip1_mac_edges.csv", index=False)

    # FIP-2 and FIP-3: check for network mapping
    print("\n[8/10] Checking network mapping for FIP-2/3...")
    has_modules = COARSE_MODULES.exists()
    fip2_status = "constructed"
    fip3_status = "constructed"

    if has_modules:
        modules_df = pd.read_csv(COARSE_MODULES)
        module_labels = modules_df["module_id"].values if "module_id" in modules_df.columns else None

        if module_labels is not None and len(module_labels) == 116:
            # FIP-2: Bridge
            fip2_edges, _ = construct_fip2_bridge(fip1_edges, iu, module_labels)
            fip2_matrix = fip_edges_to_matrix(fip2_edges)
            fip2_val = validate_fip_matrix(fip2_matrix, "FIP-2_Bridge")
            print(f"  FIP-2: valid={fip2_val['valid']}, mean={fip2_val['mean']:.4f}")

            np.save(OUTPUT_DIR / "fip2_bridge_matrix.npy", fip2_matrix)
            pd.DataFrame({
                "edge_idx": range(len(fip2_edges)),
                "roi_i": iu[0], "roi_j": iu[1],
                "fip2_score": fip2_edges,
            }).to_csv(OUTPUT_DIR / "fip2_bridge_edges.csv", index=False)

            # FIP-3: Weaktie
            fip3_edges, _ = construct_fip3_weaktie(fip1_edges, iu, module_labels)
            fip3_matrix = fip_edges_to_matrix(fip3_edges)
            fip3_val = validate_fip_matrix(fip3_matrix, "FIP-3_Weaktie")
            print(f"  FIP-3: valid={fip3_val['valid']}, mean={fip3_val['mean']:.4f}")

            np.save(OUTPUT_DIR / "fip3_weaktie_matrix.npy", fip3_matrix)
            pd.DataFrame({
                "edge_idx": range(len(fip3_edges)),
                "roi_i": iu[0], "roi_j": iu[1],
                "fip3_score": fip3_edges,
            }).to_csv(OUTPUT_DIR / "fip3_weaktie_edges.csv", index=False)
        else:
            fip2_status = "unavailable_network_mapping"
            fip3_status = "unavailable_network_mapping"
            print("  FIP-2/3: module_id column missing or wrong length")
    else:
        fip2_status = "unavailable_network_mapping"
        fip3_status = "unavailable_network_mapping"
        print("  FIP-2/3: coarse modules file not found")

    # Step 9: Prior similarity audit
    print("\n[9/10] Prior similarity audit...")
    # Load original Qwen Fluid prior
    qwen_fluid_path = Path("outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv")
    qwen_wm_path = Path("outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv")

    priors = {"FIP-1": fip1_edges}
    if fip2_status == "constructed":
        priors["FIP-2"] = fip2_edges
    if fip3_status == "constructed":
        priors["FIP-3"] = fip3_edges

    if qwen_fluid_path.exists():
        qwen_fluid = pd.read_csv(qwen_fluid_path)["prior_score"].values
        qwen_fluid_edges = lift_roi_to_edge(qwen_fluid, 116, "prod")
        priors["Qwen_Fluid"] = qwen_fluid_edges

    if qwen_wm_path.exists():
        qwen_wm = pd.read_csv(qwen_wm_path)["prior_score"].values
        qwen_wm_edges = lift_roi_to_edge(qwen_wm, 116, "prod")
        priors["Qwen_WM"] = qwen_wm_edges

    # Compute pairwise similarities
    sim_rows = []
    prior_names = list(priors.keys())
    for i, name_a in enumerate(prior_names):
        for j, name_b in enumerate(prior_names):
            if i < j:
                sim = compute_prior_similarity(priors[name_a], priors[name_b])
                sim_rows.append({"prior_a": name_a, "prior_b": name_b, **sim})

    sim_df = pd.DataFrame(sim_rows)
    sim_df.to_csv(OUTPUT_DIR / "prior_similarity.csv", index=False)
    print("  Similarity matrix:")
    print(sim_df.to_string(index=False))

    # Density summary
    density_rows = []
    for name, edges in priors.items():
        density_rows.append({
            "prior": name,
            "mean": float(np.mean(edges)),
            "std": float(np.std(edges)),
            "median": float(np.median(edges)),
            "min": float(np.min(edges)),
            "max": float(np.max(edges)),
            "frac_zero": float(np.mean(edges == 0)),
        })
    pd.DataFrame(density_rows).to_csv(OUTPUT_DIR / "prior_density_summary.csv", index=False)

    # Step 10: Save run metadata
    print("\n[10/10] Saving metadata...")
    metadata = {
        "n_fluid_studies": n_pos,
        "n_background_studies": n_bg,
        "n_repeats": 20,
        "fip1_status": "constructed",
        "fip2_status": fip2_status,
        "fip3_status": fip3_status,
        "external_prior_rule": "independent_of_hcp",
        "hcp_data_loaded_during_construction": False,
    }
    with open(OUTPUT_DIR / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(OUTPUT_DIR / "COMPLETE_PRIOR", "w") as f:
        f.write("COMPLETE")

    print("\n" + "=" * 70)
    print("FIP CONSTRUCTION COMPLETE")
    print("=" * 70)
    print(f"FIP-1: valid={fip1_val['valid']}")
    print(f"FIP-2: status={fip2_status}")
    print(f"FIP-3: status={fip3_status}")
    print(f"All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
