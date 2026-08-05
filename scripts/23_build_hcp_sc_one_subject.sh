#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C
export PATH="/usr/bin:/bin:${PATH}"

SUB=$1

RAW="data/hcp/raw/${SUB}"
OUT="data/hcp/processed/sc/${SUB}"
ATLAS_MNI="inputs/atlases/AAL116.nii.gz"

mkdir -p "${OUT}"

echo "===== ${SUB}: converting DWI ====="

mrconvert \
  "${RAW}/Diffusion/data.nii.gz" \
  "${OUT}/dwi.mif" \
  -fslgrad "${RAW}/Diffusion/bvecs" "${RAW}/Diffusion/bvals" \
  -force

mrconvert \
  "${RAW}/Diffusion/nodif_brain_mask.nii.gz" \
  "${OUT}/mask.mif" \
  -force

echo "===== ${SUB}: warping AAL116 from MNI to subject ACPC/T1w space ====="

applywarp \
  -i "${ATLAS_MNI}" \
  -r "${RAW}/T1w/T1w_acpc_dc_restore.nii.gz" \
  -w "${RAW}/xfms/standard2acpc_dc.nii.gz" \
  -o "${OUT}/aal116_acpc.nii.gz" \
  --interp=nn

mrconvert \
  "${OUT}/aal116_acpc.nii.gz" \
  "${OUT}/aal116_acpc.mif" \
  -datatype uint32 \
  -force

echo "===== ${SUB}: estimating response and FOD ====="

dwi2response dhollander \
  "${OUT}/dwi.mif" \
  "${OUT}/wm.txt" "${OUT}/gm.txt" "${OUT}/csf.txt" \
  -mask "${OUT}/mask.mif" \
  -force

dwi2fod msmt_csd \
  "${OUT}/dwi.mif" \
  "${OUT}/wm.txt" "${OUT}/wmfod.mif" \
  "${OUT}/gm.txt" "${OUT}/gm.mif" \
  "${OUT}/csf.txt" "${OUT}/csf.mif" \
  -mask "${OUT}/mask.mif" \
  -force

mtnormalise \
  "${OUT}/wmfod.mif" "${OUT}/wmfod_norm.mif" \
  "${OUT}/gm.mif" "${OUT}/gm_norm.mif" \
  "${OUT}/csf.mif" "${OUT}/csf_norm.mif" \
  -mask "${OUT}/mask.mif" \
  -force

echo "===== ${SUB}: 5TT image ====="

SCRATCH_ROOT="/tmp/mrtrix_${SUB}"
mkdir -p "${SCRATCH_ROOT}/5ttgen"

5ttgen fsl \
  "${RAW}/T1w/T1w_acpc_dc_restore.nii.gz" \
  "${OUT}/5tt.mif" \
  -force

echo "===== ${SUB}: tractography ====="

tckgen \
  "${OUT}/wmfod_norm.mif" \
  "${OUT}/tracks_2m.tck" \
  -algorithm iFOD2 \
  -act "${OUT}/5tt.mif" \
  -backtrack \
  -seed_dynamic "${OUT}/wmfod_norm.mif" \
  -select 2M \
  -cutoff 0.06 \
  -force

echo "===== ${SUB}: SIFT2 ====="

tcksift2 \
  "${OUT}/tracks_2m.tck" \
  "${OUT}/wmfod_norm.mif" \
  "${OUT}/sift2_weights.txt" \
  -act "${OUT}/5tt.mif" \
  -force

echo "===== ${SUB}: connectome ====="

tck2connectome \
  "${OUT}/tracks_2m.tck" \
  "${OUT}/aal116_acpc.mif" \
  "${OUT}/sc_116.csv" \
  -tck_weights_in "${OUT}/sift2_weights.txt" \
  -symmetric \
  -zero_diagonal \
  -force

echo "Done: ${OUT}/sc_116.csv"