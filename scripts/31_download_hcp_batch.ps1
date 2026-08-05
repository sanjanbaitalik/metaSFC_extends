param(
    [string]$BatchFile = "data/hcp/manifest/batches/batch_01.txt"
)

$ErrorActionPreference = "Stop"

$Profile = "hcp"
$Region = "us-east-1"
$Bucket = "s3://hcp-openaccess/HCP_1200"

$Subjects = Get-Content $BatchFile

foreach ($SUB in $Subjects) {
    if ([string]::IsNullOrWhiteSpace($SUB)) { continue }

    Write-Host "`n===== Downloading subject $SUB ====="

    $DEST = "data/hcp/raw/$SUB"

    New-Item -ItemType Directory -Force -Path "$DEST/rfMRI_REST1_LR" | Out-Null
    New-Item -ItemType Directory -Force -Path "$DEST/rfMRI_REST1_RL" | Out-Null
    New-Item -ItemType Directory -Force -Path "$DEST/Diffusion" | Out-Null
    New-Item -ItemType Directory -Force -Path "$DEST/T1w" | Out-Null
    New-Item -ItemType Directory -Force -Path "$DEST/xfms" | Out-Null

    aws s3 cp "$Bucket/$SUB/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_hp2000_clean.nii.gz" `
        "$DEST/rfMRI_REST1_LR/" --profile $Profile --region $Region

    aws s3 cp "$Bucket/$SUB/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_hp2000_clean.nii.gz" `
        "$DEST/rfMRI_REST1_RL/" --profile $Profile --region $Region

    aws s3 cp "$Bucket/$SUB/T1w/Diffusion/data.nii.gz" `
        "$DEST/Diffusion/" --profile $Profile --region $Region

    aws s3 cp "$Bucket/$SUB/T1w/Diffusion/bvals" `
        "$DEST/Diffusion/" --profile $Profile --region $Region

    aws s3 cp "$Bucket/$SUB/T1w/Diffusion/bvecs" `
        "$DEST/Diffusion/" --profile $Profile --region $Region

    aws s3 cp "$Bucket/$SUB/T1w/Diffusion/nodif_brain_mask.nii.gz" `
        "$DEST/Diffusion/" --profile $Profile --region $Region

    aws s3 cp "$Bucket/$SUB/T1w/T1w_acpc_dc_restore.nii.gz" `
        "$DEST/T1w/" --profile $Profile --region $Region

    aws s3 cp "$Bucket/$SUB/MNINonLinear/xfms/standard2acpc_dc.nii.gz" `
        "$DEST/xfms/" --profile $Profile --region $Region
}