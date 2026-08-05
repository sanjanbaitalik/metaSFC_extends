param(
    [string]$BatchFile = "data/hcp/manifest/batches/batch_01.txt"
)

$ErrorActionPreference = "Stop"

$Subjects = Get-Content $BatchFile

foreach ($SUB in $Subjects) {
    if ([string]::IsNullOrWhiteSpace($SUB)) { continue }

    $FC = "data/hcp/processed/fc/$SUB`_fc.npy"
    $SC = "data/hcp/processed/sc/$SUB/sc_116.csv"
    $RAW = "data/hcp/raw/$SUB"

    if ((Test-Path $FC) -and (Test-Path $SC)) {
        Write-Host "Deleting raw files for $SUB"
        Remove-Item -Recurse -Force $RAW
    }
    else {
        Write-Host "[KEEP] Missing processed output for $SUB. Raw folder not deleted."
    }
}