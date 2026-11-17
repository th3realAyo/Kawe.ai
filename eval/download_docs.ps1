# Downloads every document listed in eval\nigerian_docs\manifest.csv and
# checks that each file is a real PDF. Run from the petty_talk folder:
#     powershell -ExecutionPolicy Bypass -File eval\download_docs.ps1
# Re-running skips files that are already downloaded and valid.

$ErrorActionPreference = "Continue"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$docs = Join-Path $PSScriptRoot "nigerian_docs"
$rows = Import-Csv (Join-Path $docs "manifest.csv")
$ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Kawe-evaluation"

function Test-Pdf($path) {
    if (-not (Test-Path $path)) { return $false }
    $bytes = [System.IO.File]::ReadAllBytes($path)
    return ($bytes.Length -gt 1000) -and ([System.Text.Encoding]::ASCII.GetString($bytes, 0, 4) -eq "%PDF")
}

$ok = 0; $failed = @()
foreach ($row in $rows) {
    $out = Join-Path $docs $row.file
    if (Test-Pdf $out) { Write-Host "[skip] $($row.file) already downloaded"; $ok++; continue }
    $urls = @($row.url)
    if ($row.url -match "/article/view/") { $urls += ($row.url -replace "/article/view/", "/article/download/") }
    $done = $false
    foreach ($u in $urls) {
        try {
            Invoke-WebRequest -Uri $u -OutFile $out -UserAgent $ua -UseBasicParsing -TimeoutSec 180
            if (Test-Pdf $out) { $done = $true; break }
        } catch { Write-Host "   error from $u : $($_.Exception.Message)" }
    }
    if ($done) {
        $mb = [math]::Round((Get-Item $out).Length / 1MB, 1)
        Write-Host "[ok]   $($row.file) ($mb MB)"; $ok++
    } else {
        if (Test-Path $out) { Remove-Item $out }
        Write-Host "[FAIL] $($row.file) - open this link in your browser and save the PDF with this name:" -ForegroundColor Yellow
        Write-Host "       $($row.url)"
        $failed += $row.file
    }
}
Write-Host ""
Write-Host "$ok of $($rows.Count) documents ready."
if ($failed.Count) { Write-Host "Download these by hand (or delete their rows from manifest.csv): $($failed -join ', ')" -ForegroundColor Yellow }
