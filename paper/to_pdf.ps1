# Converts paper/fraud_triage.docx to PDF with Microsoft Word and copies it to the website. Author: Vivek Mitra
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$src = Join-Path $here "fraud_triage.docx"
$pdf = Join-Path $here "fraud_triage.pdf"
$site = Join-Path $here "..\..\..\site\papers\fraud.pdf"
$w = New-Object -ComObject Word.Application
$w.Visible = $false
try {
  $d = $w.Documents.Open($src)
  $d.ExportAsFixedFormat([string]$pdf, 17)
  $pages = $d.ComputeStatistics(2)
  $d.Close()
} finally { $w.Quit() }
# Copy to the website only when the vivekmitra.com folder sits next to this repo.
if (Test-Path (Split-Path -Parent $site)) {
  Copy-Item $pdf $site -Force
  Write-Output "  wrote paper/fraud_triage.pdf ($pages pages) and site/papers/fraud.pdf"
} else {
  Write-Output "  wrote paper/fraud_triage.pdf ($pages pages)"
}
