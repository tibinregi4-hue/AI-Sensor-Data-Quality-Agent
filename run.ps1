# run.ps1 — PowerShell launcher for Infineon AI Data Quality Agent
# Author: Tibin Regi | Infineon AI & Data Engineering Internship Project
# Usage: .\run.ps1           (uses default sample data)
#        .\run.ps1 data\your_file.csv

Set-Location $PSScriptRoot

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Infineon AI Data Quality Agent" -ForegroundColor White
Write-Host "  Built by Tibin Regi" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Kill any stale python processes (optional — comment out if unwanted)
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

Write-Host " Running pipeline..." -ForegroundColor Yellow
Write-Host ""

if ($args.Count -gt 0) {
    python run_agent.py $args[0]
} else {
    python run_agent.py
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Running pytest test suite..." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

pytest tests/ -v --tb=short

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Complete! Check reports/ folder for full results." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"
