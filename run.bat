@echo off
title Infineon AI Data Quality Agent - Tibin Regi
cd /d "%~dp0"
echo.
echo ============================================================
echo   Infineon AI Data Quality Agent
echo   Built by Tibin Regi
echo ============================================================
echo.
taskkill /F /IM python.exe >nul 2>&1
timeout /t 1 /nobreak >nul
echo  Running pipeline...
echo.
python run_agent.py %1
echo.
echo ============================================================
echo  Running pytest test suite...
echo ============================================================
echo.
pytest tests/ -v --tb=short
echo.
echo ============================================================
echo   Complete! Check reports/ folder for full results.
echo ============================================================
pause
