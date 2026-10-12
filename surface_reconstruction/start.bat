@echo off
setlocal
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "SURFACE_PYTHON=python"
if exist "%~dp0.venv\Scripts\python.exe" set "SURFACE_PYTHON=%~dp0.venv\Scripts\python.exe"
"%SURFACE_PYTHON%" -c "import numpy, scipy, cv2" >nul 2>&1
if errorlevel 1 (
    echo Python dependencies are missing. Follow the setup steps in README.md.
    exit /b 1
)
"%SURFACE_PYTHON%" "%~dp0run_reconstruction.py" %*
exit /b %errorlevel%
