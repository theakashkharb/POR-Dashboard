@echo off
echo ================================
echo POR-Dashboard Environment Setup
echo ================================

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -m venv .venv
) else (
    echo Virtual environment already exists.
)

echo.
echo Installing dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo ================================
echo Setup complete!
echo ================================
.venv\Scripts\python.exe --version

pause