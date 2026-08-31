@echo off
setlocal
cd /d "%~dp0"

set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\retaining_wall_gui\python.exe"

if not exist "%CONDA_EXE%" (
    echo Miniconda was not found at:
    echo %CONDA_EXE%
    pause
    exit /b 1
)

if not exist "%PYTHON_EXE%" (
    echo Creating the isolated retaining_wall_gui environment...
    "%CONDA_EXE%" create -n retaining_wall_gui python=3.10 -y
    if errorlevel 1 (
        echo Environment creation failed.
        pause
        exit /b 1
    )
)

echo Installing the retaining-wall interface and solver into:
echo %PYTHON_EXE%
echo.
"%PYTHON_EXE%" -m pip install -r requirements-gui.txt

if errorlevel 1 (
    echo.
    echo Installation failed.
    pause
    exit /b 1
)

echo.
echo Installation complete. You can now run:
echo start_retaining_wall_app.bat
pause
endlocal
