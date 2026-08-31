@echo off
setlocal
cd /d "%~dp0"

set "CONDA_ROOT="
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_ROOT=%USERPROFILE%\miniconda3"
if not defined CONDA_ROOT if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_ROOT=%USERPROFILE%\anaconda3"
if not defined CONDA_ROOT if exist "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe" set "CONDA_ROOT=%LOCALAPPDATA%\miniconda3"
if not defined CONDA_ROOT if exist "%LOCALAPPDATA%\anaconda3\Scripts\conda.exe" set "CONDA_ROOT=%LOCALAPPDATA%\anaconda3"
if not defined CONDA_ROOT if exist "C:\ProgramData\miniconda3\Scripts\conda.exe" set "CONDA_ROOT=C:\ProgramData\miniconda3"
if not defined CONDA_ROOT if exist "C:\ProgramData\anaconda3\Scripts\conda.exe" set "CONDA_ROOT=C:\ProgramData\anaconda3"

if not defined CONDA_ROOT (
    echo Neither Miniconda nor Anaconda could be found.
    echo Checked the standard user, LocalAppData, and ProgramData locations.
    pause
    exit /b 1
)

set "CONDA_EXE=%CONDA_ROOT%\Scripts\conda.exe"
set "PYTHON_EXE=%CONDA_ROOT%\envs\retaining_wall_gui\python.exe"
echo Using Conda installation:
echo %CONDA_ROOT%
echo.

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
