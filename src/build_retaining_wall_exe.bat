@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE="
if exist "%USERPROFILE%\miniconda3\envs\retaining_wall_gui\python.exe" set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\retaining_wall_gui\python.exe"
if not defined PYTHON_EXE if exist "%USERPROFILE%\anaconda3\envs\retaining_wall_gui\python.exe" set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\retaining_wall_gui\python.exe"
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\miniconda3\envs\retaining_wall_gui\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\miniconda3\envs\retaining_wall_gui\python.exe"
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\anaconda3\envs\retaining_wall_gui\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\anaconda3\envs\retaining_wall_gui\python.exe"
if not defined PYTHON_EXE if exist "C:\ProgramData\miniconda3\envs\retaining_wall_gui\python.exe" set "PYTHON_EXE=C:\ProgramData\miniconda3\envs\retaining_wall_gui\python.exe"
if not defined PYTHON_EXE if exist "C:\ProgramData\anaconda3\envs\retaining_wall_gui\python.exe" set "PYTHON_EXE=C:\ProgramData\anaconda3\envs\retaining_wall_gui\python.exe"
set "OUTPUT_DIR=%~dp0dist\TPC_Retaining_Wall_Analysis"
set "OUTPUT_EXE=%OUTPUT_DIR%\TPC_Retaining_Wall_Analysis.exe"

if not defined PYTHON_EXE (
    echo The retaining_wall_gui environment is not installed.
    echo Run install_retaining_wall_gui.bat first.
    echo.
    pause
    exit /b 1
)

echo Installing the executable build dependency...
"%PYTHON_EXE%" -m pip install -r requirements-build.txt
if errorlevel 1 goto :build_failed

echo.
echo Building TPC Retaining Wall Soil Analysis v1.0.0...
"%PYTHON_EXE%" -m PyInstaller --noconfirm --clean retaining_wall_app.spec
if errorlevel 1 goto :build_failed

if not exist "%OUTPUT_EXE%" (
    echo The build completed without producing the expected executable.
    goto :build_failed
)

if exist "%OUTPUT_DIR%\license_private_key.json" (
    echo SECURITY ERROR: the administrator private key entered the build.
    echo Do not distribute this folder.
    pause
    exit /b 1
)

if exist "%OUTPUT_DIR%\_internal\license_private_key.json" (
    echo SECURITY ERROR: the administrator private key entered the build.
    echo Do not distribute this folder.
    pause
    exit /b 1
)

echo.
echo Build complete.
echo Run or distribute the complete folder:
echo %OUTPUT_DIR%
echo.
echo Executable:
echo %OUTPUT_EXE%
pause
exit /b 0

:build_failed
echo.
echo The executable build failed. Review the messages above.
pause
exit /b 1
