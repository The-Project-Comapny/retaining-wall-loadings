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
if not defined PYTHON_EXE (
    echo The retaining_wall_gui environment is not installed.
    echo Run install_retaining_wall_gui.bat first.
    pause
    exit /b 1
)

"%PYTHON_EXE%" admin_license_generator.py
if errorlevel 1 pause
endlocal
