@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%USERPROFILE%\miniconda3\envs\retaining_wall_gui\python.exe"
if not exist "%PYTHON_EXE%" (
    echo The retaining_wall_gui environment is not installed.
    echo Run install_retaining_wall_gui.bat first.
    echo.
    pause
    exit /b 1
)

"%PYTHON_EXE%" retaining_wall_app.py
if errorlevel 1 (
    echo.
    echo The interface did not start.
    echo Install the GUI packages with:
    echo "%PYTHON_EXE%" -m pip install -r requirements-gui.txt
    echo.
    pause
)

endlocal
