@echo off
rem Gesture Control - double-click this file.
rem First run creates a Python environment and installs the dependencies,
rem then walks you through calibration. After that it goes straight to
rem controlling the mouse.
setlocal
cd /d "%~dp0"
title Gesture Control

set "VENV=%~dp0.venv"
set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" (
    echo Setting up Python ^(one time, about a minute^)...
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -m venv "%VENV%"
    ) else (
        where python >nul 2>&1
        if errorlevel 1 goto :nopython
        python -m venv "%VENV%"
    )
    if not exist "%PY%" goto :venvfail
)

if not exist "%VENV%\Lib\site-packages\mediapipe" (
    echo Installing dependencies...
    "%PY%" -m pip install --upgrade pip >nul 2>&1
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 goto :pipfail
)

"%PY%" -m gesture_control %*
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" (
    echo.
    pause
)
exit /b %CODE%

:nopython
echo.
echo Python was not found. Install Python 3.10 or newer from
echo   https://www.python.org/downloads/windows/
echo and tick "Add python.exe to PATH" during setup, then run this file again.
echo.
pause
exit /b 1

:venvfail
echo.
echo Could not create the Python environment in %VENV%.
echo Delete that folder and try again.
echo.
pause
exit /b 1

:pipfail
echo.
echo Installing the dependencies failed. Check your internet connection,
echo then delete the .venv folder and run this file again.
echo.
pause
exit /b 1
