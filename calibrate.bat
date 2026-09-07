@echo off
rem Recalibrate: run this after moving the webcam, changing seats,
rem or switching screen resolution.
cd /d "%~dp0"
call "%~dp0run.bat" calibrate %*
