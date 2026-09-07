@echo off
rem Measure your hand and fix clicking: run this if clicks fire when you did
rem not mean them, or refuse to fire when you did.
cd /d "%~dp0"
call "%~dp0run.bat" tune %*
