@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-portable.ps1" %*
if errorlevel 1 echo Packaging failed. Read the error above.
pause
