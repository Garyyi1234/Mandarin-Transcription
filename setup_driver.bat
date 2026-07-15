@echo off
echo ==========================================
echo UnityCapture Driver Setup
echo ==========================================
echo This script will download the UnityCapture driver from GitHub.
echo.

powershell -Command "Invoke-WebRequest -Uri 'https://github.com/schellingb/UnityCapture/archive/refs/heads/master.zip' -OutFile 'UnityCapture.zip'"
echo Downloaded UnityCapture.zip. Extracting...

powershell -Command "Expand-Archive -Path 'UnityCapture.zip' -DestinationPath '.' -Force"
echo Extracted.

echo.
echo ==========================================
echo IMPORTANT: To install the virtual camera, you must now:
echo 1. Open the folder "UnityCapture-master"
echo 2. Right-click on "Install.bat" and select "Run as Administrator"
echo ==========================================
pause
