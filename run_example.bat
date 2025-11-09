@echo off
REM Example script to run SmartEye with webcam

echo Starting SmartEye Example...
echo.

echo Step 1: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 2: Starting server in background...
start "SmartEye Server" cmd /k "python server.py"

echo Waiting for server to start...
timeout /t 5 /nobreak > nul

echo.
echo Step 3: Starting client with webcam...
python client.py --streams 0 --names webcam_0 --fps-limit 30

echo.
echo Press any key to stop the server...
pause

REM Kill server process (Windows)
taskkill /FI "WINDOWTITLE eq SmartEye Server*" /T /F > nul 2>&1

echo.
echo Example completed!

