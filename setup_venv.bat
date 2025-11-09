@echo off
REM Windows batch script to create and setup virtual environment

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Virtual environment setup complete!
echo.
echo To activate the virtual environment, run:
echo   venv\Scripts\activate
echo.
echo To start the server:
echo   python server.py
echo.
echo To start the client:
echo   python client.py --streams 0 --names webcam_0
echo.
pause

