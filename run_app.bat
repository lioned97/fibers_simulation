@echo off
echo ==============================================================
echo        NV-Fiber Optical Ray Tracing Simulator Launcher
echo ==============================================================
echo.
echo Installing / updating Python dependencies...
py -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo.
    echo WARNING: Failed to install some dependencies. The app might still run if they are already installed.
    echo.
)
echo.
echo Launching Streamlit web application...
py -m streamlit run app.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error running Streamlit. Please make sure Streamlit is correctly installed.
    echo.
)
pause
