@echo off
REM -------------------
REM SAMourAI Launcher -
REM -------------------

REM --- Set UTF-8 code page for special characters ---
chcp 65001 >nul

REM --- Set text color to green ---
color 03

cls
echo.
echo ╔──────────────────────╗
echo │╔═╗╔═╗╔╦╗┌─┐┬ ┬┬─┐╔═╗╦│
echo │╚═╗╠═╣║║║│ ││ │├┬┘╠═╣║│
echo │╚═╝╩ ╩╩ ╩└─┘└─┘┴└─╩ ╩╩│
echo ╚──────────────────────╝
echo       /GPU version/
echo ▬▬[════════════════════ﺤ 
echo.
echo.
echo                           [Initializing system...]
echo.
timeout /t 1 /nobreak >nul

REM --- Detect the directory of the script ---
SET "PROJECT_PATH=%~dp0"

REM --- Automatically detect the virtual environment ---
SET "VENV_PATH=%PROJECT_PATH%samourai_env\Scripts\activate.bat"

REM --- Check if the virtual environment exists ---
IF NOT EXIST "%VENV_PATH%" (
    echo [ERROR] Virtual environment not found: "%VENV_PATH%"
    echo          Please create the samourai_env environment before running this script.
    echo.
    pause
    exit /b
)

echo [OK] Virtual environment detected
REM --- Activate the virtual environment ---
call "%VENV_PATH%"

REM --- Path to the image_dir folder ---
SET "IMAGE_DIR=%PROJECT_PATH%image_dir"

REM --- Model name ---
SET "MODEL_NAME=sam2.1-hiera-large"

REM --- Create the image_dir folder if it does not exist ---
if not exist "%IMAGE_DIR%" (
    echo [OK] Creating image_dir folder
    mkdir "%IMAGE_DIR%"
) else (
    echo [OK] image_dir folder ready
)

echo [OK] Model: %MODEL_NAME%
echo.
echo                           [Launching SAMourAI Interface...]
echo.

REM --- Launch the GUI interface ---
python "%PROJECT_PATH%gui_gpu.py" "%IMAGE_DIR%" %MODEL_NAME%

echo.
echo                           [Session ended]
echo.
pause
