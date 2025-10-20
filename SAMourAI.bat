@echo off
chcp 65001 >nul
color 03
cls

echo.
echo ╔──────────────────────╗
echo │╔═╗╔═╗╔╦╗┌─┐┬ ┬┬─┐╔═╗╦│
echo │╚═╗╠═╣║║║│ ││ │├┬┘╠═╣║│
echo │╚═╝╩ ╩╩ ╩└─┘└─┘┴└─╩ ╩╩│
echo ╚──────────────────────╝
echo      /GPU version/
echo ▬▬[════════════════════ﺤ 
echo.
echo [Initializing system...]
echo.

SET "PROJECT_PATH=%~dp0"

SET "VENV_PATH=%PROJECT_PATH%samourai_env\Scripts\activate.bat"

IF NOT EXIST "%VENV_PATH%" (
    echo [ERROR] Virtual environment not found: "%VENV_PATH%"
    pause
    exit /b
)

call "%VENV_PATH%"

SET "IMAGE_DIR=%PROJECT_PATH%image_dir"
if not exist "%IMAGE_DIR%" mkdir "%IMAGE_DIR%"

echo [Launching SAMourAI GPU Interface...]

python "%PROJECT_PATH%launchers\launcher_gpu.py"

echo.
echo [Session ended]
pause
