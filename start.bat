@echo off
chcp 65001 >nul
cd /d "%~dp0"

set PYTHONIOENCODING=utf-8

echo.
echo ================================================
echo   RAG-RPG Yi Jian Qi Dong
echo ================================================
echo.
echo  Tip: After startup, press r + Enter to restart
echo  the main server without reloading the model.
echo.

call conda info --envs 2>nul | findstr /B /C:"rag-rpg " >nul
if %errorlevel% equ 0 (
    echo [*] Found rag-rpg env, activating...
    call conda activate rag-rpg
) else (
    call conda activate rag-rpg 2>nul
    if %errorlevel% neq 0 (
        echo [!] rag-rpg env not found, using current Python
    )
)

echo.
echo [*] Starting services...
echo.

python scripts/start_services.py

if %errorlevel% neq 0 (
    echo.
    echo [!] Service exited with error, check logs above
    pause
)
