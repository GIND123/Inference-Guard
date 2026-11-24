@echo off
setlocal
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "FALLBACK_PY=C:\Users\annat\AppData\Roaming\uv\python\cpython-3.12.15-windows-x86_64-none\python.exe"

"%VENV_PY%" --version >nul 2>&1
if %errorlevel% equ 0 (
    "%VENV_PY%" scripts\launch_presentation_app.py
) else (
    set "PYTHONPATH=%~dp0;%~dp0.venv\Lib\site-packages"
    "%FALLBACK_PY%" scripts\launch_presentation_app.py
)
endlocal
