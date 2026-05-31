@echo off
setlocal
set "ROOT_DIR=%~dp0.."
set "VENV_DIR=%ROOT_DIR%\.venv"

if defined PYTHON (
  set "PYTHON_BIN=%PYTHON%"
) else (
  set "PYTHON_BIN=py"
)

if not exist "%VENV_DIR%" (
  "%PYTHON_BIN%" -m venv "%VENV_DIR%"
  if errorlevel 1 exit /b %errorlevel%
)

"%VENV_DIR%\Scripts\python" -m ensurepip --upgrade
if errorlevel 1 exit /b %errorlevel%
"%VENV_DIR%\Scripts\python" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 exit /b %errorlevel%
"%VENV_DIR%\Scripts\python" -m pip install -e "%ROOT_DIR%"
if errorlevel 1 exit /b %errorlevel%

echo.
echo PillarLense virtual environment is ready at:
echo   %VENV_DIR%
echo.
echo Open the app with either:
echo   scripts\run_app.bat
echo.
echo or:
echo   .venv\Scripts\activate
echo   pillar-lense
