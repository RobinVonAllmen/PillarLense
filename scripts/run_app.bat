@echo off
setlocal
set "ROOT_DIR=%~dp0.."
set "APP=%ROOT_DIR%\.venv\Scripts\pillar-lense.exe"

if not exist "%APP%" (
  echo PillarLense is not installed in %ROOT_DIR%\.venv.
  echo Create the environment first with:
  echo   py -m venv .venv
  echo   .venv\Scripts\python -m pip install --upgrade pip setuptools wheel
  echo   .venv\Scripts\python -m pip install -e .
  exit /b 1
)

"%APP%" %*
