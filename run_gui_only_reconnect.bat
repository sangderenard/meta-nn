@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Launch only the standalone GUI viewer.
REM Intended for reconnecting to an already-running backend process.

set "PYTHON=python"
set "GUI_SCRIPT=wav_ml_gui_main.py"
set "OUTPUT_DIR=wav_pipeline_runs\iterative_rgb256"
set "IMAGE_SIZE=256"
set "GUI_SCALE=1"
set "CYCLE_SLOTS=8"
set "VIEWER_PORT_FILE=%OUTPUT_DIR%\.viewer_port"

REM Optional first arg overrides output dir.
if not "%~1"=="" set "OUTPUT_DIR=%~1"
set "VIEWER_PORT_FILE=%OUTPUT_DIR%\.viewer_port"

if not exist "%GUI_SCRIPT%" (
  echo [gui-only] GUI script not found: %GUI_SCRIPT%
  exit /b 1
)

if not exist "%OUTPUT_DIR%" (
  mkdir "%OUTPUT_DIR%" >nul 2>&1
)

set "GUI_PORT="
if exist "%VIEWER_PORT_FILE%" (
  for /f "usebackq delims=" %%P in ("%VIEWER_PORT_FILE%") do set "GUI_PORT=%%P"
)

set "GUI_ARGS=--output-dir "%OUTPUT_DIR%" --image-size %IMAGE_SIZE% --scale %GUI_SCALE% --cycle-slots %CYCLE_SLOTS% --port-file "%VIEWER_PORT_FILE%""
if not "!GUI_PORT!"=="" (
  echo [gui-only] reusing existing port !GUI_PORT! from %VIEWER_PORT_FILE%
  set "GUI_ARGS=!GUI_ARGS! --port !GUI_PORT!"
) else (
  echo [gui-only] no existing port file; GUI will auto-assign a port.
)

echo [gui-only] launching standalone GUI...
%PYTHON% %GUI_SCRIPT% !GUI_ARGS!
set "GUI_RC=%ERRORLEVEL%"
echo [gui-only] GUI exited with code !GUI_RC!
exit /b !GUI_RC!
