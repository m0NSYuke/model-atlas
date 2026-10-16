@echo off
cd /d "%~dp0"
where npm >nul 2>nul
if errorlevel 1 (
  echo Node.js was not found. Install Node.js 22.6 or newer from https://nodejs.org and run this again.
  pause
  exit /b 1
)
if not exist "node_modules\" (
  echo Installing dependencies for the first run... ^(npm install^)
  call npm install
  if errorlevel 1 (
    echo npm install failed. See the messages above.
    pause
    exit /b 1
  )
)
echo Starting 3D_Model dev server... (close this window to stop)
call npm run dev -- --open
pause
