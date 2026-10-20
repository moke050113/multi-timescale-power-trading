@echo off
echo ========================================
echo  Push to GitHub
echo ========================================
echo.

cd /d D:\Qoder_project

echo Setting remote URL...
git remote set-url origin https://github.com/moke050113/multi-timescale-power-trading.git

echo Renaming branch to main...
git branch -M main

echo.
echo Pushing to GitHub...
echo You will be prompted for:
echo   Username: moke050113
echo   Password: [Your Personal Access Token]
echo.

git push -u origin main

echo.
echo ========================================
echo  Done!
echo ========================================
pause
