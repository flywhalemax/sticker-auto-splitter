@echo off
chcp 65001 >nul
echo ========================================
echo    表情包自动分割工具
echo ========================================
echo.

if "%~1"=="" (
    echo ❌ 请将图片拖拽到此 BAT 文件上运行！
    pause
    exit /b
)

echo 📂 处理图片: %~nx1
echo.

python "%~dp0sticker_splitter.py" "%~1"

echo.
echo ✅ 完成！贴纸已保存到 output_stickers 文件夹
echo.
pause
