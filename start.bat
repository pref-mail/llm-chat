@echo off

:: 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python。请先安装Python 3.8或更高版本。
    pause
    exit /b 1
)

:: 检查Ray是否安装（服务器已安装，无需再次安装）
python -c "import ray" >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Ray。请联系管理员在服务器上安装Ray。
    pause
    exit /b 1
)

:: 安装其他项目依赖（不包括Ray）
pip install -r requirements.txt

:: 启动服务器
echo 正在启动LLM聊天系统...
python app.py

pause