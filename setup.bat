@echo off
setlocal

echo =========================================
echo 🔄 Removendo ambiente virtual antigo...
echo =========================================
rmdir /S /Q venv-agent

echo =========================================
echo 🧪 Criando novo ambiente virtual...
echo =========================================
python -m venv venv-agent

echo =========================================
echo 🚀 Ativando ambiente virtual...
echo =========================================
call venv-agent\Scripts\activate.bat

echo =========================================
echo 🔧 Atualizando pip e setuptools...
echo =========================================
python -m pip install --upgrade pip setuptools

echo =========================================
echo 📦 Instalando dependências do requirements.txt...
echo =========================================
pip install --no-cache-dir -r requirements.txt

IF %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ Ambiente instalado com sucesso!
) ELSE (
    echo.
    echo ❌ Ocorreu um erro durante a instalação.
)

endlocal
pause
