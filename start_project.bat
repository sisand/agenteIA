@echo off
cd /d C:\Dados\agenteIA
call venv\Scripts\activate
echo Ambiente virtual ativado. Pronto para trabalhar!
git status
flake8 .
cmd
