@echo off
rem Kampania petli malego ViT od zera: 5 ziaren po kolei, kazde z osobnym logiem (PLAN.md).
set PYTHONUTF8=1
set PY=F:\Studia\PhD_projekt\VIT\venvs\rfdetr_1_10_1\Scripts\python.exe
cd /d %~dp0
for %%s in (42 43 44 45 46) do (
  "%PY%" -u petla.py --ziarno %%s > wyniki\petla_s%%s.log 2> wyniki\petla_s%%s.err
)
echo koniec > wyniki\KAMPANIA_KONIEC.txt
