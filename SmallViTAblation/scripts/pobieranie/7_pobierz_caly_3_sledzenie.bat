@echo off
rem ============================================================
rem  KROK 7 - CALY katalog 3_Object_Tracking
rem
rem  ROZMIAR: okolo 51 GB  -  SPRAWDZ MIEJSCE NA DYSKU PRZED STARTEM
rem     annotations/   170 archiwow, razem 45,1 GB
rem                    (mediana 214 MB, najwiekszy 1,5 GB)
rem     videos/        3 archiwa, razem 5,6 GB
rem
rem  CO DAJE: 170 klipow, 469 118 klatek; ramki klatka po klatce,
rem           trwale numery obiektow ORAZ wspolrzedne koncowki narzedzia
rem
rem  UWAGA: wszystkie pliki maja w nazwie S1, czyli szpital Farabi,
rem         rozdzielczosc 720x480. Nie ma tu pelnego HD.
rem ============================================================
setlocal
cd /d "%~dp0"
call _config.bat

echo.
echo === UWAGA: to pobierze okolo 51 GB ===
echo Katalog docelowy: %DEST%
echo.
echo Jesli chcesz tylko probke, zamknij to okno i uruchom
echo 4_pobierz_sledzenie_probka.bat  (dwa klipy, okolo 100 MB).
echo.
pause

echo.
echo === Pobieram caly katalog 3_Object_Tracking ===
echo To potrwa dlugo. Przerwanie nie szkodzi - kolejne uruchomienie
echo dociagnie tylko brakujace pliki.
echo.
hf download %REPO% --repo-type dataset --local-dir "%DEST%" --include "3_Object_Tracking/*"
if errorlevel 1 goto blad

echo.
echo === Gotowe ===
echo Katalog: %DEST%\3_Object_Tracking
goto koniec

:blad
echo.
echo !!! Blad pobierania albo brak miejsca. Sprawdz logowanie: hf auth whoami

:koniec
echo.
pause
endlocal
