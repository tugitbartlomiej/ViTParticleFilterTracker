# Podsumowanie sesji 2026-10-06 22:08:07

## Cel sesji

Rozbudowa aplikacji edytora latentu małego ViT (`SmallViTAblation/app`) o kolejne sposoby wizualizacji
latentu na sekwencjach filmu, przebudowa interfejsu (czytelny układ, hierarchia przycisków, wygodne pola),
zapis ustawień, odpinanie widoków 3D i auto-szukanie latentu w wybranych patchach. Ocena, czy mały ViT
z 8 tokenami da się wytrenować lokalnie.

## Kontekst

- Repozytorium: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`, gałąź `codex/eden-nano-training-20260914`.
- Poprzednia część tej samej rozmowy: `.sessions\Session_2026-10-06_171603_Aplikacja_Latentu_Detekcja_Czasteczki_3D_Film_Makrowektory`
  (commit `56d36a2d`).
- Równoległa sesja Claude zacommitowała w międzyczasie stan refactoru jako `4b1f008f` („punkt wyjścia dla nowej
  wersji”); jej plików nie ruszano.
- Model ViT aplikacji: `outputs/vit_latent128_jedno_zdjecie_90/MODEL_VIT_K1000.pt`, latent 600 x 768 (wyjście
  MLP ostatniego bloku, za nim nie ma atencji).

## Wykonane kroki

1. Widok cząsteczek 3D: domyślnie pełna kula kierunków (wcześniej półsfera); środki cząsteczek na płaszczyźnie
   obrazu, obraz półprzezroczysty (pole „krycie”, domyślnie 0,5; `ObrazPrzezroczysty` w `widok_gl.py` bez zapisu
   do bufora głębi), sprawdzone zrzutem OpenGL.
2. Przestrzeń makrowektorów: mikrowektory wybranego patcha ze wspólnego środka plus biała wypadkowa (domyślnie),
   łańcuch jako opcja; suma 768 mikrowektorów trafia dokładnie w cząstkę (sprawdzone).
3. Szybszy start: `start.py` z okienkiem ładowania (pierwszy start do minuty przez import torch, timm,
   torchvision z zimnego dysku; potem około 7 s); błąd startu w oknie i `start.log`.
4. Exe: `Edytor latentu.exe` (48 kB, C# z csc.exe .NET 4, źródło i ikona w `app/uruchamiacz/`), uruchamia
   `F:\Python\3.13\pythonw.exe start.py` z torchem 2.10 i CUDA 12.6; `python.txt` nadpisuje ścieżkę Pythona.
5. Refactor bez zmiany działania: `wspolne.py` (siatka, kolory, kierunki, PCA), `widok_gl.py`,
   `widzety_edytora.py`, `dodaj()` w `uklad.py`; test odcisku 24 pozycji identyczny przed i po.
6. PCA wektorów w obu widokach 3D (3, 4, 8, 16, 32, 128 składowych; w filmie baza z całej sekwencji).
   TR_0001 0:20–0:30: wariancja 57,9 / 63,7 / 75,8 / 86,2 / 92,5 / 98,3 %, błąd odtworzenia klatki 0,64 do 0,13.
7. Mniejsze ViT: odpowiedź (MobileViT, TinyViT, LeViT; TokenLearner 8 tokenów, TiTok 32 tokeny). TokenLearner
   (Ryoo i in. 2021, arXiv 2106.11297) przeczytany w całości przez Undermind (workspace ViT, klucz Ryo21):
   sigmoid, wstawienie w połowie sieci, 8 i 16 tokenów jednakowo, TokenFuser do powrotu na siatkę, praca bada
   tylko klasyfikację. Pomiar na RTX 3070 (paczka 32, bf16): MalyViT 54,9 ms na krok, TokenLearner 8 po bloku 3
   41,6 ms. Plan: `SmallViTAblation/experiments/maly_vit_tokenlearner_8/PLAN.md` (czeka na trzy decyzje).
8. `wytnij_narzedzie.py` i folder `app/narzedzie_czarne_tlo/TR_0001_S1_P03_00m20.0s_400kl` (400 klatek PNG
   720x480, poza obrysem dokładnie czarne, maski i `MANIFEST.json`).
9. Liczba cząsteczek co n-ty patch (600, 150, 70, 35, 24, 12, 6) w obu widokach 3D.
10. Odjęcie położenia patcha (`polozenie.py`): średni latent każdego patcha z 3000 klatek walidacji, liczony raz
    (15 s) i zapisywany do `outputs/latent_editor_sd_20261006/POLOZENIE_*.npz`. Położenie = 39,2 % wariancji
    latentu, 13 kierunków = 95 % wkładu. R² odczytu kolumny / wiersza z położenia cząstki: 0,76 / 0,72 bez
    odjęcia, 0,01 / 0,00 po średniej z sekwencji, 0,46 / 0,26 po średniej z 3000 klatek, 0,39 / 0,23 po rzucie.
11. Folder zdjęć jako źródło sekwencji (z MANIFEST.json etykiety pasują do klatek); po wyborze folderu konwersja
    rusza sama. Na samym narzędziu na czarnym tle: pewność ViT 45,1 %, IoU 65,0 % (pełne klatki: 71,0 / 72,5 %).
12. Śledzenie czubka igły: patch czubka z etykiety (w klatce bez czubka najpewniejszy patch ViT) albo
    najpewniejszy patch ViT; na 40 klatkach patch z etykiety zmienia się 10 razy, patch ViT leży średnio
    5,4 patcha od czubka.
13. Przebudowa GUI w krokach (każdy osobny commit, test odcisku po każdym): opcje widoków w zakładkach
    (`PanelOpcji`), role przycisków (`styl.py`), Edytor SD, film, detekcja (ciemne wykresy, okno mieści się
    w 1700 px), widoki cząsteczek i atencji (`WaskiCombo`), wspólne etykiety opisu i stanu, podział kolumn filmu.
14. Wygoda (`wygoda.py`): kółko zmienia pole tylko po kliknięciu (inaczej przewija panel), liczby zatwierdzane
    Enterem, przyspieszanie strzałek (262 pola, 55 list, 317 suwaków).
15. Ustawienia (`ustawienia.py`, menu „Ustawienia”): 158 kontrolek do `app/ustawienia.json`, wczytywanie przy
    starcie, zapis przy zamknięciu, zapisz jako, wczytaj, przywróć domyślne; `EDYTOR_BEZ_USTAWIEN=1` dla testów.
16. Odepnij / Przypnij (`uklad.Odpinanie`): żywy widok 3D do osobnego okna i z powrotem (film: oba widoki,
    detekcja: widok 3D). Wymagane `AA_ShareOpenGLContexts`: bez niego GL_INVALID_VALUE przy glUseProgram.
17. Auto-szukanie w wybranych patchach (`tokeny` w `Detektor.optymalizuj`, lista „Patche”). TR_0001 kl. 870,
    J128, 1500 kroków: 600 / 40 / 72 / 1 patch dają tę samą pewność 84,1 % i IoU 99,9–100 %. Wyjaśnienie:
    latent leży za ostatnią atencją, więc patch zmienia tylko własne wyjście detekcji.

## Pliki zmienione / utworzone

### W repozytorium ViT (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`)

- `SmallViTAblation\app\`: `app.py`, `start.py`, `uruchom.bat`, `widzety_edytora.py`, `wspolne.py`, `widok_gl.py`,
  `uklad.py`, `styl.py`, `wygoda.py`, `ustawienia.py`, `polozenie.py`, `film.py`, `detekcja.py`,
  `zakladka_detekcji.py`, `zakladka_filmu.py`, `czasteczki.py`, `czasteczki3d.py`, `przestrzen_makro.py`,
  `opcje_atencji.py`, `wytnij_narzedzie.py`, `README.md`, `.gitignore`, `uruchamiacz\Uruchamiacz.cs`,
  `uruchamiacz\ikona.ico`, `uruchamiacz\ikona.png`, `Edytor latentu.exe` (poza gitem),
  `narzedzie_czarne_tlo\TR_0001_S1_P03_00m20.0s_400kl\` (PNG poza gitem, `MANIFEST.json` w gicie).
- `SmallViTAblation\experiments\maly_vit_tokenlearner_8\PLAN.md`.
- `SmallViTAblation\outputs\latent_editor_sd_20261006\POLOZENIE_*.npz` (poza gitem).
- Commity: `433b1e75`, `aeb0e6bd`, `0e6f50f4`, `3fd8e83f`, `6b76ef0a`, `63b9f4d6`, `4a79f964`, `2aba4958`,
  `aec2fa11`, `50fc62cd`, `bbf93145`.

### Zewnętrzne (poza repozytorium)

- Pamięć Claude: `C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\memory\vit-tokenlearner-8-plan.md`.
- Skrypty testów i zrzuty: katalog roboczy sesji w `%TEMP%\claude\...\scratchpad` (nieprzechowywany).
- Nic nie zmieniano w `external/` ani w projekcie grantowym.

## Otwarte problemy

- Plan TokenLearner czeka na decyzje: które ramiona, 1000 czy 10 000 kroków, MLP czy cztery konwolucje.
- Pola procentowe długości w widokach cząsteczek bez miejsc po przecinku (`setDecimals(0)`); do cofnięcia na życzenie.
- Folder zdjęć uruchamia konwersję automatycznie; użytkownik zgłosił, że wczytywanie działało już wcześniej.
- Test odcisku i skrypty testów żyją tylko w katalogu roboczym sesji; warto przenieść je do repozytorium.

## Następne kroki

- Po decyzji: uruchomić plan `maly_vit_tokenlearner_8` lokalnie (jeden bieg naraz, proces odłączony).
- Auto-szukanie „każdy patch osobno”: ranking patchy wg osiągalnej pewności (ograniczone do ramki).
- Użytkownik ogląda nowy układ GUI na ekranie (`Edytor latentu.exe`).
