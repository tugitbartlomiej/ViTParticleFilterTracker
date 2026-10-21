# Session: Rzuty z 192 wymiarów, okienka R1–R9 na GPU, dobór osi separacji, commit

**Created:** 2026-10-21 18:40
**Type:** Mixed (Analysis + Software + Literature Research)
**Status:** Completed (commit `cff8ece6` + poprawka końców linii)

## Quick Summary
Kontynuacja `Session_2026-10-21_102949`. Zbudowano w zakładce „Lasery i koloidy” sekcję „Rzuty z 192 wymiarów” z okienkami R1–R9: mapa klas, siły, wiązki ważone, rama siły, rama Schura, wędrówka w czasie uczenia, 7 widoków Fano, panel bez rzutu i separacja narzędzia od tła (LDA ze skurczem). Liczenie stanów 192D i rzutowanie całej historii idzie na kartę (torch), rysowanie w shaderach GLSL 4.30 (`rzuty_gpu.py`, zgodnie z uwagą drugiej sesji). Dwa researche Undermind o doborze osi i miarach separacji dały raport z przepisem do artykułu. Zmiany scommitowane na gałęzi `feature/rzuty-192d-okienka-gpu`.

## Key Results
- Separacja narzędzia od tła na ostatnim checkpoincie (806, krok 65640, 18 tokenów narzędzia): w 192D sylwetka 0,52, trafienia kNN 0,71; oś LDA (λ = 0,1, pula 100 checkpointów) 0,86 / 0,90, margines 3,0 (zera permutacyjne 0,11 / 0,06 / 0,5). Obecny P3 z aplikacji: 0,02 / 0,21 — na poziomie przypadku.
- Część przewagi LDA to usunięcie wymiarów z szumem wewnątrz klas. Bez czysto wykrywalnego artefaktu: PCA chmury bez etykiet już daje 0,85 / 0,88.
- Brzeg prostokątnej ramki GT zawiera tokeny tła (sieć sama tak klasyfikuje); po ich pominięciu separacja rośnie do 0,91 / 0,97, margines 7,6.
- Osie 2–3 „najmniejszego rozrzutu” i „PCA narzędzia” zawodzą (spiętrzanie danych przy n < d), zgodnie z literaturą.
- Stabilność: osie z końca uczenia zachowują separację do ok. checkpointu 200; kąty główne 9°/14°/30° względem puli 100.
- Pełna historia (806 checkpointów) liczy się ok. 2,3 min na GPU; 360 MB na zdjęcie + 227 MB wspólne (V.npy).

## Files
- `SmallViTAblation/app/rzuty_dane.py`: stany 192D, bazy A/A_t/S/W/D/F7/P3/L, mierniki wierności, rama siły, rama Schura, tory zerowe, miary separacji i zera permutacyjne.
- `SmallViTAblation/app/rzuty_okienka.py`: okienka R1–R9, sekcja parametrów (w tym L: λ, pula, osie 2–3, brzeg ramki, permutacje), kadr z całej historii.
- `SmallViTAblation/app/rzuty_gpu.py`: element GLSL 4.30 (przeniesiony z drugiej sesji, poprawiony: ślad pomijał bieżący checkpoint; bufor PS dla śladów wygładzonych).
- `SmallViTAblation/app/testy/test_rzuty_dane.py`: 8 testów (Fano, jakobian, rama Schura, rama siły, sumy, separacja L, rdzeń ramki).
- `SmallViTAblation/app/README.md`: opis R1–R9.
- `SmallViTAblation/reports/Osie najlepszej separacji narzędzia od tła.md` + `research_notes/Osie najlepszej separacji narzędzia od tła/` (3 pliki): dobór osi, miary separacji, walidacja, przepis do artykułu.
- `SmallViTAblation/reports/Rzutowanie wielowymiarowego ruchu do 3D.md` i `Fizyka chmury koloidów i okluzja.md` (przeniesione z katalogu głównego).

## Commands Used
- `git switch -c feature/rzuty-192d-okienka-gpu develop` (gałąź z `develop` fast-forward do `0f0d2b3d`)
- `git commit -F …` (commit `0f0d2b3d`, `cff8ece6`, poprawka LF)
- `W:/Python/3.11/python.exe` z `QT_QPA_PLATFORM=offscreen` do testów dymnych zakładki i zrzutów okienek
- `W:/Python/3.11/python.exe -m unittest` dla `testy/test_rzuty_dane.py` (8/8 OK)

## Issues Encountered
- Rozdzielenie narzędzia od tła w R6 (rzut D) było słabe: osie dobrane pod ruch, nie pod klasę. Rozwiązane osobnym oknem R9.
- Kadr kamery liczony z pierwszego checkpointu wypychał narzędzie poza ekran; naprawione kadrem z całej historii.
- Pliki zapisane przez `Path.write_text` na Windowsie dostały CRLF; poprawiono commitem z LF.
- ustawienia.py miał CRLF (poprzednia sesja), znormalizowany.
- Scite: limit miesięczny 250 wywołań (odnowi się 13.11.2026); badania oparte na Undermind i pełnych tekstach.
- Nie zrobiono jeszcze: zera permutacyjnego z ponownym dopasowaniem osi (DiProPerm), walidacji blokowanej po checkpointach, miar GONG/DSC, testu na drugiej klatce.
- Niezacommitowane zmiany innych sesji w `.sessions/` i `SmallViTAblation/` (nie moje) zostały poza commitem.

## Next Steps
- [ ] Dodać do R9 zero z ponownym dopasowaniem osi przy każdej permutacji (DiProPerm) i walidację krzyżową blokowaną po checkpointach.
- [ ] Policzyć separację na drugiej klatce TEST i na 24 seedach (replikacja).
- [ ] Ustalić etykietę narzędzia jako udział pikseli w komórce, przed dopasowaniem.
- [ ] Sprawdzić ręcznie wymienione w raporcie 13 twierdzeń przed cytowaniem.
- [ ] Decyzja użytkownika: czy dodać zero DiProPerm do okienka (koszt: ok. 100 dopasowań LDA na pomiar).
