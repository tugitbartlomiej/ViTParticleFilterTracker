# Session Summary: Rzuty z 192 wymiarów, okienka R1–R9 na GPU, dobór osi separacji, commit

## Metadata
- **Date:** 2026-10-21
- **Time:** ok. 10:30 – 18:40 (Europe/Warsaw), po zapisie `Session_2026-10-21_102949`
- **Status:** Completed
- **Type:** Mixed (Analysis + Software + Literature Research)

## Objective
- Zobaczyć wędrówkę cząsteczek (narzędzie, tło, wybrane) w czasie uczenia w rzutach 3D z wiarygodnym miernikiem wierności.
- Zbudować w aplikacji okienka rzutów z 192 wymiarów (osobne okienka z własnymi parametrami), liczone i rysowane na GPU.
- Znaleźć metodologię doboru osi dające najlepsze, mierzalne rozdzielenie narzędzia od tła, zweryfikowane literaturą (Undermind; Scite wyczerpany).
- Zrobić commit i odbić gałąź.

## Context
Poprzednie części: `Session_2026-10-21_045038` (przeciwne wiązki, dynamika koloidów) i `Session_2026-10-21_102949` (pole pułapek, fizyka chmury, okluzja, publikacje). Użytkownik poprosił o rzuty „jak najlepiej”, potem o wędrówkę i rozdzielenie narzędzia od tła, a na końcu o commit i zapis sesji.

## Actions Taken
1. Przeniesienie raportów i notatek z katalogu głównego do `SmallViTAblation/reports` i `research_notes`.
2. Trzy badania Undermind o projekcjach liniowych/nieliniowych/dynamicznych oraz wektorach sił, potem raport „Rzutowanie wielowymiarowego ruchu do 3D” z przepisem dla naszych widoków.
3. Moduł `rzuty_dane.py` (stany 192D, bazy, mierniki, rama siły, rama Schura, tory zerowe) i sekcja „Rzuty z 192 wymiarów” z okienkami R1–R8 (`rzuty_okienka.py`).
4. Element shaderowy `rzuty_gpu.py` z drugiej sesji (poprawiony błąd śladu, dodany bufor wygładzonych położeń); torch na karcie dla rzutowania historii.
5. Testy dymne bez okna (wszystkie opcje, 0 błędów) i zrzuty OpenGL; naprawa kadru kamery (narzędzie wypadało poza ekran).
6. Badanie separacji narzędzia od tła: pomiary 192D i kilku baz, LOO, stabilność w czasie, wpływ brzegu ramki; dwa researche Undermind (supervised projections, miary separacji i walidacja).
7. Okienko R9 „L” z sekcją parametrów (λ, pula checkpointów, dopasowanie na końcu lub bieżącym, osie 2–3, brzeg ramki, permutacje), testy (8/8) i raport o doborze osi.
8. Commity: `0f0d2b3d` (rzuty, pole pułapek, raporty), `cff8ece6` (R9, raport o doborze osi), poprawka końców linii LF.

## Results

### Key Findings
- Na ostatnim checkpoincie tokeny narzędzia są w 192D umiarkowanie odseparowane (sylwetka 0,52, trafienia kNN 0,71); P3 z aplikacji jest na poziomie przypadku (0,02 / 0,21).
- Oś LDA ze skurczem (λ = 0,1), dopasowana na 100 ostatnich checkpointach, daje 0,86 / 0,90 / margines 3,0 (zera 0,11 / 0,06 / 0,5). Bez etykiet PCA chmury daje już 0,85 / 0,88, więc rozdział nie jest wyłącznie efektem dopasowania.
- Brzeg prostokątnej ramki GT zawiera piksele tła; po jego pominięciu 0,91 / 0,97, margines 7,6.
- Osie 2–3 najmniejszego rozrzutu i PCA narzędzia zawodzą (zgodnie z zjawiskiem spiętrzania danych, Ahn–Marron).
- Stabilność: separacja utrzymuje się w osiach z końca uczenia do ok. checkpointu 200; kąty główne 9°/14°/30°.
- Literatura: dla dwóch klas i n₁ < d zalecany przepis to LDA ze skurczem + LOL (Vogelstein i in. 2021) lub PDA-index; zero musi refitować oś przy każdej permutacji (DiProPerm); walidacja blokowana po checkpointach.

### Issues Encountered
- Rzut D (ruch) nie rozdziela klas; osobne okienko R9 z własnymi parametrami.
- Kadr kamery z pierwszego checkpointu; naprawione kadrem z całej historii.
- CRLF z zapisu na Windowsie; poprawione commitem LF.
- Scite niedostępny (limit 250 wywołań do 13.11.2026); badania oparte na Undermind.
- Tokeny brzegu ramki: 4 z 6 rozstrzygnięte w stronę tła przy progu 1, 2 w stronę narzędzia (nie po połowie, jak wstępnie opisano).

## Files Generated/Modified
- `SmallViTAblation/app/rzuty_dane.py`, `rzuty_okienka.py`, `rzuty_gpu.py`, `testy/test_rzuty_dane.py`, `README.md`, `zakladka_laserow.py`, `ustawienia.py`
- `SmallViTAblation/reports/Rzutowanie wielowymiarowego ruchu do 3D.md`, `Osie najlepszej separacji narzędzia od tła.md`, `Fizyka chmury koloidów i okluzja.md`
- `SmallViTAblation/research_notes/Rzutowanie…/` (4 pliki), `Osie najlepszej separacji…/` (3 pliki), `Fizyka chmury…/` (3 pliki)
- `SmallViTAblation/experiments/koloidy_wiazki_dynamika_0_200_20261021__01-48-10/` (pola_pulapek.py, pulapki_newton_diag.py, pokaz_pola_pulapek.py, POLA_PULAPEK.csv)
- `.sessions/Session_2026-10-21_102949/`, `.sessions/Session_2026-10-21_184029/`
- Pamięć: `reports-location-smallvitablation.md` (raporty do `SmallViTAblation/`)

## Commands Used
- `git switch -c feature/rzuty-192d-okienka-gpu develop`
- `git commit -F` (`0f0d2b3d`, `cff8ece6`, poprawka LF)
- `W:/Python/3.11/python.exe` (Qt offscreen, testy dymne, zrzuty), `python -m unittest` dla testów rzutów

## Next Steps
- [ ] Zero DiProPerm (refit osi w każdej permutacji) i walidacja blokowana po checkpointach w oknie R9.
- [ ] Replikacja separacji na drugiej klatce TEST i na 24 seedach.
- [ ] Etykieta narzędzia jako udział pikseli w komórce, przed dopasowaniem.
- [ ] Ręczna weryfikacja 13 twierdzeń z raportu przed cytowaniem.
- [ ] Decyzja użytkownika o dodaniu zera DiProPerm do okienka.
