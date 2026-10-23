# Session Summary: Session_2026-10-23_201556

## Metadata
- **Date:** 2026-10-23
- **Time:** ok. 17:30–20:15 (ciąg dalszy sesji z 00:18 i 02:36 tego dnia)
- **Status:** Completed
- **Type:** Mixed (Analysis + Tools)
- **Branch:** `feature/rzuty-192d-okienka-gpu` (nic nie zacommitowane w tej sesji)

## Objective
1. Wyjaśnić użytkownikowi w języku jego układu (lasery, koloidy, gwiazda, pole) co znaczą wyniki płaszczyzny krok × podkrok:
   „nadajnik kontekstu”, czy uwaga ustawia tokeny pod MLP 4, gdzie MLP „nie widzi” tokenów, czym jest strata, życzenie, okno odczytu,
   czy lasery to pułapki, czym jest uczenie.
2. Zmierzyć to, co z rozmowy wynikało jako sprawdzalne: lasery wybiórcze klasowo, temperatura widoczna/niewidoczna dla głowy,
   cos(v_c, w_c), asymetria jakobianów (z istniejących danych).
3. Znaleźć alegorię fizyczną całego układu i zbudować ją w aplikacji jako nowe okienko.

## Context
Wcześniej tego dnia: eksperyment `kroki_podkroki_afinicznosc_uwagi_20261023__00-41-29` (s47), replikacja
`plaszczyzna_krok_podkrok_replikacja_3047_20261023__02-16-41` (10 klatek TEST, predykcje P1–P7, 2/7 PASS), tezy i wniosek artykułu
w `SmallViTAblation/reports/`, pomiar łodygi CNN `lodyga_cnn_wplyw_na_czasteczki_20261023__17-33-38`. Użytkownik zabronił replikacji
na 5 seedach i uruchomień na Edenie.

## Actions Taken
1. Odpowiedzi pojęciowe (bez narzędzi): gwiazda/nadajnik kontekstu; złożenie uwaga → MLP (nie wypadkowa); czapa lasera jako strefa
   na sferze LN, nie pułapka; „MLP nie widzi” = ciemna strona wszystkich progów; strata jako potencjał na 5 odczytach głowy, nie na
   pozycjach; życzenie = siła sprężyny na ekranie cofnięta przez optykę i przeguby (τ = Jᵀ F); okno odczytu = rura promieni (cień);
   wagi zamrożone w przejściu, skokowe między paczkami; ciecz bez własnej dynamiki, pompa = życzenia, tłumienie = rozpad wag.
2. Szybkie pomiary z wag i zapisanych jakobianów (bez nowych folderów): odsetek zapalonych laserów (43 % → 4–19 %),
   cos(v_c, w_c) na 3 checkpointach 3047, asymetria jakobianów 0,86–1,0, stałe LR 5e-4 po 300 krokach rozgrzewki, WD 0,05.
3. Eksperyment `lasery_progi_miedzy_klasami_20261023__18-34-55` (agent Sonnet w tle): lit narzędzie/tło, klasy laserów, dekompozycja
   skoku Fishera, cos z osią narzędzie − tło, zero z przetasowania; 46 ckpt × 10 klatek TEST.
4. Eksperyment `temperatura_widoczna_dla_glowy_20261023__18-38-56` (agent Sonnet w tle, predykcja zapisana przed pomiarem):
   rozkład przesunięcia stanu końcowego między checkpointami na podprzestrzeń jakobianu głowy (5 wymiarów) i resztę; werdykt FAIL.
5. Alegoria: teatr cieni z 600 okienek → mikrosoczewki (matryca, nie jedna soczewka) → układ trzypiętrowy: zdjęcie naświetla
   półfabrykaty przez łodygę, sfera formuje, lampa i ekran odczytują. Rozróżnienie odczytu A (koloid = soczewka, lasery = maszyna)
   i B (stany = światło, bloki = szkło warstwowe, D²NN).
6. Implementacja okienka M w `LaserLatentSpaceApp`: `gpu/microlens_gl.py`, `lasers_tab/microlens_view.py`, klucz `glowa` w
   `history.podkroki`, `passage_store.wczytaj` wymaga `glowa`, podpięcie w `tab.py`, `substeps_view.py`, `data_view.py`, `layout.py`,
   `parameter_groups.py`, README, testy. Zrzuty sprawdzone wizualnie (cała scena, góra, dół, tryb różnicy, krok 0).
7. Testy: `tests.test_modules.TestMikrosoczewki`, `test_tooltips`, `TestPanelParametrow`, `TestOkna` (31 OK);
   `test_gui.TestHistoriaWidocznosc/TestPanel/TestPopupy/TestUstawienia` (17 OK, w tym nowy `test_04_okienko_m_mikrosoczewki`).

## Results

### Key Findings
- **Lasery wybiórcze są tłowe.** Blok 4 (3047): krok 0 — ~300 laserów widzi obie klasy, 0 wybiórczych; 60 ep — obie 11, tylko tło 34,5,
  tylko narzędzie 5; 400 ep — obie 9,5, tylko tło 40,5, tylko narzędzie 2; zero (przetasowane etykiety): 0 wybiórczych. Narzędzie jest po
  uczeniu ciemniejsze niż tło w każdym bloku (blok 4: 0,05 vs 0,15 zapalonych). 48 laserów wybiórczych (6 %) daje 73 % (60 ep) i 91 %
  (400 ep) skoku Fishera w MLP 4 (1,54 → 3,36; same wybiórcze 3,19; losowy podzbiór 48: 1,53). Lasery „tylko tło” pchają tło przeciwnie
  do osi narzędzie − tło (cos −0,28 / −0,23). Wniosek: rozdział robi ruch większości (tła), narzędzie stoi w cieniu progów.
- **Temperatura:** energia przesunięcia stanu końcowego między checkpointami (na 1000 kroków, / R²) spada do ~30 ep i jest płaska do
  400 ep w obu składowych (narzędzie: widoczna 3,5 → 0,0017, niewidoczna 20 → 0,017); stosunek spadków 0,8–0,9 (próg ≥ 3 niespełniony);
  udział widoczny 0,08–0,10 przy zerze losowym 0,023 (3–4× więcej energii na wymiar w kierunkach czytanych przez głowę). Linearyzacja
  0,9–1,05 od 10 ep; jakobian głowy pełnego rzędu 5 dla 276 000 tokenów. Interpretacja: stacjonarny szum przy stałym LR, strata nigdy
  nie wiotczeje do końca; powolne zmiany geometrii do 400 ep to dryf pod szumem, nie osobny kanał.
- **Pola nie są potencjalne:** asymetria jakobianów 0,86–1,0 we wszystkich 12 podkrokach i czasie; cos(v_c, w_c): średnia |cos|
  0,09–0,22 (losowo 0,06), znak: blok 1 +0,07 (60 ep), bloki 5–6 −0,08…−0,15 (siła przywracająca); laser działa lokalnie w czapie
  wokół w_c (ok. 25 % losowych kierunków, ~10 % koloidów), ale pcha w bok.
- **Uczenie:** AdamW, LR stałe 5e-4 po 300 krokach rozgrzewki, WD 0,05 — więc gaśnięcie temperatury nie jest efektem harmonogramu.

### Issues Encountered
- Pierwsze zrzuty okienka M: kamera zbyt nisko (matryca i ekran na krawędzi) → elevation 30, distance 330.
- `widac('m')` tuż po `visibilityChanged` bywa fałszywe → odroczenie `QTimer.singleShot(0)` w `_mk_widoczne`.
- Zapisane przejścia bez klucza `glowa` → `passage_store.wczytaj` zwraca None (liczone od nowa), zamiast podbijać WERSJA.
- Agent pomiaru laserów: numeracja bloków od 1 (stan przed MLP b = X[2b−1]); zero z przetasowania daje 0 wybiórczych, więc dodał
  kontrolę „losowy podzbiór tej samej liczności”.

## Files Generated/Modified
Eksperymenty (`SmallViTAblation/experiments/`):
- `lasery_progi_miedzy_klasami_20261023__18-34-55/` (Kod, Opis, Wyniki/Dane: `lasery_klasy.npz/json`, `tabele_odczyt.txt`, `log.txt`)
- `temperatura_widoczna_dla_glowy_20261023__18-38-56/` (Kod, Opis z predykcją przed pomiarem, Wyniki/Dane: `temperatura_glowy.npz/json`,
  `tabele_odczyt.txt`, `log.txt`)
- `experiments_index.md` (+2 linie)

Aplikacja (`SmallViTAblation/LaserLatentSpaceApp/`):
- nowe: `laser_app/gpu/microlens_gl.py`, `laser_app/lasers_tab/microlens_view.py`
- zmienione: `laser_app/compute/history.py` (klucz `glowa`), `laser_app/compute/passage_store.py`, `laser_app/lasers_tab/tab.py`,
  `substeps_view.py`, `data_view.py`, `layout.py`, `laser_app/ui/parameter_groups.py`, `README.md`, `tests/test_gui.py`,
  `tests/test_modules.py`

Sesja: `.sessions/Session_2026-10-23_201556/` (README, SESSION_SUMMARY, `analysis/DROGA_DO_WNIOSKOW.md`).

## Commands Used
```
W:/Python/3.11/python.exe experiments/lasery_progi_miedzy_klasami_20261023__18-34-55/Kod_eksperymentu/lasery_klasy.py Wyniki/Dane
W:/Python/3.11/python.exe experiments/temperatura_widoczna_dla_glowy_20261023__18-38-56/Kod_eksperymentu/temperatura_glowy.py Wyniki/Dane
cd LaserLatentSpaceApp; PYTHONPATH=. python -m unittest tests.test_modules.TestMikrosoczewki tests.test_tooltips
cd LaserLatentSpaceApp; PYTHONPATH=. python -m unittest tests.test_gui.TestHistoriaWidocznosc tests.test_gui.TestPanel tests.test_gui.TestPopupy tests.test_gui.TestUstawienia
```

## Next Steps
- [ ] Commit i push okienka M, eksperymentów i raportów (użytkownik nie zdecydował).
- [ ] Pomiar „lasery rzeźbiące optycznie vs tylko przestawiające”: cos(v_c, wiersze jakobianu głowy) per laser i blok (lokalnie).
- [ ] Kontrola do temperatury: czy wzbogacenie w kierunkach głowy nie wynika z pokrycia z 5 głównymi kierunkami chmury.
- [ ] Replikacja na 5 seedach i interwencja przyczynowa (zastąpienie uwagi średnią) — odłożone na polecenie użytkownika.
- [ ] Ewentualny audyt Undermind: uwaga jako ośrodek nielokalny / fotoniczne transformery, D²NN w detekcji.
