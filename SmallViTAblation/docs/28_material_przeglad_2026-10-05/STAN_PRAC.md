# Stan prac nad przeglądem „jak badać to w SmallViTAblation” (przerwane 2026-10-05 przez limit użycia)

## Zlecenie
Użytkownik: poprzedni przegląd trafił do projektu grantowego i opierał się na audycie; ma powstać największy przegląd
osadzony w `SmallViTAblation`: co tu zrobiono, jak ta sama metoda zachowywała się w szeregach czasowych i jak to dalej
badać tutaj. Planowany dokument: `docs/28_...md` (jeszcze nie napisany).

## Co jest gotowe
- Streszczenia pomocników zapisane w tym folderze: 01_linia_rfdetr_21-28_wrzesnia.md, 03_maly_vit_1-5_pazdziernika.md, 04_aparat_w_szeregach_czasowych.md, 05_inwentarz_kodu_i_danych.md.
- Poprzedni przegląd literatury i liczby: `F:\Studia\MCHTR\Grant\Projekt\Research\syntheses
it-idealny-latent-sd-langevin-dobor-danych-2026-10-05.md`
  oraz `...\Research\syntheses\materialy
it-idealny-latent-sd-langevin-2026-10-05\` (OSIAGALNOSC.json, TELEMETRIA43_TOZSAMOSC.json).
- Nowy eksperyment `experiments\maly_vit_odpowiedz_latentu_na_paczke` (PLAN.md z kryteriami, `kod\odpowiedz.py`).
  Wykonano TYLKO próbę kodu na 100 startach (`wyniki\proba\ODPOWIEDZ.json`); to nie jest wynik.
  Pełny bieg: `cd kod` i `py -3.11 odpowiedz.py --wyniki ../wyniki` (około 10-15 minut, około 2 GB RAM).
  Z próby do sprawdzenia: R(1) odczytane z regresji h = 50 wyszło -0,82 przy zaledwie 100 startach (53 regresory),
  podczas gdy regresja h = 1 daje umiejętność 0,956; przed wnioskami porównać R z h = 1, 10, 20, 50 na pełnym biegu.
- Trzy nowe wyszukiwania Undermind (workspace ViT) uruchomione 2026-10-05, wyniki do odczytania:
  „Step-to-step oscillation of representations under AdamW and what controls it”,
  „Tracking a concept direction through training when the representation rotates”,
  „What is predictable about per-example training dynamics and how it is used to choose data”.

## Co zostało
1. Odczytać trzy wyszukiwania i (jeśli brak plików 02 i 05) powtórzyć dwa streszczenia.
2. Uruchomić pełny bieg funkcji odpowiedzi i opisać go w WYNIKI.md.
3. Napisać `docs/28`: aparat z szeregów czasowych, tabela przekładu pojęć (cząstka, zegar, wejście, tryb uczciwy i
   warunkowy), mapa 56 rodzin eksperymentów według zegara (czas wideo, czas uczenia), co się powtórzyło, a co nie,
   literatura, program badań w tym repozytorium z kryteriami, co do publikacji.

## Główna oś porównania (do dokumentu)
- Szeregi czasowe, zegar danych: prognoza uczciwa 79,8, warunkowa 90,4 (stary test); luka to nieznana sprzedaż jutra.
- Obrazy, zegar uczenia: prognoza uczciwa 0,22 na 10 kroków, z prawdziwą siłą każdego kroku 0,985; luka to nieznane paczki.
- W obu dziedzinach dobór danych po wielkości błędu albo po sygnale z latentu nie pobił losowania; schemat grantu każe
  celować w część usuwalną błędu i w przyczynę, nie w wielkość.
