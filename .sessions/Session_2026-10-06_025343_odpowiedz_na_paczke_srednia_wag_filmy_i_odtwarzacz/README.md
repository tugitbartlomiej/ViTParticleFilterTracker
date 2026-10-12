# Sesja 2026-10-05 .. 2026-10-06: odpowiedź latentu na paczkę, średnia wag, dwa filmy i odtwarzacz punktów

## Cel sesji

Dokończyć cykl pomiarów dynamiki uczenia małego ViT (odpowiedź latentu na jedną paczkę, kręgosłup trajektorii wag,
odświeżany kierunek tożsamości, uśrednianie wag) i zebrać je w przeglądzie docs/28. Potem, na prośbę użytkownika,
zainstalować video-use i zrobić filmy pokazujące pipeline doboru zdjęć oraz odtwarzacz, który zatrzymuje film
w zaznaczonych punktach.

## Typ

experiment (pomiary) oraz maintenance (filmy, narzędzie do pokazu)

## Workerzy użyci

- 7 podagentów (general-purpose) budujących sceny Manim pierwszego filmu (`animations/slot_1..7`).
- 6 podagentów (general-purpose) budujących sceny Manim drugiego filmu „Modele w ruchu” (`animations/slot_m1..m6`).

## Co zrobiono (skrót)

- Odpowiedź latentu na jedną paczkę (projekcja lokalna): rośnie do 3,2 w 5. kroku, wraca do ~0,4 po 500 krokach.
- Trajektoria wag: PC1 0,886 wobec 0,811 dla błądzenia losowego; trwały dryf mały (cos odległych okien 0,10-0,16).
- Odświeżany kierunek tożsamości: d′ ~7 stale, kierunek stały spada z 27 do 9 w 50 kroków; związek z mAP słaby.
- Średnia wag z okna 100 kroków daje +0,06 do +0,11 mAP (w kroku 1000: 0,416 wobec 0,337).
- docs/28 częściowo napisany (sekcje 1-8), sekcje 0 i 9-13 czekają.
- video-use zainstalowane (bez klucza ElevenLabs); film 1 `final.mp4` (2:52), film 2 `final_modele.mp4` (3:32).
- Odtwarzacz `odtwarzacz.html`: zatrzymywanie w punktach, edytor punktów, wysuwane menu, pełny ekran.

Szczegóły: [SESSION_SUMMARY.md](SESSION_SUMMARY.md).
