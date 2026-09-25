# Koopman dla elementów latentu identyfikujących narzędzie

Data: 2026-09-25. Zlecenie użytkownika: sprawdzić, czy elementy latentu, które identyfikują narzędzie („pełna
identyfikacja narzędzia w latencie”), da się przewidywać Koopmanem w trakcie uczenia; dlaczego dotychczas wychodzi
gorzej niż oczekiwano; pobrać brakujące checkpointy z Edena; wyliczyć najlepszy operator dla tych wektorów.

## Dane

- Bieg 1810662 (`koopman_gesta_historia`): RF-DETR Nano od wag COCO, 8 epok po 100 aktualizacji, zapis co 5
  aktualizacji, 160 checkpointów wag (kroki 5..800). Lokalnie przed dzisiejszym dniem było 73 (5..365); pobranie
  reszty przerwała awaria SSH 24.09. Dziś wznowione (`pobierz_wynik.py`, sumy SHA256 z manifestu).
- Sondy: 72 obrazy (12 źródeł × 6 wariantów: original, rotation90, length08, black, yellow, foreground_bright),
  48 fit / 24 holdout po źródłach. Tokeny 576 × 384 z ostatniej warstwy backbone, maski udziału narzędzia.
- Tokeny dla kroków 370..800 liczone lokalnie tą samą funkcją `extract()` co dotychczasowe (`ekstrakcja.py`,
  zapis do `wyniki/tokeny_pelne`, katalog równoległej sesji nietknięty).

## Definicja „elementów identyfikujących narzędzie” (liczona tylko na obrazach fit i checkpointach ≤ S)

- `w_lda`: kierunek LDA tokeny narzędzia (maska ≥ 0,5) wobec tła (maska = 0) — jeden wektor w 384 wymiarach.
- `S_k`: k głównych kierunków różnicy „token narzędzia − średnie tło obrazu”, k ∈ {1, 4, 8, 16, 32}.
- Skalar tożsamości obrazu: a_i(t) = w_lda · (m_i(t) − tło_i(t)), m_i = średnia tokenów ważona maską.
- Kontrole: losowa podprzestrzeń tego samego wymiaru, dopełnienie, losowe kierunki poza S_32.

## Protokół (podział czasowy zamrożony)

- Dopasowanie do stanu S, test na przejściach S..K−2, rollout 20 checkpointów (100 aktualizacji) od S,
  bez prawdziwych stanów pośrednich. Dla K = 73: S = 48 (krok 245). Dla K = 160: S = 48 i S = 96 (krok 485).
- Modele per jednostka: alpha (v̂ = α v), ar2, hiperboliczne 1/t (Gur-Ari i in. 2018), pełne macierze na
  prędkości / położeniu w S_k. Modele wspólne między obrazami: macierz na prędkości z L = 1..4 opóźnieniami
  (Hankel), wariant z prawem 1/t. Kontrole: brak zmian, ostatni przyrost. Wybór τ i grzbietu wyłącznie na
  obrazach fit (końcówka okna). Miara: RMSE względem braku zmian na obrazach holdout.
- Bramka jak w poprzednich eksperymentach: ≥ 5 procent lepiej od ostatniego przyrostu, ≥ 20 procent od braku
  zmian, cos ≥ 0,5; nie zmieniana po wyniku.

## Skrypty

`licz.py` (kierunki, przewidywalność w podprzestrzeniach, wspólny Koopman), `najlepszy.py` (warianty operatora),
`ekstrakcja.py` (tokeny pełnej historii). Raporty: `wyniki/REPORT_K{K}.json`, `wyniki/REPORT_najlepszy_K{K}*.json`.

## Stan (2026-09-25, wieczór)

Obie części zakończone, wyniki i wnioski w `WYNIKI.md` (sekcje 1–5). Dodatkowe skrypty: `rollout_dlugi.py` (rollout
skalara tożsamości od S do końca historii), `pokaz_fastgan.py` (przewidziany latent jako cel dla FastGAN, sekcja 6,
wynik negatywny po stronie generatora). Poza gitem: `wyniki/tokeny_pelne` (9,5 GB), `*.npz`, obrazy; sumy w
`wyniki/SHA256.json`.

## Stan (2026-09-25, noc)

Sekcja 7 `WYNIKI.md` zakończona: wyszukiwarka DINOv2 równoległej sesji (tylko odczyt) dobrała klatki do obrazów FastGAN
przewidzianego latentu (top-6 → 17 klatek, pilot 1814538 przerwany; top-50 → 134 klatki, zadanie 1814611, 3 ziarna ×
3 ramiona), kontynuacja jednej epoki od `model_0000245.pth` przez `eden/kontynuacja.py` + `kontynuacja150.slurm`
(`trening.py` równoległej sesji bez zmian). Wynik: dobrane klatki = losowe = +0,001 mAP@0,5 względem braku dodania,
w rozrzucie ziaren. Podsumowanie `podsumuj_kontynuacje.py` → `wyniki/dobor_klatek/PODSUMOWANIE_KONTYNUACJA.json`,
`KONTYNUACJA.png`. Na Edenie zostały wagi końcowe (`~/smallvit_runs/2026-09-25_kontynuacja_dobor150/results/*/trening/FINAL.pth`,
9 × 120 MB) i katalog przerwanego pilotu `2026-09-25_kontynuacja_dobor`; do sprzątnięcia za zgodą.

Dopisek (2026-09-25, noc): efekt lokalny (`efekt_lokalny.py`, sekcja 7d) policzony lokalnie na RTX 3070 z wag `FINAL.pth`
pobranych do `wyniki/dobor_klatek/trening150/wagi/` (1,1 GB, poza gitem, sumy w `SHA256.json`): na klatkach val podobnych
do kotwic dobrane klatki nie pomagają (−0,003 AP@0,5, losowe +0,005, oba w szumie); okolica kotwic jest trudna (AP 0,55
na top-134, 0,38 wokół TR_0134). Predykcje wszystkich zestawów wag na całym val zapisane (`efekt_lokalny/PREDYKCJE_*.npz`).

