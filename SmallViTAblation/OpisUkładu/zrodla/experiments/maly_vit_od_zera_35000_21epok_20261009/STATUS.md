# Status biegu 35 000 · 21 epok

## 2026-10-09

- Na Edenie 8 października o 22:19 wyczyszczono pliki części starych biegów, zostawiając katalogi. Brakowało
  `2026-09-28_maly_vit_od_zera/deps/timm` i adnotacji walidacji. Bieg ma więc własne kopie w `deps/timm`
  (timm 0.9.16 z `2026-10-08_k1000_podmiana150_sd/deps`) i `wejscia/pretrain/valid/_annotations.coco.json`
  (SHA-256 `7921f089…` zgodne z lokalną kopią `experiments/maly_vit_od_zera/eden/wejscia`).
- Zadania 1826051 i 1826053 padły w pierwszej sekundzie na sprawdzeniu plików (brak timm), a 1826052 i 1826054
  anulowano. Nic nie policzyły ani nie zapisały.
- Próba 1826055 (512 klatek, 2 epoki, H100): ukończona w 82 s, 9 plików wag, mAP@0,5 0 → 0,079, `/dev/shm`
  posprzątane. Wagi próby wczytują się w aplikacji: `ruch_naturalny.discover` i `load_state_dict(strict=True)`.
- Bieg właściwy 1826056 ruszył 2026-10-09 o 17:50 na węźle hopper (H100). Kwota przed startem: 98,9 GB
  zajęte z 200 GB.

- Bieg 1826056 ukończony 2026-10-09 o 18:03, po 13 min 25 s (uczenie 728 s, około 34 s na epokę, 32 ms na
  krok). 21 epok, 22 974 kroki, 341 plików wag (3,8 GB). Walidacja: 3000 klatek, żadna nie należy do treningu.
  mAP@0,5 na końcach epok: 0,449 po 1. epoce, 0,622 średnio w epokach 8-14, 0,623 w epokach 15-21, 0,652 po 21.
  Strata uczenia dalej spadała (0,47 w krokach 7658-15316, 0,42 w krokach 15316-22974), a walidacja stała:
  plateau przy stałym lr 5e-4, bez wygaszania.
- Pobrano lokalnie krok 0 i 21 końców epok (22 pliki, SHA-256 zgodne) oraz `dane/`.
- Na polecenie użytkownika dalszy trening do 60 epok tym samym przepisem: zadanie 1826076, `wznow 60` od
  `stan_e21.pt`. Pierwsze wznowienie 1826075 padło na `set_rng_state` (stan generatora wczytany na kartę), bez
  zmian w wynikach; poprawione. Po wznowieniu: epoka 22 mAP 0,630, epoka 23 mAP 0,637.

- Zadanie 1826076 ukończone 2026-10-09 o 18:36 (24 min). Łącznie 60 epok, 65 640 kroków, 806 plików wag
  (8,8 GB na Edenie). Kwota projektu po biegu: 107,9 GB z 200 GB.
- Walidacja po 21. epoce nie rośnie: mAP@0,5 0,60-0,65 w epokach 22-45 i 0,59-0,65 w epokach 46-60, a epoka 60
  ma 0,608. Najlepsze końce epok: 0,6533 (epoka 26), 0,6524 (epoka 21). Przy stałym lr 5e-4 dalsze epoki tylko
  dopasowują model do zbioru uczącego. Lepszą walidację dałoby raczej wygaszanie lr (np. cosinus od epoki 21)
  niż kolejne epoki.

## Pomiar naturalny 100 losowych zdjęć z puli (2026-10-09)

- `outputs/vit_35000_losowe100_20261009`: 100 klatek wylosowanych z ziarnem 2026 z puli 35 000 (68 filmów),
  oryginalne bajty z zipów, `MANIFEST_100.jsonl`. To klatki uczące tego modelu.
- Pomiar: końce epok 0-21 (22 checkpointy, krok co 1094), q16, 60 000 cząsteczek, 13,3 s.
  Wynik: `outputs/ruch_czasteczek_gui/20261009T162654_natural_78cd84bd6edd`.
- Średnio na 100 zdjęciach: pewność 0,014 i IoU 0,003 w kroku 0, 0,676 i 0,634 po 1. epoce, 0,834 i 0,824 po
  10., 0,869 i 0,840 po 21. Obie miary ≥ 90 procent: 17 zdjęć po 21. epoce.
- Zrzuty (PCA i siatka zdjęcia, kolor narzędzie/tło, płynne przejścia, EMA 0,4, ślad 4 zapisy):
  `outputs/vit_35000_losowe100_20261009/zrzuty`.

## Gęsty zapis ruchu sond i sił (2026-10-09, `kod/obserwator_gesty.py`)

- Fizyka ruchu na historii K1000 (180 000 cząsteczek, PCA3, kroki 60–1000): wykładnik MSD α = 0,32 (silna
  subdyfuzja, uwięzienie), VACF przy opóźnieniu 10 kroków −0,19 (zawracanie), pole kolektywne 35 % energii,
  czas powrotu OU około 50 kroków. GMM z BIC nie znajduje osobnych grup (ARI z narzędziem 0,01, z obrazem 0,00).
  Temperatura i norma 768D rozdzielają narzędzie od tła (AUC 0,82 i 0,86); ruch w rozgrzewce w 84 % zależy od
  numeru patcha. Rysunek i cechy: `outputs/vit_35000_losowe100_20261009/fizyka_ruchu_k1000/`.
- Brakowało: wymiarów (PCA3 ze stałej bazy kroku 0), gęstości (co 10 kroków) i sił (skład paczki, strata,
  aktualizacja wag). Nowy bieg `gesty` (zadanie 1826161, po próbie 1826160 COMPLETED): ten sam przepis, 21 epok,
  100 sond (SONDY_100.jsonl) z zapisem PCA32 i norm 768D co 1 krok do kroku 2188 i co 10 dalej, pełne 768D dla
  8 sond co 1 do kroku 1094 i co 50 dalej, oraz co krok: indeksy paczki, strata przed/po (eval), norma aktualizacji
  wag, norma gradientu po obcięciu, lr. Szacunek miejsca: około 29 GB; obserwator przerywa zapis poniżej 6 GB
  wolnej kwoty. Wyniki: `wyniki/gesty/gesty/` na Edenie.

## Po zakończeniu

```bash
bash pobierz_z_edena.sh epoki      # krok 0 i końce wszystkich 60 epok
```

Potem w aplikacji: „Ruch zbioru”, strona „Uczenie naturalne”, przycisk „Historia 35 000 · epoki”, „Oblicz ruch
naturalny”.
