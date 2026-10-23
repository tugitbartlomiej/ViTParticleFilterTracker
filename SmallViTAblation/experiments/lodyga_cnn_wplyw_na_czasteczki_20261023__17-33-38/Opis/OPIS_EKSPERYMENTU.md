# Łodyga CNN: czy uczy się razem z blokami i jak wpływa na ruch cząsteczek w ViT

**Data:** 2026-10-23, 17:33 (Europe/Warsaw). **Typ:** pomiar na istniejących checkpointach, bez treningu.
**Rodzice:** `plaszczyzna_krok_podkrok_replikacja_3047_20261023__02-16-41` (klatki TEST, moduł `wspolne.py`, checkpointy 3047),
`kroki_podkroki_afinicznosc_uwagi_20261023__00-41-29` (s47, klatka TRAIN 0).

## Cel
Pytanie użytkownika: przed blokami ViT działa CNN (łodyga: 4 konwolucje 3×3 z krokiem 2, 3→24→48→96→192, pierwsze trzy z BatchNorm
i GELU). Czy łodyga uczy się w treningu i czy ma wpływ na to, jak cząsteczki „pływają” w blokach?

## Hipotezy
H1. Łodyga uczy się: optymalizator AdamW obejmuje `named_parameters()` bez wyjątków (`trening.py` wiersze 27–29), BatchNorm w trybie
train, więc wagi konwolucji, γ/β i średnie kroczące BN zmieniają się między checkpointami.
H2. Łodyga wyznacza punkt startu cząsteczek (stan 0), więc jej zamiana między checkpointami zmienia wejście (Dirichlet, Fisher, norma),
ale dynamika w blokach (źródła uwagi, rozrzut MLP 1, front w bloku 4) jest własnością bloków: łodyga z kroku 0 z blokami końcowymi
daje gorsze IoU, lecz zachowuje gwiazdę uwagi i rozpychanie MLP 1; bloki z kroku 0 z łodygą końcową nie dają ani jednego, ani drugiego.

## Model i dane
Seed 3047: checkpointy 0, 1000, 10 000, 65 640, 218 750, 437 500 (`skae_replikacja_mapy_vit3047/Wyniki/Dane/Wejscia/wagi`), 10 klatek TEST
z replikacji (`klatki.json`). Seed s47: kroki 0 i 65 640, klatka TRAIN 0. Kod modelu `maly_vit_od_zera_35000_21epok_20261009/kod/model.py`.

## Procedura
`Kod_eksperymentu/lodyga.py Wyniki/Dane`: (A) względna zmiana wag i cos między checkpointami osobno dla konwolucji łodygi, γ/β BN,
statystyk kroczących BN, bloków i głowy; (B) dryf stanu 0 na 10 klatkach (cos do kroku 0, norma, Dirichlet, Fisher); (C) złożenie
łodygi z checkpointu A z blokami i głową z checkpointu B: IoU, pewność, Fisher po MLP 4 i 6, źródła uwagi bloku 4, rozrzut MLP 1,
wskaźnik balistyczny, względna odległość stanów końcowych od układu pełnego B; (D) to samo dla s47 na klatce TRAIN 0.

## Ograniczenia
Zamiana łodygi między checkpointami łączy wagi, które nigdy razem nie trenowały; wynik mówi o wrażliwości bloków na wejście, nie o
przyczynowości w treningu. 10 klatek, dwa seedy. Bez wykresów, raportu i wniosku (nie zamówione).

## Stan (2026-10-23, 17:45): pomiar ukończony, `Wyniki/Dane/lodyga.json`, `log.txt`

- H1 potwierdzona: łodyga uczy się mocniej niż bloki. Względna zmiana wag konwolucji 0→60 ep: 3,0 (cos 0,15; norma ×3,0) na obu seedach,
  wobec bloków 1,46 (cos 0,37) i głowy 1,38; do 400 ep (3047): łodyga 3,12 (cos 0,05), bloki 1,94 (cos 0,21). γ/β BN: 0,53–0,74; statystyki
  kroczące BN: 1,6–1,8 (wariancja ×2,4–2,6). Między 200 a 400 ep łodyga zmienia się jeszcze o 0,75 (cos 0,72), bloki o 0,87.
- Wejście bloków (stan 0) dryfuje: cos do kroku 0 ≈ 0,70–0,75 od 1. epoki; Dirichlet 0,06 → 0,33 (łodyga wyostrza cechy); Fisher na wejściu
  na 3047 zostaje ≈ 0,45–0,56 (łodyga sama nie rozdziela narzędzia od tła liniowo; na s47 0,62 → 0,84).
- H2 częściowo: zamiana łodygi (3047, 10 klatek): łodyga z k0 + bloki z 400 ep → IoU 0,00, Fisher po MLP 4 0,43 (brak frontu), źródła uwagi
  bloku 4: 41 (gwiazda osłabiona, nie zniknęła), rozrzut MLP 1 ×2,7 (bloki rozpychają obce wejście jeszcze mocniej), odległość stanów końcowych
  1,57 promienia chmury. Łodyga z 400 ep + bloki z k0 → IoU 0,15, brak gwiazdy (571 źródeł), brak rozpychania (×1,03), Fisher ≈ 0,45.
  Łodyga z 200 ep + bloki z 400 ep → IoU 0,535 (pełny 0,540), Fisher 3,34, odległość 0,32; łodyga z 60 ep → IoU 0,476, Fisher 2,94.
  Wniosek: gwiazda, rozpychanie i front w bloku 4 są własnościami bloków, ale uruchamiają się tylko na cechach wyuczonej łodygi; po ~60 ep
  dalsze zmiany łodygi mają już mały wpływ na dynamikę w blokach.

