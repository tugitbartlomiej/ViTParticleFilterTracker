# Replikacja płaszczyzny krok × podkrok na seedzie 3047 (46 checkpointów, 10 klatek TEST) z predykcjami zapisanymi przed pomiarem

**Data:** 2026-10-23, 02:16 (Europe/Warsaw). **Typ:** pomiar na istniejących checkpointach, bez treningu.
**Rodzic:** `kroki_podkroki_afinicznosc_uwagi_20261023__00-41-29` (bieg s47, jedna klatka TRAIN 0). **Pochodzenie:** `Kod_eksperymentu/pochodzenie.json`.

## Cel

Sprawdzić na innym biegu (seed 3047, 0–400 epok) i na 10 klatkach spoza treningu (25 filmów TEST podziału kampanii), czy tezy z
`reports/Tezy artykułu o płaszczyźnie krok × podkrok.md` się powtarzają. Predykcje P1–P7 z progami są zapisane w
`Opis/PREDYKCJE_PRZED_POMIAREM.md` przed uruchomieniem skryptów i zakodowane w `Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py`.

## Hipotezy (skrót; pełne progi w PREDYKCJE)

P1 kolaps uwagi w blokach 4–5 poprzedza detekcję; P2 uwaga jest gwiazdą na narzędziu; P3 ruch po głębokości prosty w 1. epoce, zygzak później;
P4 złożenie 12 siodeł nie kurczy, nakładanie podprzestrzeni kurczących losowe; P5 front rozdziału w bloku 3–4, blok 6 nie podnosi;
P6 uwaga nie wygładza po siatce, MLP tak; P7 log|det| całego przejścia ujemny, dla narzędzia bardziej niż dla tła.

## Model, checkpointy, dane

- MałyViT seedu 3047 (kod `experiments/maly_vit_od_zera_35000_21epok_20261009/kod/model.py`), 46 checkpointów z
  `experiments/skae_replikacja_mapy_vit3047_20261022__17-14-27/Wyniki/Dane/Wejscia/wagi/` (kroki 0, 300, 1000, 3000, 6000, 10 000, 16 000,
  24 000, 33 000, 44 000, 55 000, 65 640 i co 10 epok do 437 500 = 400 ep; epoka = 1093,75 kroku). Źródła: trening kompensacja_funkcjonalna
  (0–100 ep), `vit_kontynuacja_do_200epok_20261020__19-06-42` (70–200 ep), `vit_kontynuacja_do_400epok_20261020__23-50-57` (210–400 ep).
  SHA256 czterech checkpointów w `pochodzenie.json`. Checkpointy do 437 500 chronione; nic nie jest trenowane ani usuwane.
- Klatki: po jednej z 10 filmów TEST (`split_manifest.json` kampanii; rozłączne z 117 filmami treningu), wybór deterministyczny
  (`wspolne.wybierz_klatki`): klatka z cystotomem o liczbie łat z udziałem obrysu ≥ 0,25 najbliższej 20 (minimum 8), ramka GT
  niezdegenerowana; 10 filmów o liczbie łat najbliższej 20. Lista z SHA256 JPG w `Wyniki/Dane/klatki.json`.
- Etykieta narzędzia: cystotom (kategoria, na której trenowano detektor), udział ≥ 0,25 (s47: 0,5 przy 3 łatach), tło = udział 0.

## Procedura

1. `Kod_eksperymentu/pomiar_3047.py Wyniki/Dane`: dla każdego z 46 checkpointów i 10 klatek jedno przejście (GPU) z przechwyceniem
   13 stanów, wag uwagi (6 × 3 × 600 × 600) i jasności laserów; obserwable jak w rodzicu, w 192D (udział translacji tu w 192D na surowych
   stanach), dodatkowo udział trzech największych odbiorców uwagi w obrysie ∪ halo 1 łaty, IoU i pewność.
2. `Kod_eksperymentu/jakobiany_3047.py Wyniki/Dane`: 10 checkpointów (0, 300, 1000, 3000, 10 000, 24 000, 65 640, 109 375, 218 750, 437 500)
   × 10 klatek × 6 tokenów (3 narzędzia, 3 tła), mapy własne float64; pełny jakobian uwagi dla 1 tokenu narzędzia i 1 tła.
3. `Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py Wyniki/Dane`: PASS/FAIL P1–P7 z liczbami.

## Metryki

Jak w rodzicu (docstringi skryptów). Zera: przetasowane Δ (CV-R²), losowe podprzestrzenie (nakładanie), jednostajna uwaga (0,013 na
8 sąsiadach), udział narzędzia (wzbogacenie = 1).

## Ograniczenia

- Siatka checkpointów 3047 jest rzadka poniżej 10 epok (0, 300, 1000, 3000, 6000, 10 000), więc P1 i P3 są testowane na tej siatce.
- 10 klatek z jednego podziału TEST; cystotom ~20 łat; jakobiany dla 6 tokenów na klatkę.
- Wykresy, raport i krótki wniosek: nie zamówione w tym zleceniu (użytkownik zamówił eksperyment z predykcjami; o materiały zapytać).

## Odtwarzanie

```
cd experiments/plaszczyzna_krok_podkrok_replikacja_3047_20261023__02-16-41
W:/Python/3.11/python.exe Kod_eksperymentu/pomiar_3047.py Wyniki/Dane
W:/Python/3.11/python.exe Kod_eksperymentu/jakobiany_3047.py Wyniki/Dane
W:/Python/3.11/python.exe Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py Wyniki/Dane
```

## Stan i werdykt (2026-10-23, 03:10; po pomiarze, progi niezmienione)

Pomiary ukończone: `Wyniki/Dane/plaszczyzna_3047.npz` (46 × 10), `jakobiany_3047.npz` (10 × 10 × 6 tokenów, błąd odtworzenia ≤ 1,4·10⁻¹⁵),
`klatki.json`, `tabele_odczyt.txt`, `tabele_jakobiany.txt`, `ocena_predykcji.json/.txt`. Ocena wg progów z `PREDYKCJE_PRZED_POMIAREM.md`:
**2 z 7 PASS (P5, P6), 5 FAIL (P1, P2, P3, P4, P7).**

- P1 FAIL: k* = 1000 (IoU 0,353); źródła bloku 4 = 73 (< 100 ✓), bloku 5 = 160 (✗); w k1000 73 / 160 (✗ < 50). Na tym seedzie blok 5
  kolapsuje ostatni (< 50 dopiero od k175000), a pierwsze są bloki 3 (< 50 od k1000), 1 i 4 (od k3000).
- P2 FAIL: udział 3 największych odbiorców w obrysie ∪ halo 0,886 (próg 0,90); wzbogacenie ≥ 10 w ≥ 8 klatkach nie na każdym checkpoincie
  (minimum 1 klatka). Mediana wzbogacenia 11,0 / 13,5 (bloki 4 / 5); przy 20 łatach narzędzia (3,3 % chmury) maksymalne możliwe wzbogacenie
  to ok. 30, więc próg 10 był źle skalowany względem s47 (3 łaty, maksimum 200). To uwaga po fakcie, nie zmiana werdyktu.
- P3 FAIL: k0 1,014 ✓, maksimum 2,78 (k1000) ✓, ale 1,203 w k24000 (próg ≤ 1,2; od k33000 wszystkie ≤ 1,16).
- P4 FAIL: udział σ < 1 iloczynu 0,479–0,557 (próg 0,45–0,55; maksimum przekroczone), nakładanie − losowe: średnia 0,014, maksimum 0,036
  (próg 0,03; przekroczone w k1000–k3000, gdy MLP pchają spójnie).
- P5 PASS: największy skok Fishera w m4 w 352 z 390 par (klatka × checkpoint ≥ k24000), w m3 34; Fisher po MLP 6 ≤ 1,05 · po uwadze 6
  we wszystkich checkpointach (średnio m4 3,48, m5 3,61, m6 2,86).
- P6 PASS: masa uwagi na 8 sąsiadach ≤ 0,018 (< 0,026); zmiany Dirichleta w MLP −0,08…−0,20 wobec uwagi −0,02…−0,03.
- P7 FAIL: Σ log|det| dla narzędzia +2,7 w k10000 (ujemny dopiero od k24000: −3,2; potem −14,5 (60 ep), −17,0, −25,4, −26,9 (400 ep));
  narzędzie < tło w 8–10 z 10 klatek na każdym checkpoincie ≥ k24000 ✓.

Materiały opcjonalne (wykresy, raport, wniosek) nieprzygotowane: do decyzji użytkownika.
