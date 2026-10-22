# Session Summary: Grupa narzędzia w latencie, filmy niewidziane i 24 seedy

## Metadata
- **Date:** 2026-10-21 – 2026-10-22
- **Time:** do 20:09
- **Status:** Completed
- **Type:** Analysis

## Objective
- Ustalić, co z mikroskopu latentnego da się opublikować i co jeszcze zbadać.
- Wyjaśnić, czemu LDA w R9 „ładnie oddzielała” narzędzie na klatkach B i C.
- Wczytywać etykietę narzędzia z obrysu (segmentacji), nie z ramki; sprawdzać na ostatnich checkpointach; poprawić R9.
- Metodologia bez etykiet: wykryć grupę tokenów narzędzia w latencie, a etykietę użyć tylko do sprawdzenia.
- Wizualizacja LDA (oś z etykiety) na początku i końcu uczenia.
- Sprawdzić wyniki na filmach TEST (niewidzianych w treningu) i na 24 seedach.

## Context
R9 (rzuty 192D na GPU) pokazywał LDA z etykietą z ramki GT. Cienki cystotom zajmuje ok. 3 z 18 łat ramki, więc etykieta z ramki mieszała narzędzie z tłem, a zero z losowych tokenów było zbyt łagodne. Klatki MANIFEST_1500 pochodzą z filmów treningu.

## Actions Taken
1. `app/etykiety_narzedzia.py`: udział pikseli wielokąta `<cat>_seg` w każdej łacie 16×16, indeks ramek wszystkich filmów, przesunięcia maski (zero przestrzenne).
2. `lasery_dane.Historia`: pola `narz` (obrys), `mieszane`, `udzial`, `narz_ramka`, `etykieta`; `rzuty_dane` WERSJA 2 (tło bez łat mieszanych).
3. `rzuty_dane`: `zero_przestrzenne_gpu`, `k_sasiadow`, `wykryj_grupe` (TokenCut, τ = 0,2, spójna), `zgodnosc_z_obrysem`.
4. R9 (`rzuty_okienka.py`): etykieta obrys / ramka GT / grupa wykryta bez etykiet; „Oś LDA” z obrysu albo z grupy; próg udziału; zero przestrzenne na pauzie; Fisher na pierwszym miejscu.
5. Eksperyment `separacja_narzedzia_replikacja_klatek_20261021__19-11-15`: separacja na wielu klatkach z obrysem i zerem przestrzennym, klatki A/B/C.
6. Eksperyment `wykrywanie_grupy_narzedzia_bez_etykiet_20261021__21-42-46`: wybór metody na filmach DEV (448 konfiguracji: b3–b6, x6, z6, n6; kurtoza, odstawanie, TokenCut), ocena na TEST, LDA w przebiegu uczenia, rysunki LDA początek–koniec, `niewidziane_filmy.py` (tryby `glowny`, `seedy`).
7. Przegląd literatury (nowość): `SmallViTAblation/research_notes/Publikowalność wyników mikroskopu latentnego/nowosc_i_wyprzedzenie.md`.

## Results

### Key Findings
- Wcześniejsze twierdzenia poprawione po weryfikacji: „klatka B bez rozdziału” i „segregacja rośnie po plateau IoU” nie trzymają się przy etykiecie z obrysu i zerze przestrzennym.
- 25 filmów TEST (500 klatek):

| Checkpoint | Dice grupy | 192D > zero | LDA > zero | IoU |
|---|---|---|---|---|
| s47 k0 | 0,04 | 3 % | 2 % | — |
| s47 k100 | 0,03 | 57 % | 47 % | — |
| s47 k1000 | 0,11 | 76 % | 36 % | — |
| s47 k10100 | 0,38 | 85 % | 74 % | — |
| s47 k65640 | 0,40 | 91 % | 90 % | — |
| s3047 ep400 | 0,40 [0,34–0,45] | 90 % | 90 % | 0,65 |

- 24 seedy: Dice 0,14 (k1000) → 0,31 (k10000) → 0,38 (k65640, 0,33–0,43); 192D 73 % → 85 % → 88 %.
- Nowość: odkrywanie obiektów w tokenach ViT jest znane (DINO, TokenCut) dla modeli samonadzorowanych; śledzenie tego w uczeniu detektora uczonego ramkami, z zerem przestrzennym i protokołem po filmach, nie było opisane.

### Issues Encountered
- Filmy TR_0058 i TR_0118: więcej wierszy adnotacji niż obrazów, pominięte (VAL).
- Wspólne GPU spowalniało przebiegi: tylko filmy TEST, zero przestrzenne dla seedów tylko na k1000/k10000/k65640.
- Hook pre-commit (> 5 MB): `niewidziane_*.jsonl` w `.gitignore`.
- Okno 12 „dolina” wczytuje wagi ze starej ścieżki F: (nienaprawione).

## Files Generated/Modified
- `SmallViTAblation/app/etykiety_narzedzia.py` (nowy), `lasery_dane.py`, `rzuty_dane.py`, `rzuty_okienka.py`, `README.md`
- `SmallViTAblation/experiments/separacja_narzedzia_replikacja_klatek_20261021__19-11-15/`
- `SmallViTAblation/experiments/wykrywanie_grupy_narzedzia_bez_etykiet_20261021__21-42-46/`
- `SmallViTAblation/research_notes/Publikowalność wyników mikroskopu latentnego/nowosc_i_wyprzedzenie.md`
- Commity: 640a96a6, 3822effd, fe63e166, a896894c, a34a6466, 3f1f4260

## Commands Used
```bash
W:/Python/3.11/python.exe Kod_eksperymentu/wykryj.py
W:/Python/3.11/python.exe Kod_eksperymentu/niewidziane_filmy.py glowny
W:/Python/3.11/python.exe Kod_eksperymentu/niewidziane_filmy.py seedy
W:/Python/3.11/python.exe Wyniki/Code_Interpretujacy_Wyniki/podsumuj_niewidziane.py
```

## Next Steps
- [ ] Więcej przebiegów do ep. 400 (teraz tylko seed 3047)
- [ ] Drugi typ narzędzia (teraz tylko cystotom)
- [ ] Pełne teksty: Isobe 2026, Martinelli 2025, TGO-II
- [ ] Ścieżka F: w oknie 12 „dolina”
