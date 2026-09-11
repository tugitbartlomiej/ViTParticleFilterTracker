# Session: Selekcja danych wg literatury + ważność próbki (RF-DETR / RT-DETR)

**Created:** 2026-09-11 16:07:38
**Type:** Analysis
**Status:** Completed

## Quick Summary
Zweryfikowałem kod selekcji danych (`SmallViTAblation/pipeline/`) i metodologię artykułu
wobec literatury o koresetach i przycinaniu zbiorów dla detekcji — dwadzieścia prac
przeczytanych w całości przez Undermind. Powstały trzy dokumenty projektu (19, 20, 21):
sześć rozbieżności kodu z literaturą, odpowiedź na pytanie o LightGBM jako „obserwatora"
ważności danych oraz katalog sześciu rodzin metod ważności próbki dla obrazów.
Dodatkowo uporządkowałem bibliografię zbioru RT-DETR: linki, BibTeX sklejony i per praca,
65 nowych PDF-ów pobranych automatycznie.

## Key Results
- **Sześć rozbieżności kodu z literaturą** (`docs/19`): EL2N z proxy uczonego na 2500 z 71k
  i jednej inicjalizacji, klatki tła z maksymalną trudnością, k = budżet + centroid, selekcja
  z puli augmentowanej, cechy CLS całego kadru 224 px, nazwa EL2N bez uśredniania.
- **Kalibracja oczekiwań**: realny zysk selekcji nad losowaniem to jednostki punktów mAP
  (AdaDeDup +0,0065 mAP przy 70% danych; VPS +1,1 do +2,0 pp; CSOD +6,4 pp przy 200 obrazach).
- **Blokada arytmetyczna**: `budgets.tier_b: 20000` przy 17 832 kadrach źródłowych treningowych.
- **RF-DETR ≠ RT-DETR** — rozstrzygnięcie nazewnicze i konsekwencje dla argumentacji.
- **LightGBM**: mierzy ważność cech, nie próbek; pełnego łańcucha „drzewo → wartość obrazu →
  podzbiór treningowy detektora" nikt nie opublikował (171 prac sprawdzonych).
- **DetGain** [Sun25] — jedyna metoda liczaca wartość obrazu jako przyrost mAP; +2 mAP nad
  losowaniem i poprawne traktowanie klatek bez obiektu.
- **Bibliografia**: 29 wpisów BibTeX (sklejony + per praca), 112 PDF-ów w `pdf_all`.

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
