# Tożsamość narzędzia: kierunek stały i odświeżany na 10 ziarnach (bez treningu, z zapisanych latentów)

Plan: [PLAN.md](PLAN.md). Wynik: [WYNIKI.md](WYNIKI.md). Deklaracja: [WYNIKI.json](WYNIKI.json).

Przepis na kierunek tożsamości (`_ledoit_wolf`, `_cov`, `lda`, `wektor`) skopiowany z
`../maly_vit_petla_sd_rownanie/kod/sila.py`. Jedna zmiana: `numpy.linalg.solve` zamiast `lstsq` (ten sam wynik,
szybciej). Kontrola: siła wzdłuż kierunku stałego odtwarza liczby z `../maly_vit_wektor_tozsamosci/PLAN.md`
(6,13 w kroku 1000; 5,30 / 4,36 w kroku 4500).

Dane tylko do odczytu: `../maly_vit_wektor_tozsamosci/wyniki_eden/petla_s47..s56` (230 plików `LATENT.npz`, 5,7 GB).

## Uruchomienie (lokalnie, Python 3.11, bez GPU)

```bash
cd kod
py -3.11 odswiezana.py --wyniki ../wyniki                  # pomiar (około 20 minut) i analiza
py -3.11 odswiezana.py --wyniki ../wyniki --tylko-analiza  # sama analiza z POMIAR.npz, sekundy
```

## Pliki wyników

- `wyniki/POMIAR.npz`: miary dla (ziarno, ramię, chwila co 50 kroków), siła każdej sondy, mAP co 100 kroków,
  macierze Grama układu sond i kierunki w krokach 1000, 2500, 4500.
- `wyniki/WYNIK.json`: kryteria K1-K5 z planu.
- `wyniki/UZUPELNIENIE.json`: opis dodany po wyniku (wczesne kroki, starzenie się kierunku wokół chwili
  dopasowania, prognoza z krzywej innych ziaren bez startu w kroku 1000, zmienność siły i mAP).
