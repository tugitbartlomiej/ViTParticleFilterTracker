# Sufit przewidywalności latentu małego ViT na 100 kroków

Pomiar bez treningu na zapisanej gęstej historii tokenów (bieg Eden 1818732, ziarno 47, kroki 0..1000 co 1).
Plan, aparat i kryteria: [PLAN.md](PLAN.md). Wynik: [WYNIKI.md](WYNIKI.md). Deklaracja: [WYNIKI.json](WYNIKI.json).
W tym repozytorium deklaracja rodziny jest lokalna; narzędzi rejestru projektu grantowego nie uruchamiano.

## Uruchomienie (Python 3.11, około 25 minut, około 6 GB RAM)

```bash
cd kod
py -3.11 sufit.py --wyniki ../wyniki/pelny        # 12 obrazów -> wyniki/pelny/METRYKI.json
py -3.11 tabela.py ../wyniki/pelny/METRYKI.json 100   # tabela dla horyzontu 100
```

`dane.py` czyta `LATENT_GESTY.npy` przez mmap, po jednym obrazie. Pliku źródłowego nie zmienia.

## Zawartość

- `kod/dane.py`: wczytanie danych i opis pułapek (BatchNorm, kolejność tokenów, maski).
- `kod/sufit.py`: sufit wyroczni, podłoga z samych drgań, prognozy przyczynowe bez dopasowania i prognoza
  liniowa z własnej historii dopasowana krzyżowo między połowami obrazów.
- `kod/tabela.py`: tabela z `METRYKI.json`.
- `kod/prognoza_900.py`, `kod/wykres_900.py`: równanie ze ściąganiem do średniej od kroku 900 do 1000, krok po
  kroku, z porównaniem do NNDMD. Opis: [WYNIKI_PROGNOZA_900.md](WYNIKI_PROGNOZA_900.md).
  `py -3.11 prognoza_900.py --wyniki ../wyniki/prognoza_900` (około 13 minut), potem `py -3.11 wykres_900.py`.
- `wyniki/proba/`: próba na obrazach 0 i 1 (sprawdzenie kodu, nie wynik).
- `wyniki/pelny/`: pełny pomiar, `METRYKI.json` i logi.
