# mAP surowych i uśrednionych wag małego ViT (bez treningu, sama ocena zapisanych wag)

Plan: [PLAN.md](PLAN.md). Wynik: [WYNIKI.md](WYNIKI.md). Deklaracja: [WYNIKI.json](WYNIKI.json).

`kod/dane.py`, `kod/model.py`, `kod/trening.py` skopiowane bez zmian z `../maly_vit_ruch_czastek_gesta_historia/kod`
(sumy SHA256 zgodne). Nowe: `kod/srednia_wag.py` (pomiar z planu) i `kod/pozna_faza_s48.py` (uzupełnienie po wyniku).
Wagi tylko do odczytu: `../maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/wagi` (ziarno 47, co 10 kroków) i
`../maly_vit_rownanie_latentu/nagranie_s48_sily/wagi` (ziarno 48, co 100 kroków). Niczego się nie uczy.

## Uruchomienie (lokalnie, Python 3.11, karta z CUDA)

Ścieżki wejść ustawia sam skrypt (jak w `../maly_vit_petla_sd_rownanie/uruchom_lokalnie.cmd`): klatki walidacji i
bazy z `../maly_vit_od_zera/cache`, adnotacje z `../maly_vit_od_zera/eden/wejscia`.

```bash
cd kod
py -3.11 srednia_wag.py --wyniki ../wyniki/proba --proba   # próba kodu na 300 klatkach, to nie jest wynik
py -3.11 srednia_wag.py --wyniki ../wyniki                 # około 5 minut na RTX 3070, 3 GB RAM
py -3.11 pozna_faza_s48.py --wyniki ../wyniki              # około 4 minut
```

## Pliki wyników

- `wyniki/SREDNIA_WAG.json`: mAP 51 surowych zapisów (kroki 500-1000), wagi uśrednione w 5 chwilach dla okien 50,
  100, 200 kroków i dwóch wariantów BatchNorm, średnia wykładnicza, kryteria K0-K2.
- `wyniki/POZNA_FAZA_S48.json`: to samo dla ziarna 48 w krokach 1000-3000 (zapisy co 100 kroków).
