# Małe ViT o różnej wielkości latentu, zbiór 5000 klatek: plan (czeka na zgodę na uruchomienie)

Stan: przygotowane, kod sprawdzony lokalnie (próba 20 kroków: femto, tiny_tl8, small), nic nie uruchomiono
na Edenie. Data: 2026-10-06.

## Po co

Podgląd, jak wygląda latent detektora końcówki narzędzia przy różnej jego wielkości: od 600 x 128 do 600 x 1536
oraz w wersji z 8 tokenami (TokenLearner) zamiast 600. Wszystkie modele w tym samym przepisie i na tym samym
zbiorze, żeby różniła się tylko wielkość sieci. Modele trafią potem do aplikacji (widoki cząsteczek, PCA,
przestrzeń makrowektorów).

## Znane ViT i wielkość latentu (z pamięci, do sprawdzenia w literaturze przed cytowaniem)

Latent w tym projekcie = warstwa ukryta MLP ostatniego bloku = tokeny x (mlp x szerokość).

| Model | szerokość tokenu | MLP (latent na token) | tokeny przy 224 px | uwagi |
|---|---:|---:|---:|---|
| ViT-B/16 | 768 | 3072 | 196 | Dosovitskiy i in. 2021 |
| ViT-S/16 (DeiT-S) | 384 | 1536 | 196 | |
| ViT-Ti/16 (DeiT-Ti) | 192 | 768 | 196 | szerokość jak nasz MalyViT |
| MobileViT-XXS | 64 / 80 / 96 (etapy) | 2 x szerokość | zmienna | hybryda konwolucja + transformer |
| TinyViT-5M | 64 / 128 / 160 / 320 | 4 x szerokość | zmienna | hierarchiczny |
| LeViT-128S | 128 / 256 / 384 | 2 x szerokość | zmienna | hierarchiczny |
| TokenLearner | jak sieć | jak sieć | 8 albo 16 | Ryoo i in. 2021 (przeczytany, Undermind Ryo21) |
| TiTok | | | 32 (1D) | tokenizer obrazu do generowania |

## Ramiona

Wszystkie: łodyga konwolucyjna z krokiem 16, 6 bloków, MLP 4 x szerokość, wejście 480 x 320, siatka 20 x 30,
głowa gęsta (środek narzędzia + ramka na patch), jak MalyViT aplikacji.

| Ramię | szerokość | głowy | tokeny w drugiej połowie | latent | parametry |
|---|---:|---:|---:|---:|---:|
| femto | 32 | 2 | 600 | 600 x 128 | 0,16 M |
| pico | 64 | 2 | 600 | 600 x 256 | 0,41 M |
| nano | 96 | 3 | 600 | 600 x 384 | 0,82 M |
| tiny | 192 | 3 | 600 | 600 x 768 | 2,93 M |
| small | 384 | 6 | 600 | 600 x 1536 | 11,18 M |
| tiny_tl8 | 192 | 3 | 8 (TokenLearner po bloku 3) | 8 x 768 | 2,97 M |
| nano_tl8 | 96 | 3 | 8 (TokenLearner po bloku 3) | 8 x 384 | 0,83 M |

`tiny` ma dokładnie budowę MODEL_VIT_K1000 aplikacji (te same klucze wag, wczytanie strict), więc jest
odniesieniem. TokenLearner: wersja 1.1 pracy (MLP z GELU, sigmoid, średnia ważona), TokenFuser oddaje 8 tokenów
na siatkę 600 patchy przed głową. Ramiona `_tl8` zapisują też 8 map uwagi (który patch zbiera każdy token).

## Dane: 5000 klatek

Baza 1500 klatek (jak dotąd) + 3500 z puli treningowej (split train, z ramką narzędzia, bez 1501 wykluczonych:
baza i kotwice), równo po 118 filmach treningowych (19 do 30 klatek na film, w filmie równo w czasie;
`dane.zbior_n`, ziarno 20261006). Sprawdzone lokalnie: 3500 unikalnych, 0 wspólnych z bazą. Walidacja bez zmian:
3000 klatek z filmów walidacyjnych.

## Przepis (wspólny, jak trening.Uczen dotąd)

Od zera, ziarno 47, paczka 32, AdamW lr 5e-4, wd 0,05, rozgrzewka 300 kroków, potem stały lr, bf16,
odbicie poziome i jasność. Długość 5000 kroków (32 epoki przy 5000 klatkach; K1000 miał 1000 kroków na 1500).
Ocena mAP@0,5 na 3000 klatkach walidacji co 500 kroków. Zapis co 500 kroków: wagi i latent 12 klatek pomiaru
(te same co w gęstej historii), przy `_tl8` także mapy uwagi; na końcu wagi ze stanem AdamW i MODEL_<ramię>.pt
z konfiguracją (do aplikacji).

## Eden

- Katalog: `~/smallvit_runs/2026-10-06_maly_vit_rozne_latenty_5000` (kod, logi, wyniki/<tryb>_<ramię>).
- Wejścia z poprzednich biegów (sprawdzone 2026-10-06): `2026-09-28_maly_vit_od_zera/wejscia` (pretrain COCO,
  index, wykluczone), `deps/timm`, `2026-10-01_..._gesta_historia/wejscia/obrazy.npy`, archiwum
  `tracking_static36.tar` (2 GB, rozpakowane do /dev/shm węzła i usuwane pułapką EXIT).
- `bieg_ramiona.slurm`: zadanie tablicowe 0..6, `%1` = jedno ramię naraz (zasada: jeden bieg naraz).
  Najpierw `sbatch bieg_ramiona.slurm proba` (20 kroków każde ramię), potem `sbatch bieg_ramiona.slurm pelny`.
- Koszt (szacunek): 5 do 20 minut na ramię na H100 (small najdłużej), razem około 1 do 1,5 godziny.
  Dysk: około 2 GB i około 110 plików (kwota 2026-10-06: 165,8 / 200 GB, 110 tys. / 200 tys. plików).

## Czego to nie pokaże

- Jedno ziarno: różnice mAP między ramionami w granicach kilku punktów mogą być szumem.
- 5000 kroków to nie zbieżność dla każdego ramienia; większe sieci mogą potrzebować dłużej.
- Wielkość latentu zmienia się razem z pojemnością sieci; to porównanie modeli, nie samego latentu.
- TokenLearner nie był sprawdzany w pracy na detekcji; wynik `_tl8` to pierwszy pomiar tego zastosowania.
