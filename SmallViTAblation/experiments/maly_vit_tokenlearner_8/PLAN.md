# Mały ViT z 8 tokenami (TokenLearner): plan, do zatwierdzenia

Stan: PLAN, nic nie uruchomiono. Data: 2026-10-06.

## Pytanie

Czy mały ViT do detekcji końcówki narzędzia (Cataract-LMM) da się przebudować tak, żeby w drugiej połowie sieci
obraz był opisany 8 tokenami zamiast 600? I czy te 8 tokenów samo uczy się „pilnować” narzędzia, źrenicy, tła?
Cel praktyczny: 8 cząstek po 768 mikrowektorów do widoków aplikacji zamiast 600, każda z mapą „skąd ciągnie”.

## Źródło metody

Ryoo i in. 2021, „TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?” (arXiv 2106.11297),
przeczytany w całości przez Undermind (workspace ViT, klucz Ryo21). Najważniejsze ustalenia:

- Moduł: S map uwagi liczonych z mapy cech (obrazy: cztery konwolucje 3x3 z GELU; wersja 1.1: MLP, czyli dwie
  warstwy liniowe z GELU), sigmoid (nie softmax), token i = średnia przestrzenna z cech razy mapa i.
- Miejsce: domyślnie w połowie sieci (po bloku 6 z 12). Połowa daje „prawie tę samą” dokładność przy niemal
  połowie obliczeń; po 3/4 sieci bywa lepiej niż bez modułu.
- S: próbowano 8, 16, 32; domyślne 8 i 16. B/16 po bloku 6: 8 tokenów 83,2 %, 16 tokenów 83,2 % (bez TokenFusera),
  z TokenFuserem 83,7 i 83,9 %. ViT-B/16 bez modułu: 84,73 % przy 55,6 GFLOPs, z 8 tokenami 83,65 % przy 28,7.
- TokenFuser: miesza S tokenów i oddaje je na siatkę: X_wyj = sigmoid(Lin(X_wej)) · Y + X_wej.
- Ograniczenie ważne dla nas: praca bada tylko klasyfikację (ImageNet, wideo). Detekcji, segmentacji ani wyjść
  na każdy patch nie sprawdzano. Nie wiadomo, czy 8 tokenów utrzyma dokładne położenie końcówki.

## Architektura (zmiana tylko w środku sieci)

Ten sam `MalyViT` co dotąd (łodyga konwolucyjna, dim 192, 6 bloków, 3 głowy, 600 tokenów 20x30, głowa gęsta:
mapa środków + ltrb na patch). Po bloku k:

1. TokenLearner v1.1: LayerNorm, MLP 192 -> 192 -> 8, sigmoid; 8 tokenów = średnia 600 tokenów z wagami map.
2. Bloki k+1..6 liczą już tylko na 8 tokenach (atencja 8 x 8 zamiast 600 x 600).
3. TokenFuser: powrót na 600 patchy (resztowo na cechy z bloku k), potem ta sama głowa detekcji.

Latent do aplikacji: wyjście MLP ostatniego bloku na 8 tokenach, czyli [8, 768] na klatkę, plus 8 map uwagi
[8, 600] (która cząstka zbiera z których patchy). Te same widoki: 8 cząstek, PCA, przestrzeń makrowektorów.

## Ramiona

| Tag | Co | Po co |
|---|---|---|
| `tl8_po3` | TokenLearner 8 po bloku 3 (połowa sieci, jak w pracy) | ramię główne |
| `tl8_po1` | TokenLearner 8 po bloku 1 | ile kosztuje wczesne ściśnięcie |
| `tl16_po3` | TokenLearner 16 po bloku 3 | czy 8 to za mało dla detekcji |
| `_kontrola600` | ten sam MalyViT bez modułu, ten sam przepis i ziarno | odniesienie; osobny sufiks, żeby nie nadpisać |

## Przepis (jak model K1000 używany w aplikacji)

- Dane lokalne, bez Edenu: baza 1500 klatek treningowych i 3000 klatek walidacji
  (`experiments/maly_vit_od_zera/cache/baza1500.npy`, `wal3000.npy`, adnotacje COCO
  w `experiments/koopman_fastgan_petla/wyniki/local_1810448/run/datasets/pretrain/`).
- Od zera, paczka 32, AdamW lr 5e-4, wd 0,05, rozgrzewka 300 kroków, bf16, te same augmentacje (odbicie, jasność).
- Długość: 1000 kroków (jak K1000; mAP@0,5 kontroli na koniec około 0,31 do 0,34 w gęstej historii), potem
  ramię główne i kontrola dłużej, 10 000 kroków, jeśli 1000 pokaże sensowny wynik.
- Ocena: mAP@0,5 na 3000 klatkach walidacji, IoU, pewność; mapy uwagi 8 tokenów na kilku filmach
  (np. TR_0001 0:20–0:33, ta sama sekwencja co w folderze `app/narzedzie_czarne_tlo`).
- Zapis: wagi co 100 kroków, latent [8, 768] i mapy [8, 600] sond na każdym checkpoincie (pełna telemetria).

## Sprzęt: lokalnie wystarczy

Zmierzone 2026-10-06 na RTX 3070 (8 GB, losowe wejście 480x320, bf16, AdamW), skrypt
`scratchpad/bench_tokenlearner.py`:

| paczka | model | ms / krok | obrazów / s | szczyt pamięci |
|---:|---|---:|---:|---:|
| 32 | MalyViT 600 tokenów | 54,9 | 583 | 1,31 GB |
| 32 | TokenLearner 8 po bloku 3 | 41,6 | 769 | 0,99 GB |
| 32 | TokenLearner 8 po bloku 1 | 36,7 | 873 | 0,74 GB |
| 128 | MalyViT 600 tokenów | 789,4 | 162 | 5,00 GB |
| 128 | TokenLearner 8 po bloku 3 | 143,1 | 895 | 3,73 GB |

1000 kroków przy paczce 32 to około minuty samego liczenia na ramię, 10 000 kroków około 7 do 9 minut;
z oceną mAP na 3000 klatkach i zapisem sond razem kilkanaście minut na ramię. Cztery ramiona po 1000 kroków
to mniej niż godzina. Eden nie jest potrzebny; wchodzi dopiero przy dłuższych biegach lub wielu ziarnach.
Paczka 128 dla 600 tokenów zbliża się do granicy 8 GB (inne programy zajmowały 3,6 GB), więc zostajemy przy 32.
Bieg jako odłączony proces z logiem, jeden naraz.

## Ryzyka

- Detekcja końcówki może stracić dokładność, bo 8 tokenów ściska położenie; TokenFuser ma to naprawić, ale praca
  tego nie sprawdzała. Kontrola 600 tokenów odpowie wprost.
- Mapy uwagi mogą wyjść rozmyte (sigmoid, bez rywalizacji między mapami). Wtedy wariant z softmaxem po mapach
  jako osobne ramię, dopiero po zgodzie.

## Do decyzji użytkownika

1. Ramiona: wszystkie cztery, czy tylko `tl8_po3` plus kontrola?
2. Długość: najpierw 1000 kroków, czy od razu 10 000?
3. Moduł map: wersja MLP (1.1, prostsza) czy cztery konwolucje 3x3 (jak w eksperymentach obrazowych pracy)?
