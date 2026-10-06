# Konwerter latentu małego ViT na cechy DINOv2, uczony na strumieniu filmów (Eden, DGX)

Stan: kod gotowy, lokalny test dymny przeszedł 2026-10-06. **Zgłoszenie na Eden czeka na zgodę użytkownika.**

## Pytanie

Czy z pełnego naturalnego latentu zamrożonego małego ViT (`MODEL_VIT_K1000.pt`, aktywacje po GELU ukrytej
warstwy MLP ostatniego bloku, 600 tokenów x 768) da się przewidzieć cechy DINOv2 ViT-S/14 tego samego zdjęcia
na filmach VAL, których konwerter nigdy nie widział, lepiej niż średnia i lepiej niż liniowa głowa na
uśrednionym latencie?

Wcześniejsze lokalne próby (`vit_translator_latent_dino`, `vit_translator_mlp_random10000`) miały za mało par
(400, 10 000) i zwykłą warstwę gęstą na 460 800 wejściach. Wynik: cosinus 0,80 przy cosinusie samej średniej
0,81 i top1 wyszukiwania 1,4 procent w galerii 512. Ten bieg zmienia dwie rzeczy naraz: liczbę par (cały TRAIN, ok. 313 tys.
klatek, kilkanaście przejść) i architekturę (transformer na tokenach z zachowaną siatką 20x30).

## Dane: tylko filmy MP4, w locie

- 118 filmów TRAIN, 25 filmów VAL, podział całymi filmami z `FILMY.json` (ten sam co bieg SD 1823167).
- Każdy proces ładujący czyta MP4 z ZIP do RAM, dekoduje PyAV, z natywnej klatki robi trzy obrazy:
  ViT 480x320 (INTER_AREA, jak w całym potoku), DINO CLS 224 (dokładnie `DINO_SPEC.json`: PIL Resize 256
  bicubic + CenterCrop 224, zgodne z indeksem FAISS 35000), DINO tokeny 420x280 (siatka 20x30, ta sama co ViT).
- Na GPU: zamrożony ViT liczy latent, zamrożone DINOv2 liczy cele, konwerter się uczy. Nic z tego nie trafia na dysk.
- Mieszanie: każdy proces przeplata 4 filmy i trzyma bufor 256 klatek w RAM, z którego losuje.
- VAL: co 15. klatka każdego filmu VAL (ok. 4400 klatek), policzona raz i trzymana w pamięci GPU.

## Model

- Konwerter: standaryzacja kanałów latentu (stała, z 40 paczek TRAIN przed uczeniem), Linear 768 -> 384,
  uczone pozycje 600 tokenów, token CLS, 6 bloków transformera (384, 6 głów), dwie głowy:
  CLS 384 (L2) i 600 tokenów 384. Ok. 11 mln parametrów.
- Kontrola w tym samym biegu, na tych samych paczkach: liniowa głowa na uśrednionym latencie (bez układu przestrzennego).
- Odniesienie bez uczenia: średni CLS DINO z TRAIN.

## Strata

`(1 - cos CLS) + 0,5 * (1 - cos tokenów) + 0,2 * InfoNCE(CLS)`, temperatura 0,07, negatywy ze wszystkich
4 GPU (paczka globalna 256). Klatki tego samego filmu bliżej niż 30 klatek wypadają z negatywów, bo to prawie ten sam obraz.

## Budżet

4 x A100 na `dgx` (partycja `short`), 16 rdzeni, 96 GB RAM, limit 8 h. 20 000 kroków x 256 klatek = 5,1 mln
klatek, ok. 16 przejść przez TRAIN. Szacunek z pomiaru lokalnego: ok. 115 klatek/s na proces dekodujący, 12 procesów,
więc ok. 1 h; rzeczywistą prędkość poda SMOKE.json. Brak doboru modelu na VAL: stały harmonogram, ocena ostatniego kroku,
więc VAL pozostaje uczciwym zbiorem odłożonym.

## Kontrole

1. Test dymny na Edenie przed uczeniem (to samo zadanie): TF32 kontra FP32 dla celów DINO (próg cosinusa 0,999)
   i latentu, kształty, skończone i niezerowe gradienty konwertera, brak gradientów ViT i DINO, identyczne
   predykcje po odczycie checkpointu, zmierzona prędkość. Uczenie rusza tylko przy `passed=true`.
2. Metryki VAL co 2000 kroków dla konwertera, kontroli liniowej i średniej: cosinus surowy, cosinus po odjęciu
   średniej TRAIN (średnia daje 0), top1/top5 i mediana pozycji w galerii VAL, top1 z tolerancją sąsiedniej
   próbki, trafienie filmu (z losowym odniesieniem), mediana po filmach, cosinus tokenów.

## Kryteria

- Konwerter ma sens, gdy cosinus wycentrowany na VAL jest wyraźnie dodatni i wyższy niż w kontroli liniowej,
  a top1 i mediana pozycji biją wynik pilota MLP.
- Obala: konwerter nie bije kontroli liniowej, czyli układ przestrzenny latentu nic nie dodaje do przewidywania DINO.

## Czego bieg nie rozstrzyga

- Działania na sztucznym latencie (spoza naturalnego rozkładu), np. celu z `vit_mlp_cel_sztuczny_faiss_top32`.
- Zgodności z indeksem FAISS zbudowanym z JPEG-ów: uczymy na klatkach MP4, a kompresja obu źródeł się różni.
  CLS liczony jest tym samym przepisem, ale różnicę domeny trzeba zmierzyć osobno.
- Innych checkpointów ViT: konwerter jest związany z k1000.
