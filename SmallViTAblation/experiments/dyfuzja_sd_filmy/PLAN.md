# Stable Diffusion uczone na całych klatkach filmów Cataract-LMM, na wielu kartach Edenu (plan, 2026-09-30)

## Prośba użytkownika

„Po wrzuceniu filmów możesz opracować, jak uczyć Stable Diffusion model z całych filmów. Tym razem trening modelu możesz
[uruchomić] na tylu GPU, ile dasz radę na Edenie.” Wcześniej: „wrzuć filmy z Cataract-LMM na mój dysk [na Edenie]”.

Stan wyjścia: `../dyfuzja_narzedzie` uczy (inna sesja) własny mały model dyfuzyjny w przestrzeni pikseli (UNet 22 mln
parametrów, wycinki 176 px wokół narzędzia, co piąta klatka, jedna karta H100). Ten plan to osobny, większy model:
prawdziwe Stable Diffusion, całe klatki, wszystkie klatki filmów, wiele kart.

## Dane (na Edenie od 2026-09-30)

`~/smallvit_data/cataract_lmm/3_Object_Tracking/`: `videos/videos_part00{1,2,3}.zip` (170 filmów MP4, 5,6 GB) i
`annotations/<film>.zip` (170 archiwów, 42 GB): każda klatka jako JPEG 720×480 (część filmów 720×576) oraz
`annotation.csv` z wielokątami trzech obiektów na każdej klatce: cystotomu (`cystotome_seg`, narzędzie), rogówki
(`cornea_seg`) i źrenicy (`pupil_seg`). Razem 469 118 klatek fazy kapsuloreksji.

Podział: tylko 118 filmów `train` z `pool_manifest.jsonl` (~313 tys. klatek). 25 filmów `val` i 27 filmów spoza
manifestu nie wchodzą do uczenia: służą do oceny. Obrazy z modelu mają trafiać do uczenia detektora, więc model nie może
widzieć filmów walidacyjnych (inaczej przeciek walidacji do danych uczących).

## Model

- **Stable Diffusion 1.5** (`stable-diffusion-v1-5/stable-diffusion-v1-5`, licencja CreativeML OpenRAIL-M, dostępne
  bez ograniczeń) z lepszym dekoderem `stabilityai/sd-vae-ft-mse` (MIT). Model rysuje w **przestrzeni latentnej**:
  koder VAE ściska obraz 8 razy w każdym wymiarze do 4 kanałów, UNet (860 mln parametrów) zdejmuje szum w tej
  ściśniętej przestrzeni, dekoder VAE zamienia wynik z powrotem na obraz. Start z wag uczonych na LAION, więc model
  zna już tekstury i światło; dostrajamy go na klatkach operacji.
- **Rozdzielczość 576×384** (latent 72×48; proporcje 3:2 jak klatka 720×480). Do małego ViT obraz zmniejszamy do
  480×320. Klatki 720×576 ściskamy do 576×384 tak samo jak robi to wejście małego ViT (spójność z detektorem).
- **Warunek: układ sceny.** Do zaszumionego latentu dokładamy 3 kanały masek w rozdzielczości latentu (narzędzie,
  rogówka, źrenica). Pierwsza warstwa UNet dostaje 7 zamiast 4 kanałów wejścia, nowe wagi startują od zera (jak
  w modelach inpaintingu SD). W 20 % przykładów maski są zerowane, więc model umie też rysować bez warunku.
  Po co: przy generowaniu podajemy obrys narzędzia w wybranym miejscu i kącie, więc wiemy, gdzie są tokeny narzędzia
  dla siły tożsamości małego ViT. To zastępuje wygenerowany kanał obrysu z modelu pikselowego.
- Tekst nie gra roli: stałe osadzenie pustego napisu z kodera CLIP policzone raz (koder tekstu nie jest ładowany).
- Harmonogram szumu SD 1.5 (DDPM, 1000 kroków, przewidywanie szumu), przesunięcie szumu 0,05 (offset noise), żeby
  model umiał ciemne tło klatek. Próbkowanie DDIM 50 kroków albo DPM-Solver 25 kroków.

Różnice wobec `dyfuzja_narzedzie` (nazwane, do zatwierdzenia): przestrzeń latentna zamiast pikseli; model gotowy
dostrajany zamiast uczonego od zera; całe klatki zamiast wycinków 176 px; wszystkie klatki zamiast co piątej; maska
jako warunek wejściowy zamiast rysowanego czwartego kanału; wiele kart zamiast jednej.

## Przygotowanie danych (`kod/przygotuj_latenty.py`, zadanie GPU)

Każdy film: JPEG z archiwum → 576×384 → koder VAE (bf16, średnia rozkładu × 0,18215) → `latenty/<film>.npy`
(N, 4, 48, 72) float16; maski wielokątów rysowane w pełnej rozdzielczości i zmniejszane do 72×48 (udział pola,
uint8) → `maski/<film>.npy` (N, 3, 48, 72); `indeks/<film>.json` (numery klatek). ~313 tys. klatek × 38 KB ≈
12 GB, 354 pliki. Wiele kart dzieli filmy między siebie. Szacunek: 20–40 min na 4 kartach.

## Uczenie (`kod/trenuj_sd.py`, `eden/trenuj_sd.slurm`)

- `torchrun --nproc_per_node=<liczba kart>` na jednym węźle: hopper (8 × H100, limit 5 dni) albo dgx (8 × A100,
  limit 24 h). Kod działa na dowolnej liczbie kart; **stała paczka globalna 256** (akumulacja gradientu dopasowuje się
  do liczby kart), więc wynik nie zależy od tego, ile kart akurat było wolnych.
- AdamW lr 2e-5 z rozgrzewką 500 kroków, bf16, obcinanie normy 1, EMA wag 0,9999, SDPA (wbudowana uwaga PyTorch).
- Budżet: 20 000 kroków × 256 = 5,1 mln przykładów (~16 przejść przez 313 tys. klatek). Szacunek przepustowości SD 1.5
  przy 72×48: ~20 obrazów/s na A100 i ~40 na H100, czyli 8 × H100 ~ 9 h, 8 × A100 ~ 18 h, 4 × A100 ~ 36 h
  (dwa fragmenty). Pierwsze zadanie próbne zmierzy to naprawdę.
- Zapis co 1000 kroków (`ostatni.pt`: UNet, EMA, optymalizator, krok; ~14 GB, jeden plik nadpisywany atomowo),
  `ema_XXXXXX.safetensors` w fp16 (1,7 GB) co 5000 kroków, podgląd 16 próbek co 2000 kroków (maski z filmów `val`),
  `LOG.jsonl`, `STATUS.json`. SIGTERM przed końcem czasu → zapis i czyste wyjście; kolejne zgłoszenie wznawia.
- Katalogi tymczasowe (`TMPDIR`, pamięć podręczna Tritona i torch.distributed) w `/dev/shm/<zadanie>`, kasowane pułapką;
  nic nie zostaje na węźle.

Kwota po wysyłce filmów ~105 GiB; dojdzie ~12 GB latentów, ~14 GB `ostatni.pt`, ~7 GB migawek EMA: razem ~140 GiB z 200.

## Ocena (`kod/ocena_sd.py`)

- Podglądy i próbki z masek filmów `val` (układ, którego model nie widział).
- FD-DINOv2 (odległość Frécheta w przestrzeni cech DINOv2 ViT-S/14; wagi już są na Edenie) między 5000 próbek a 5000
  klatek `val`; odniesienie: ta sama miara między klatkami `train` i `val`.
- Kopiowanie: odległość każdej próbki do najbliższej klatki uczącej w DINOv2; próbki bliżej niż sąsiednie klatki tego
  samego filmu oznaczają kopię.
- Zgodność z maską: czy mały ViT (albo prosty segmentator) widzi narzędzie tam, gdzie była maska.

## Dalej (osobne kroki, osobna zgoda)

Prowadzenie siłą tożsamości małego ViT (FreeDoM jak w `dyfuzja_narzedzie/kod/generuj.py`, gradient przez dekoder VAE).
Ewentualnie warstwy czasowe (sekwencje klatek) — dopiero gdy model pojedynczych klatek będzie dobry.

## Kroki (zatrzymanie przed zgłoszeniem)

1. Lokalnie: próba kodu na małym UNet i 2 filmach (RTX 3070), koder VAE na kilkudziesięciu klatkach, sprawdzenie masek.
2. Po zgodzie: wysyłka kodu, pobranie wag SD na Eden (~2,5 GB), `diffusers` do katalogu `deps`.
3. Zadanie próbne: przygotowanie 2 filmów + 200 kroków uczenia na 2 kartach (pomiar s/krok, pamięć, zapis i wznowienie).
4. Przygotowanie wszystkich 118 filmów, potem uczenie na tylu kartach, ile będzie wolnych (8 docelowo).

## Czego to nie dowodzi

Dobry wygląd próbek ani niska FD-DINOv2 nie znaczą, że obrazy pomogą detektorowi. Użycie generatora w pętli uczenia
detektora to osobne ramię i osobna kampania.
