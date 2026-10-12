# Exact1500: pełne stany cząsteczek do NNDMD

Data: 2026-10-02T14:35:03+02:00 (Europe/Warsaw). Dokument zapisany przed benchmarkiem i pełnym cache. Dane i zasoby: plan_poprawek; model, trening i wybór: pykoopman_obliczenia; audyt read-only: Astra. Osobny eksperyment, bez nadpisywania dawnych wyników.

## Pytanie i granice dowodu

Czy NNDMD na wszystkich 1500 obrazach źródłowych, wszystkich 600 tokenach i pełnych 768 kanałach daje użyteczną autonomiczną prognozę przyszłych 10 checkpointów? Kontrole modelu to persistence, velocity/reversion i jawny EDMD. Wszystkie widzą tę samą dostępną historię i te same cele. Brak przewagi, rozbieżny rollout albo błędna amplituda ograniczają wniosek o użyteczności, również przy wysokiej trafności znaku.

85 procent znaków J128 pozostaje celem, nie wynikiem. Rozpoznanie obecności ponad 90 procent jest innym zadaniem. Ocena przyszłości tych samych 1500 obrazów to known-image temporal test, nie holdout nowych obrazów ani niezależny nowy trening ViT. J128 pochodzi z historycznej identyfikacji na k500; learned state768 nie jest listą oryginalnych kanałów.

Nowa autoryzacja 1500 obrazów odwołuje wcześniejsze zatrzymanie. Częściowe pięć obrazów BASE1500_CORRECTED pozostaje cancelled; nowy reader ich nie używa i nie traktuje jako complete.

## Dokładne źródła i preprocessing

COCO train pochodzi z maly_vit_od_zera/eden/wejscia/pretrain/train/_annotations.coco.json. Kolejność coco.images odpowiada źródłowemu kod/dane.py i cache baza1500.npy. Wszystkie 1500 frame-ID są unikalne. Przygotuj_manifest.py sprawdził dla każdego obrazu bitową zgodność RGB cache z jego źródłowym plikiem po CV2 resize INTER_AREA. Zapisano source-ID/film/frame, źródłowy hash obrazu, hash RGB, anotacje, hash COCO i cache.

Wejście: uint8 RGB (1500,320,480,3), źródłowe obrazy 720×480 pomniejszone do480×320. Do modelu FP32/255, ImageNet normalizacja wewnątrz źródłowego forward. Miejsce pomiaru: bloki[-1].mlp.act, GELU ostatniej MLP, (600,768). Eval/inference_mode, bez autocast i TF32, wszystkie parametry frozen, bufory BatchNorm bez aktualizacji. Zmiany pomiędzy checkpointami obejmują wagi i bieżące statystyki BatchNorm.

Dostępne są wszystkie 101 checkpointów 0..1000 co10. Indeks checkpointu90 to optimizer-step900. Każdy checkpoint ma zapisany hash; source architektura jest skopiowana losslessly do model_snapshot.py z tym samym SHA. Cztery wcześniejsze sondy rzeczywiście należą do tej bazy i mają identyczne piksele; pozostałych ośmiu nie dopisujemy do źródłowych1500.

COCO ma 1500 anotacji instrument z bbox XYWH, **bez segmentacji**. Nie fabrykujemy maski z bbox. Główna funkcja celu obejmuje wszystkie 600 tokenów z wagą1/600 w obrębie obrazu i równą masą obrazów/horyzontów. Bbox może służyć dodatkowej opisowej ocenie, ale nie jest maską narzędzia.

## Zasoby i ograniczony cache

Audyt: F około 118,7GB wolne, RAM około 16,9GB wolne z 34,2GB, RTX 3070 około 5848MiB wolne. Pełny dump101×1500× 600× 768 float32 potrzebowałby 279244800000B. Nie zapisujemy go i nie redukujemy po cichu 1500 obrazów,600 tokenów lub 768 kanałów.

Rolling cache ma **11 pełnych checkpointów na dysku**, razem 30412800000B plus nagłówki. To mmap, nie 30,4GB resident RAM. Reader pobiera tylko potrzebne obrazy: okno 11×batch 32× 600× 768 float32 to648806400B. Microbatch/gradient accumulation ma zachować32 obrazy jako logical batch, a nie 32 tokeny.

Stałe sloty 0..10 są zastępowane modulo 11 wyłącznie w nowo utworzonym katalogu cache_window11. Nie usuwamy ani nie nadpisujemy żadnych starych danych. Przed budową slot ma status building; complete dopiero po zapisaniu 1500 obrazów, finite, hash i kontroli buforów. Przerwany slot nie jest czytany jako gotowy. Append-only CACHE_EVENTS.jsonl zachowuje provenance już przetworzonych checkpointów, nawet po zmianie zawartości rolling slotu.

## API i harmonogram

FrozenWindowReader w dane.py:

- begin_epoch(phase,epoch) zapisuje fazę i epokę.
- ensure_window(origin,width=10) zapewnia stany checkpointów origin..origin+10.
- get_batch(image_indices,origin) zwraca (11,N,600,768) float32, kopiując tylko wskazane obrazy.

Origin jest **indeksem checkpointu**, nie krokiem optymalizatora. Source ViT wykonuje ekstrakcję w tym samym procesie co NN, po zwolnieniu grafu ostatniego microbatcha. Po ensure_window jest przenoszony do CPU, a NN może pozostać GPU. Nie uruchamiamy konkurencyjnych procesów GPU. Pomiar eval nie aktualizuje source parametrów i nie powinien zmieniać RNG treningu NN.

Harmonogram window-major: wewnętrzny fit 0..80 daje 71 okien origins 0..70; wewnętrzna walidacja 80→90. Final refit 0..90 daje 81 okien origins 0..80. Final known-image future test 90→100 obejmuje 10 przejść bez przyszłych korekt pomiarowych. Każdy origin przechodzi wszystkie 1500 image-ID z seeded shuffle: 46 pełnych batchów 32 i ostatni 28. Ostatni niepełny batch musi mieć poprawną normalizację obrazu, nie być usuwany.

Kolejna epoka może wymagać ponownego odczytu/ekstrakcji checkpointów po wyparciu z ograniczonego cache. Koszt jest jawny; nie losować losowo originów prowadząc do thrashing. Benchmark source 32 obrazów i gradientowego NNbatcha poprzedza decyzję o liczbie epok oraz ETA. Pełna ekstrakcja81+checkpointów nie zostaje uruchomiona przed uzgodnieniem budżetu.

## Kontrole i odtwarzanie

Manifest1500 oraz source obrazy/cache są zweryfikowane. Cache pełnych source reprezentacji na checkpointach 0/500/1000 porównuje cztery wspólne sondy z LATENT_GESTY, z maxabs i relativeL2 i tolerancją 1e-5. Wszystkie 600× 768 zachowane. Bez przyszłych reszt, wag albo latentów jako wejść rollout; cache przyszłości zostaje użyty tylko w odłożonej fazie oceny.

Source RNG CPU/CUDA jest zachowany przy tworzeniu frozen modelu, a eval forward nie losuje augmentacji. Source model jest offloadowany po ekstrakcji. Prefix/poison NN musi dodatkowo objąć reader i preprocessing, normalizację oraz autonomiczny rollout. Normowanie jest tylko z historycznej części, np. common scalar zcp0 wszystkich 1500, zgodnie z ostatecznym planem NN.

Każdy slot zapisuje indeks/krok checkpointu, zakres wszystkich 1500 image-ID, shape/dtype, finite, SHA, hash manifestu/checkpointu, kontrolę BN i czas. Raport zawiera coverage originów i obrazów, nie samą liczbę optimizer-step NN. Source bbox/coordinate convention pozostaje jawne.

## NNDMD i status

Nieliniowy encoder 768→768, A768× 768 i **nieliniowy decoder 768→768** to bieżący projekt modelu. Bieżąca funkcja celu to MSE full768 z równymi wagami obrazów i horyzontów (beta dodatkowego J128=0), z rekonstrukcją zgodnie z protokołem NN; J128 oceniane dopiero po decode. Wcześniejsza propozycja dodatkowego betaJ128 została zastąpiona tym pełnoobrazowym wariantem. Wewnętrzna macierz learned nie jest macierzą natywnych osi J. Dobór/protokół/freeze NN należy do agenta obliczeniowego i audytu Astry, bez strojenia na final90→100. Nie zmieniamy source ViT ani jego optymalizatora.

Aktualny etap: źródła verified, reader przygotowany, benchmarki poprzedzają ustalenie realnego budżetu. To nie wynik jakości prognozy. Zachowujemy wszystkie negatywy i awarie. Bez pobierania i bez nowego treningu ViT.

Warunek cache: dokładnie jeden writer, bez równoległego ensure_window lub prefetch. _valid sprawdza complete/index/provenance checkpointu, NPY shape/dtype/bytes i SHA raz przy ponownym użyciu w procesie; partial/stary slot jest odrzucany.
