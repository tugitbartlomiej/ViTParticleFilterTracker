# Audyt rozszerzonego NNDMD na 1500 obrazach

Data: 2026-10-02, Europe/Warsaw. Audytor: Astra. Status o 15:11: **audyt przed uruchomieniem zakończony — PASS; wyniki treningu jeszcze nie istnieją**. Poniższe wpisy zachowują chronologię kontroli; końcowy werdykt i hashe znajdują się na końcu.

Zakres własności audytora obejmuje ten raport. Odczytuję kod i małe artefakty innych wykonawców; nie zmieniam ich implementacji, parametrów ani danych. Nie korzystam z GPU. Wcześniejsza kontrola modelu z samym z768 znajduje się w [odrębnym audycie](../maly_vit_nndmd_1500_obrazow_768/AUDYT.md) i nie zatwierdza automatycznie nowego modelu rozszerzonego.

## Intencja i granice interpretacji

Najnowsze zlecenie użytkownika zatrzymuje wcześniejsze lokalne przebiegi i przenosi nowy trening na Eden. Wejście ma obejmować pełną cząsteczkę z768, historyczne przyrosty tej samej cząsteczki oraz historyczne charakterystyki zmian parametrów i zapisanych stanów AdamW. Wyuczony stan ma 768 współrzędnych, wspólny operator A ma rozmiar 768×768, decoder ma być nieliniowy. J128 służy dodatkowej ocenie po dekodowaniu, nie dociążeniu straty treningowej.

600 tokenów jednego obrazu to 600 cząsteczek. Paczka 32 obrazów nie oznacza paczki 32 tokenów. Wspólny operator działający osobno na cząsteczkach nie identyfikuje sam z siebie oddziaływań między tokenami. Rozszerzone wejście kompresowane do learned state768 nie ma gwarancji odwracalności ani zachowania pełnej informacji. Uczenie encodera nie jest PCA, ale nadal jest uczoną zmianą reprezentacji.

Dane obejmują jeden historyczny bieg uczenia ViT. 1500 obrazów i 900 tysięcy tokenów nie tworzą tylu niezależnych realizacji aktualizacji wag ani AdamW. Wyniki dotyczą predykcji reprezentacji znanych obrazów przy odłożonym przedziale czasu, nie nowych niezależnych biegów uczenia.

## Zweryfikowany odczyt danych i telemetrii, 14:53

### Manifest i rolling cache

Manifest zachowuje dokładnie 1500 źródłowych obrazów, ich kolejność i hashe pikseli, 101 checkpointów co 10 aktualizacji oraz model snapshot. Wcześniejszy lokalny generator zakończył sprawdzenie zgodności wszystkich 1500 obrazów z cache źródłowym. Nowe [przygotuj_eden.py](przygotuj_eden.py) ponownie sprawdza hashe źródłowych plików i wynikowych obrazów RGB po identycznym przetwarzaniu. Zbiór zawiera bbox XYWH, bez masek segmentacji; trening wszystkich 600 tokenów nie wymaga fabrykowania masek z bbox.

[dane.py](dane.py) używa 13 slotów: dla origin t przechowuje cp od max(0,t−2) do t+10. Przed nadpisaniem slot ma status building, po ukończeniu wszystkich 1500 obrazów zapisuje metadane i SHA. Odczyt sprawdza cp, manifest, checkpoint, rozmiar, nagłówek NPY, dtype, shape i SHA. Źródłowy ViT działa w eval; buforów BN nie wolno zmienić. Batch jest kopią wybranych obrazów, a nie widokiem utrzymywanym podczas nadpisywania cache.

Konieczny jest pojedynczy writer cache i porządek window-major. `get_history_batch` zwraca kolejno `[max(0,t−2), max(0,t−1), t, ..., t+10]`. Clamp do cp0 zapobiega odczytowi przyszłości, lecz caller musi jawnie oznaczać brak historii dla t=0 i t=1. Pierwsze powtórzone z0 nie są rzeczywistymi wcześniejszymi pomiarami.

### Rzeczywiste stany AdamW

Źródłowe [trening.py](../maly_vit_ruch_czastek_gesta_historia/kod/trening.py) zapisuje `stan()` jako net, opt i krok. Zgodnie z metadanymi pełne stany zapisano co 100 aktualizacji, czyli k0000, k0100, ..., k1000. Wagi zapisano co 10 aktualizacji. Wykonawca danych potwierdził na Edenie istnienie 11 pełnych plików i w k0100 89 stanów parametrów zawierających step, exp_avg oraz exp_avg_sq; ja nie ładowałem tych zdalnych plików.

[telemetria.py](telemetria.py) sprawdza zgodność net pełnego checkpointu z checkpointem wag. Porządek parametrów odtwarza konstrukcyjnie: `named_parameters`, najpierw ndim≥2, potem ndim<2. Jest to dokładnie kolejność dwóch grup w odczytanym źródłowym konstruktorze AdamW. Samo sprawdzenie kształtu byłoby niewystarczające przy kilku parametrach tego samego kształtu; dowodem jest dodatkowo zgodny snapshot architektury i odtworzenie tego samego konstruktora. Zaleciłem zapis hasha źródłowego trening.py i jawne opisanie tej podstawy w metadanych.

Parametry i pływające bufory BN mają osobne RMS zmian. RMS exp_avg i exp_avg_sq jest statystyką amplitudy, nie wektorem kierunku gradientu ani pełnym stanem optymalizatora. Przyrost wag między checkpointami jest zmianą netto dziesięciu aktualizacji, nie pojedynczym gradientem. Takie charakterystyki mogą pomagać predykcyjnie, lecz nie identyfikują przyczynowego sterowania.

Wejście momentów używa ostatniej faktycznie zapisanej obserwacji oraz jej wieku i maski dostępności. Target momentów ma ważną etykietę tylko przy faktycznym pełnym zapisie z niepustym stanem. Pusty opt w cp0 oznacza brak obserwacji momentów, nie zmierzone zerowe momenty. Niedozwolona byłaby interpolacja między dwoma zapisami z użyciem późniejszego snapshotu.

## Warunki do sprawdzenia przed sbatch

1. Inner fit używa stanów cp≤80, zatem okna długości 10 kończą się najpóźniej w cp80. Walidacja to autonomiczny rollout cp80→90. Refit po zamrożeniu wyboru używa cp≤90; test cp90→100 nie podaje prawdziwych przyszłych stanów na wejście.
2. Scaler bloków z, Δz i aux powstaje wyłącznie z odpowiedniego fit prefixu. Brakujące etykiety momentów i cp0 nie mogą wejść jako rzeczywiste zera do estymacji skali. Test prefix-poison musi objąć także aux, maski, historię oraz preprocessing.
3. Prawdziwe przyszłe aux mogą być etykietami treningowymi wewnątrz fit prefixu. Nie mogą być wejściami podczas autonomicznego rollout. Upływ czasu do zaplanowanego snapshotu nie oznacza otrzymania jego wartości; wiek obserwacji nie może samoczynnie resetować się w prognozie.
4. Należy rozróżnić encode-once, A^h i wspólne dekodowanie przyszłego z/aux od pętli reencode z przesuwaniem historii. Obie konstrukcje da się zrobić przyczynowo, ale to inne modele. Historia Δz musi wynikać ze zmian prognoz albo jej niezgodność powinna być raportowana; nie wolno poprawiać jej prawdziwą przyszłością.
5. Strata pełnego z768 ma osobną, jawną normalizację. Powielona po tokenach i obrazach jedna globalna trajektoria aux nie może otrzymać wagi rosnącej z liczbą kopii. Dla aux z brakami mianownik uwzględnia wyłącznie ważne etykiety i zachowuje ustaloną wagę bloku.
6. Gradient accumulation obejmuje wszystkie 600 tokenów każdego z faktycznych B obrazów, także ostatnią paczkę B=28. Clipping i optimizer.step następują raz na paczkę obrazów. RMSE to pierwiastek po agregacji MSE, a nie średnia RMSE dowolnych mikrofragmentów.
7. Regularizacja odległości A od I nie jest warunkiem stabilności spektralnej. Należy osobno raportować promień spektralny, normy potęg i błędy rekonstrukcji. Żadna z tych kontroli nie gwarantuje osiągnięcia 85 procent znaku.

Na tym etapie nie odczytałem jeszcze kompletnego nowego modelu i drivera treningowego. **Brak końcowego werdyktu przed uruchomieniem.**

## Aktualizacja 15:00: rdzeń expanded i rzeczywista telemetria

Odczytano [state.py](state.py), [model.py](model.py), [test_expanded.py](test_expanded.py), zakończony log pięciu testów oraz pobrane AUX_HISTORY.npz/json. Nadal oczekuję kompletnego drivera przed werdyktem dotyczącym całego joba.

### Stan i przyczynowość rdzenia

Rzeczywiste Q=59 to cztery rodzaje statystyk dla 14 grup oraz trzy cechy dostępności/wieku. Wejście ma wymiar `3·768 + 4·59 + 4 = 2544`: z, dwa przyrosty, aux bieżące i poprzednie, ich maski, dwie flagi historii i dwa wieki. Encoder 2544→768→768 i decoder 768→768→827 mają aktywację SiLU w warstwie ukrytej. Output827 obejmuje pełne z768 i 59 proxy aux. Model korzysta z rzeczywistych modułów PyKoopman NNDMD, ale ma jawnie zastąpiony trening/stratę i postprocessing; nie jest to niemodyfikowane wywołanie stock `fit`.

`rollout` otrzymuje wyłącznie trzy stany do origin, koduje początek raz i wykonuje `phi_h = A phi_(h−1)`. Funkcja `aux.input` jest wywołana tylko dla origin i origin−1, z `allowed=origin`. Następne decoded aux są wyłącznie prognozami; ich średnia i rozrzut między tokenami nie są podawane ponownie na wejście. To autonomiczny rollout A^h, a nie pętla pomiarowa obserwatora. Obliczenie historii różnic prognoz nie wpływa na ten rollout. Nie wykryłem teacher forcing w tej funkcji.

W treningu przyszłe stany wewnątrz dozwolonego okna są osobno kodowane dla reconstruction loss. Nie zastępują learned state w gałęzi prediction; odczyt prawdziwych etykiet wewnątrz fit prefixu jest legalny. Guard blokuje origin+H>allowed. Dla origin0 i1 nieobserwowane różnice są zerowane i oznaczone flagą braku historii.

### Normalizacja i utrzymanie równej wagi obrazów

Normalizer aux używa tylko zakresu `[:fit_last+1]`. Średnie i wariancje liczy po wartościach z ważną maską, oddzielnie dla input i target. Scaler natywny jest podany z drivera: jego poprawność czasowa wymaga jeszcze odczytu drivera.

Dla B obrazów, P=600 tokenów i D=768 główna strata prognoz to średnia kwadratowego błędu po H, B, P i D w jednostkach z/native_scale. Aux ma oddzielną wagę i średnią po B, P oraz liczbie ważnych cech danego kroku. Powielanie jednej etykiety globalnej nie mnoży wagi tego bloku przez liczbę tokenów. Nie czyni jednak tych etykiet niezależnymi obserwacjami.

Reconstruction loss jest sumą błędu origin oraz średniej błędów przyszłych punktów okna; nie jest pojedynczą średnią wszystkich H+1 stanów. To jawny wybór wag. Pole `reconstruction_native_mse_raw` odzwierciedla tę sumę dwóch składników. Pole `aux_prediction_normalized_SSE` przechowuje surową sumę po tokenach, nie końcową miarę MSE; jego wartości nie wolno prezentować jako MSE bez właściwego mianownika.

### Niezależna kontrola gradientów CPU

Zbudowałem ręcznie pełną stratę Torch, niezależnie od `_loss_piece` i `direct_loss`, i porównałem wszystkie gradienty z `accumulated_gradients`. Użyto nieliniowego encodera i decodera, D=4, learned=5, hidden=7, H=10, scale=2,3, wag reconstruction=0,7, aux=0,2 i identity=0,03; maski zawierały brakujące etykiety momentów. Nie wykonano optimizer.step ani treningu produkcyjnego.

| Paczka | Mikrofragment | Bezwzględna różnica loss | Maksymalna różnica gradientu |
|---|---:|---:|---:|
| B=3, P=5 | 6, niepełny ogon | 5,2407742e−8 | 1,4901161e−8 |
| B=28, P=2 | 17, niepełny ogon | 7,7042586e−9 | 7,4505806e−9 |

Wszystkie parametry spełniły rtol=3e−5, atol=1e−6. Osobna kontrola dwukrotnego powielenia całej populacji tokenów dała różnicę loss1,4901161e−8 i maksymalną różnicę gradientu1,8626451e−9. Mechanizm akumulacji oraz jednorazowego backward przez exp(G) jest zgodny z pełną stratą w tej kontroli.

### Sprawdzenie pobranej telemetrii

SHA AUX_HISTORY.npz zgadza się z metadanymi: `df64e3bdc049b444c477de13701ddc90a56e1b2065a732513f19f95941470bfa`. Niezależnie sprawdzono wszystkie kopie hold-last momentów oraz dokładny zbiór ważnych moment targets: cp10,20,...,100. W cp0 input moments są nieważne. Dla cp89/90/91/100 wieki wynoszą90/0/10/0 aktualizacji. To opis obserwowanej historii, nie harmonogram automatycznego otrzymywania przyszłych obserwacji przez rollout.

Metadane obecnie poprawnie deklarują podstawę mapowania parametrów i zapisują SHA trening.py `6e2ea34909ed4383d8480ec93d01bbb9595bec9affc21b45debedceb15080284` oraz zgodny SHA modelu. Raport wykonawcy potwierdza89 stanów parametrów w każdym pełnym checkpointcie poza pustym cp0. Nie odczytywałem przyszłych metryk finalnego modelu; nie zostały jeszcze obliczone.

## Końcowy audyt przed jobem, 15:11

**Werdykt: nie stwierdzono otwartego błędu blokującego uruchomienie kontrolowanego eksperymentu według sprawdzonego protokołu.** Właściciel joba może wykonać autoryzowany submit po potwierdzeniu zgodności uploadowanych hashy i testów docelowego runtime. To ocena konstrukcji i obliczeń, nie potwierdzenie skuteczności prognozy, zakończenia treningu lub osiągnięcia progu 85 procent.

Odczytano finalne [train.py](train.py), [CONFIG.json](CONFIG.json), [PLAN_MODELU.md](PLAN_MODELU.md), [PLAN_EDEN.md](PLAN_EDEN.md), [bieg_eden.slurm](bieg_eden.slurm) oraz testy drivera. Wykonawca zobowiązał się nie zmieniać wykonywanych źródeł podczas joba. Runtime używa PyKoopman1.2.1 na Torch2.9.1+cu128 potrzebnym źródłowemu ViT; deklarowane zależności starego pakietu nie są tym samym co test zgodności wykonania. Lokalny log wskazuje 12 zakończonych testów rdzenia i drivera. Odczytany log docelowego Eden CPU potwierdza na tym etapie 5 testów rdzenia; job owner wykonuje również końcowy zestaw 12 testów po uploadzie.

### Zakres czasu i wybór

- `training_origins`, train.py:27, daje 71 origins0..70 przy inner fit do cp80 i 81 origins0..80 przy refit do cp90. Wszystkie okna mają 10 przejść i nie przekraczają granicy odpowiedniego fitu.
- Native scalar, train.py:185, wyznaczany jest tylko z cp0 wszystkich 1500 obrazów, bez centrowania. Aux normalizatory powstają z prefixów kończących się odpowiednio w80 i90. J128 nie uczestniczy w stracie ani wyborze epoki.
- Jeden wariant, maksymalnie10 pełnych epok; patience2 i min_delta1e−5 dotyczą zatrzymania. Best epoch jest minimum pełnego768 walidacyjnego `sqrt(mean MSE)` po wszystkich obrazach i10 horyzontach; przy remisie wcześniejsza epoka. Różnica między kryterium zatrzymania i wyborem minimum jest jawna.
- Final refit, train.py:256, tworzy nowy model z tym samym seedem i uczy przez wybraną liczbę epok. Zapisuje checkpoint, A, G, scaler, hashe i FREEZE przed wywołaniem testowego readera. `evaluate(origin=90)`, train.py:121, dodatkowo wymaga istniejącego FREEZE i zgodnego SHA modelu, zanim `ensure_window` otworzy przyszłe cele.
- Test dotyczy cp90→100, czyli100 aktualizacji ViT. Nie jest testem500 aktualizacji, nowych obrazów ani nowego niezależnego biegu. Wygenerowany raport wyraźnie stwierdza, że źródłowa historia była już eksplorowana wcześniejszymi wariantami.

### Poprawiony rzeczywisty błąd

W pierwszej wersji ścieżki resume można było wznowić inner training po epoce z już spełnionym early stopping i wykonać kolejną epokę, ponieważ istniejący wynik walidacji był pomijany bez ponownego sprawdzenia reguły zatrzymania. Zgłosiłem to wykonawcy przed jobem. W finalnym kodzie train.py:244 guard `early_stop(val_rows,...)` działa na początku pętli, przed pomijaniem zapisanej walidacji. **Status: poprawione i ponownie odczytane.**

Checkpoint po całym originie przechowuje cursor, model i stan Adam. Niedokończony origin jest po wznowieniu powtarzany od ostatniego kompletnego zapisu. Nie należy sumować bez deduplikacji historycznych logów prób, które zostały cofnięte do checkpointu; autorytatywny licznik stanu modelu jest zapisany w checkpointcie/freeze.

### Metryki i niezależna kontrola

`Metrics.add/finish`, train.py:45, sumuje błędy po wszystkich tokenach i dzieli przez rzeczywistą liczbę tokenów oraz kanałów. Ponieważ każdy obraz ma600 tokenów, odpowiada to równej wadze obrazów; ogon B=28 nie jest traktowany jak pełna paczka32. Aggregate RMSE jest pierwiastkiem po uśrednieniu kwadratów, a nie średnią RMSE paczek.

Niezależnie porównałem 120 wartości na syntetycznych, niesymetrycznych tablicach full768/J128 z ręcznymi obliczeniami NumPy: MSE, RMSE, błąd persystencji, kontynuacji ostatniej różnicy, historycznej średniej i znak. Maksymalna różnica wyniosła5,5511151e−17. Osobno sprawdzono kolejność osi zapisanego J128. W finalnym kodzie dodatkowo raportowane są znaki wszystkich trzech baseline; cosine i iloraz norm ujawniają udział pominiętych zerowych wektorów, podczas gdy znak obejmuje wszystkie współrzędne bez odsiewu.

Baselines to persystencja, liniowa kontynuacja ostatniego przyrostu oraz stała średnia trzech ostatnich dostępnych stanów. Wszystkie używają tej samej przestrzeni i originu. Błąd prognozy stanu i błąd skumulowanego przemieszczenia są algebraicznie równe, jeżeli w obu odejmuje się ten sam rzeczywisty stan originu; nie są dwiema niezależnymi miarami.

Dla późniejszego niezależnego audytu CPU zapisano w config z góry cztery obrazy zgodne z istniejącymi sondami:439,734,1471,1215. `FORECAST_FULL768_AUDIT_PROBES.npz` ma zawierać ich pełne prognozy768 oraz H0 reconstruction. Prawdziwe cele są już dostępne w lokalnym dense history, więc audyt nie wymaga ponownego GPU ani pełnego dumpu1500. `TEST_PER_IMAGE.jsonl` obejmie wszystkie1500; główne metryki również obejmują wszystkie1500. Tych artefaktów wynikowych jeszcze nie zweryfikowano, bo trening nie został zakończony.

### Hashe zakresu sprawdzonego przed jobem

| Plik | SHA256 |
|---|---|
| train.py | `ebed1b9dce4a9d73082789034fcfdf81d856ce857c84f0d29ca5a32aaacb3e9a` |
| model.py | `733a4dfa4b4dde3253a126fd5dda279725fe1c2b321cc5b172ab02e966d36f2d` |
| state.py | `03ed6aa3ac63ba42f91d56a4d63826ffc1ec2d7b7504386aeed0788c279fbd99` |
| CONFIG.json | `c748028121ca0710a421a168e5fbb38130aaeb2eb91c65dc432ad11f75f32dd8` |
| PLAN_MODELU.md | `e2c03290efdae6251a8c02c3b1b88ec9f3314201ca0b4b9906bd2790ad9d61ad` |

Pliki wykonania zachowują własny snapshot źródeł i sprawdzają niezmienność hashy przed FREEZE. Interpretacja pozostaje ograniczona: RMS telemetrii to stratne charakterystyki, learned768 nie jest natywną bazą768, regularizacja A−I nie gwarantuje stabilności, a korelacje900 tysięcy tokenów nie dostarczają900 tysięcy niezależnych trajektorii uczenia.

Po ukończeniu obliczeń pozostaje niezależne sprawdzenie rzeczywistych liczników, zakresów czasu, freeze/hash, zapisanych prognoz i agregacji metryk oraz jawne odnotowanie ewentualnego wyniku negatywnego. Bez zmiany wybranego modelu po zobaczeniu testu.
