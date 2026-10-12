# Niezależny audyt NNDMD: 1500 obrazów, batch 32 obrazy, okna 10

Status: **w trakcie — sprawdzono dane/projekt cache i rdzeń gradientu; pełny protokół, trening oraz test pozostają do rozliczenia**. Cutoff pierwszego odczytu: 2026-10-02, około 14:33–14:38, Europe/Warsaw. Autor: Astra. Zmiany wyłącznie w tym audycie; bez zmian kodu/danych innych agentów i bez użycia GPU.

**Zmiana autoryzacji około 14:40:** użytkownik polecił zatrzymać dotychczasowe lokalne przebiegi i przygotować rozszerzony stan NN do treningu na Edenie. Poniższy pierwszy przegląd dotyczy początkowego rdzenia 1500; nie jest zgodą audytową na nieodczytany jeszcze rozszerzony model. Nowy zakres i wymogi przyczynowości opisano na końcu. Aktualna intencja: full latent training, J128 wyłącznie jako dodatkowa ocena.

Na tym etapie nie ma wyniku predykcyjnego nowego eksperymentu. Nie należy przypisywać mu wyników poprzedniego NNDMD ani twierdzić, że osiągnął 85 procent znaku. Wczesna kontrola nie wykazała błędu w zasadniczym ważeniu mikrobatchy i pochodnej operatora. Otwarta była rozbieżność opisu straty J128 oraz potrzeba mocniejszego sprawdzania ponownie używanego cache.

## Zakres i poprzednie ustalenia

Odczytano [PLAN_DANYCH.md](PLAN_DANYCH.md), [MANIFEST_1500.json](MANIFEST_1500.json), [przygotuj_manifest.py](przygotuj_manifest.py), [dane.py](dane.py), [model.py](model.py), [test_model.py](test_model.py) i istniejący log testów. Stosuję wcześniej wczytaną procedurę `audit-project-correctness`, z jawnym ograniczeniem zapisu do tego pliku. Nie tworzę rejestru innego projektu.

Poprzednik: [audyt NNDMD na 34 obrazach](../maly_vit_pykoopman_30_obrazow_128/AUDYT_NNDMD_ASTRA.md). Tamten wynik pozostaje ujemny i ważny jako wynik tamtej architektury. W nowym modelu decoder jest nieliniowy, więc szczególne ograniczenie kwadratowego liniowego C z poprzedniego audytu już nie opisuje nowej konstrukcji. Nadal nie ma dowodu injektywności encodera, pełnej rekonstrukcji ani obserwowalności ukrytego stanu uczenia ViT. Problem identyfikacji kodu wykonania należy zapobiegawczo rozliczyć przez snapshoty/hash przed treningiem; późniejsze SHA pliku na dysku nie zastępuje wersji załadowanej wcześniej do procesu.

## Dane i jednostka czasu

Manifest ma dokładnie 1500 rekordów w kolejności indeksów 0..1499, 1500 unikalnych frame-ID i 101 checkpointów k0..k1000 co10. Kod przygotowania porównuje każdy obraz źródłowy po RGB i resize INTER_AREA do `baza1500.npy`; wszystkie rekordy mają potwierdzoną zgodność pikseli. Audytor niezależnie przeliczył strukturę manifestu, unikalność, komplet indeksów i kroków; nie powtarzał odczytu wszystkich źródłowych obrazów.

Obraz pierwszy: `TR_0001_S1_P03/00000076`, ostatni: `TR_0170_S1_P03/00001941`. Cztery dokładne dopasowania do dawnych sond mają indeksy źródłowe 1/4/7/8. To nie jest 1500 nowych niezależnych realizacji treningu: wszystkie obrazy są mierzone podczas tego samego biegu zmian wag i BN.

**Brak segmentacji jest potwierdzony.** Jest 1500 anotacji bbox XYWH i zero rekordów z segmentacją. Prostokąta nie wolno nazywać prawdziwą maską narzędzia. Primary loss po wszystkich 600 tokenach z wagą 1/600 obejmuje również tło. Ewentualne bbox-ROI jest osobnym podzbiorem geometrycznym, z odpowiednią etykietą. Wynik na wszystkich tokenach nie jest wprost porównywalny z dawną oceną 60 tokenów z masek narzędzia.

Indeks checkpointu 90 oznacza aktualizację 900, a 100 oznacza 1000. Przy oknie 10 przejść potrzebnych jest 11 stanów. Poprawne granice:

- wewnętrzny fit: stany 0..80, początki 0..70, cele najwyżej 80;
- walidacja: origin 80, cele 81..90, bez gradientu i dopasowania statystyk z tych celów;
- refit po wyborze: stany 0..90, początki 0..80, cele najwyżej 90;
- końcowy test: origin 90, cele 91..100, po zamrożeniu modelu.

J128 historycznie wybrano na k500, więc jest dostępne przed originem k800/k900. To nie usuwa wcześniejszej ekspozycji tych historii i czterech dawnych sond. Nowy test jest prognozą przyszłości tych samych znanych 1500 obrazów, a nie holdoutem obrazów ani nowym niezależnym biegiem ViT.

## Streaming i rotacyjny cache

Pełne `[101,1500,600,768]` float32 wymaga 279 244 800 000 bajtów przed nagłówkami. Jedenaście slotów checkpointowych wymaga 30 412 800 000 bajtów. API [dane.py](dane.py), wiersze 19 i 65–75, mapuje checkpoint `cp` do `cp mod 11`; dla każdego kolejnego okna 11 stanów indeksy slotów są różne. Przesunięcie okna o jeden zastępuje wyłącznie najstarszy zbędny slot. `get_batch` zwraca skopiowane tablice, nie trwałe referencje do nadpisywanego mmap.

Źródło jest w eval, z wyłączonym gradientem, TF32 i aktualizacjami BN. Przed budową sprawdzany jest hash checkpointu oraz `krok`; po budowie skończoność, 1500 obrazów, niezmienność buforów, hash pliku i metadane. Stan `building` jest zapisany przed nadpisaniem slotu. `complete` powstaje dopiero po zakończeniu. To poprawnie chroni przed uznaniem własnego przerwanego zapisu za kompletny. Źródłowy ViT jest przenoszony na CPU po budowie.

**A-01 — P2, potwierdzona luka walidacji cache, otwarte przy pierwszym odczycie.** `_valid` w wierszach 20–23 sprawdzało tylko status, indeks i hash manifestu. Nie sprawdzało nagłówka/dtype/kształtu/liczby bajtów pliku NPY ani zgodności checkpoint SHA z rekordem. Zgłoszono wykonawcy: tanie kontrole nagłówka i rozmiaru oraz hash treści przy ponownym użyciu slotu w nowym procesie; nie hashować 2,76 GB przy każdym minibatchu. Hash manifestu 1,57 MB można bezpiecznie przechowywać w instancji po jego ustaleniu, zamiast wielokrotnie czytać go dla wszystkich 11 slotów.

Cache wymaga jednego writera i serializacji `ensure_window` z odczytem okna. Równoległy prefetch innego originu mógłby nadpisać potrzebny slot. Plan przewiduje jeden proces i harmonogram window-major; docelowy driver musi tego rzeczywiście przestrzegać. Zmiana epoki może wymagać ponownej ekstrakcji całego prefiksu; 11 slotów nie jest pełnym cache 101 checkpointów. Benchmark źródła musi uwzględniać osobno transfer, zapis i hashowanie, a ETA treningu również tę powtarzaną pracę.

## Model i gradient jednego batcha obrazów

[model.py](model.py), wiersze 16–41, używa faktycznego rdzenia `pykoopman.regression.NNDMD`: nieliniowych sieci pakietu i `matrix_exp(G)`. W nowej wersji encoder i decoder mają SiLU, wejście/stan/wyjście po 768. Jawny wrapper zastępuje trening na materializowanej liście przez streaming. `fit()` domyślnego pakietu jest celowo wyłączony, ponieważ jego końcowe wyznaczanie liniowego C nie jest poprawne dla nieliniowego decoderu. Nie ma fikcyjnej macierzy C ani deklaracji, że oryginalne API `.fit()` wykonuje tę nową pętlę. Brak PCA.

Batch 32 obrazów zawiera 32×600 cząsteczek. Przy tym modelu każda cząsteczka nadal ma własny rollout w shared learned space; grupowanie obrazów wyznacza gradient, ale samo nie wprowadza oddziaływań między 600 tokenami. Operator A jest wspólny, nie jest średnią operatorów obrazów.

Dla rzeczywistego B obrazów, P=600 tokenów, D=768 i H=10 część prognostyczna jest średnią `sum squared_error/(B*P*D*H)`. Rekonstrukcja początku ma mianownik BPD; rekonstrukcja przyszłych stanów ma mianownik BPDH. Wszystkie obrazy, tokeny i horyzonty mają równe masy zgodnie z celem all600. Ostatni batch ma B=28, a nie sztucznie dopisane 32. Istotne jest rzeczywiste B w mianowniku.

Mikrochunk w wierszach 72–92 sumuje błędy i dzieli przez pełny mianownik batcha. Nie uśrednia równoważnie chunków różnej długości. Gradient względem odłączonego A_leaf sumuje się przez wszystkie chunki; następnie `sum(A_graph * grad_A_leaf.detach())` przenosi całą pochodną przez exp(G) jeden raz. Jest to prawidłowa reguła łańcuchowa. Ta sztuczna funkcja służy przeniesieniu gradientu, nie jest raportowaną wartością straty. Gradienty encodera i decoderu akumulują się zwykłym backward. Clipping i optimizer.step są wykonywane dopiero po pełnym logicznym batchu (`:106–114`).

**Niezależna kontrola CPU, wykonana przez audytora:** oddzielnie zapisane równanie referencyjne całego batcha, bez użycia helpera `direct_loss`, porównano z mikroakumulacją. Nieliniowe E/D, H=10, D=4, hidden7, skalar normalizacji 2,3, waga rekonstrukcji 0,7 i kara operatora 0,03. Dwa przypadki: B3/P5/chunk6 (ogon3) i B28/P2/chunk17 (ogon5). Maksymalna różnica wartości loss: 1,70×10⁻⁸; maksymalna różnica gradientu dowolnego parametru: 3,73×10⁻⁹; gradient G: najwyżej 1,40×10⁻⁹. Wszystkie parametry zgodne przy atol1e−6/rtol2e−5. Sama akumulacja wykonała zero kroków optymalizatora. Test dotyczy małego kontrolowanego układu CPU; nie jest testem wydajności modelu 768D ani treningiem ViT.

Istniejący [test_model.py](test_model.py) i log wykonawcy dodatkowo obejmują jeden krok Adam po batchu oraz encode-once rollout. Audytor przeczytał te testy i artefakt; nie traktuje ich jako wyniku nowego treningu produkcyjnego.

## Otwarte uzgodnienia przed fitami

**A-02 — P1, potwierdzona rozbieżność plan/kod przy pierwszym odczycie.** PLAN_DANYCH opisuje `full768 + beta J128`, natomiast obecne `direct_loss` i `microbatch_gradients` liczą full768 i rekonstrukcję, bez dodatkowego składnika J128. Zgłoszono rootowi i wykonawcy. Trzeba określić przed fitami, czy J128 jest tylko końcową oceną, czy dodatkową wagą loss, i odpowiednio ujednolicić plan/kod. Nie wolno raportować użycia beta, którego funkcja nie liczy.

**A-03 — P2, potwierdzone ograniczenie nazwy stabilizacji.** `stability_weight * ||A−I||_F²/D` jest karą bliskości identyczności, nie wymusza rho(A)≤1 i nie gwarantuje stabilnego rollout. Jest matematycznie poprawnie dodawana do gradientu. W opisie należy nazwać ją regularizacją operatora i nie obiecywać stabilności na tej podstawie. Nieliniowy decoder również może wzmacniać błąd niezależnie od samego rho(A).

Normalizacja i pełna chronologia wyboru jeszcze nie były dostępne w driverze przy pierwszym odczycie. Wymóg: statystyki tylko z prefiksu fit danej fazy, stałe w walidacji i teście. Jeżeli trening minimalizuje normalized MSE, a raportuje raw RMSE, te dwa obiekty trzeba nazwać wprost. RMSE powstaje jako sqrt ważonego MSE po agregacji; średnia RMSE mikrochunków lub obrazów jest inną metryką. Wspólny skalar zachowuje proporcjonalność full raw MSE; skalowanie per kanał zmieniałoby wagi zadania.

## Do zamknięcia po udostępnieniu kolejnych artefaktów

Przeczytać ostateczny PLAN/driver, snapshot wykonania i zasady liczby epok, kolejność okien, losowanie obrazów, liczbę aktualizacji 47 batchy na origin, skalowanie, brak ruchomych statystyk oraz zamrożenie wyboru przed końcowymi celami 91..100. Zrealizowana epoka obejmująca wszystkie okna powinna mieć 71×47=3337 update'ów w fazie wewnętrznej i 81×47=3807 w reficie. Losowany podzbiór originów byłby odrębnym budżetem i musi być jawnie opisany.

Po treningu: niezależnie przeliczyć zapisane prognozy/MSE/znak wszystkich współrzędnych, równe wagi obrazów, finalny niepełny batch i podzbiór decoded J128. Sprawdzić rekonstrukcję, ruch/normę, baseline persystencji oraz zgodność odtworzenia A^h z zapisanym wynikiem. Brak teacher forcing: przyszłe rzeczywiste stany są celami loss lub późniejszej oceny, nie wejściami autonomicznego rollout. Bez wyboru nowego zwycięzcy po teście. Użyteczność MSE i cel 85 procent znaku pozostają odrębnymi warunkami.

## Aktualizacja: expanded state i Eden

Najnowsze zlecenie obejmuje pełne z768, historię jego przyrostów oraz historyczne cechy zmian parametrów i dostępnego AdamW. Learned state 768 i A768×768 mają pozostać, z nieliniowym decoderem. Rozszerzone wejście ma więcej niż 768 współrzędnych, więc encoder do learned768 stanowi teraz uczoną kompresję rozszerzonego stanu; nie należy deklarować zachowania całej informacji na podstawie samego wymiaru.

**A-02, aktualizacja intencji:** użytkownik wskazał trening pełnego latentu i J128 tylko do oceny. Obecny brak dodatkowej wagi J w rdzeniu jest zgodny z tą najnowszą intencją; wcześniejszy zapis betaJ w PLAN_DANYCH jest do usunięcia/oznaczenia jako zastąpiony. Nie należy dodawać betaJ wbrew nowemu zakresowi. To korekta zakresu przed nowym fitem, nie dobór po teście.

**A-01, aktualizacja implementacji:** ponowny odczyt `dane.py` wykazał dodanie kontroli metadanych shape/dtype/bytes, nagłówka NPY, checkpoint SHA i jednorazowego sprawdzenia SHA treści przy zmianie slotu. Hash manifestu jest zapamiętany w instancji. Poprawka jest **wdrożona; pełny ponowny odczyt rzeczywistego slotu 1500 nie został przez audytora uruchomiony**. Nie należy przypisywać tego sprawdzenia etapowi, którego jeszcze nie wykonano.

### Faktyczna dostępność optymalizatora

Lokalny zbiór 101 checkpointów `wagi` ma wyłącznie `net,krok`, co potwierdza dawny [raport pobrania](../maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/pobranie_wag_2026-10-01/POBRANIE_WAG_2026-10-01.md). **Nie oznacza to, że optymalizatora w ogóle nie zapisano.** Źródłowy [bieg.py](../maly_vit_ruch_czastek_gesta_historia/kod/bieg.py), wiersze 107–109, zapisuje `u.stan()` oddzielnie do `pelny/k*.pt`. [trening.py](../maly_vit_ruch_czastek_gesta_historia/kod/trening.py), wiersze 35–36, definiuje stan jako `net,opt,krok`. Metadata ukończonego biegu podaje `zapis_pelny=100` i listę 11 plików k0000,k0100,...,k1000. Są to momenty AdamW co 100 aktualizacji, a nie co każdy checkpoint co10. Ich bieżącą obecność i klucze na Edenie sprawdza właściciel danych; na tym cutoffie audytor potwierdził kod i historyczny manifest, nie odczytał zdalnych binariów.

Zmiana wag `theta_k−theta_(k−10)` jest sumaryczną zmianą z 10 aktualizacji. Nie jest pojedynczym gradientem i nie identyfikuje jednoznacznie m1/m2 AdamW. Wagi i bufory BN należy rozróżniać przy definiowaniu cech. W źródle AdamW ma beta1=0,9, beta2=0,999, LR docelowy 0,0005 i warmup 300; parametry weight decay są pogrupowane według wymiaru tensora. Znany harmonogram czasu/LR może być sygnałem deterministycznym; przyszłe rzeczywiste gradienty nim nie są.

### Warunki przyczynowego rozszerzenia

1. Stan na originie t może używać wyłącznie z i checkpointów parametrów z czasu ≤t. Lag Δz_t oznacza z_t−z_(t−1) w siatce checkpointów; brak historii na początku wymaga jawnego paddingu i maski dostępności albo jawnego odrzucenia wczesnych originów. Nie wolno wypełniać braków przyszłymi wartościami.
2. Przy rzadkim AdamW wolno użyć ostatniego snapshotu z czasu ≤t, wraz z wiekiem pomiaru i maską dostępności. Interpolacja momentu z k100 i k200 do k150 jest przeciekiem, bo używa k200. Statystyki scalerów obejmują tylko prefiks fit odpowiedniej fazy.
3. W rollout od k900 do k1000 żadna rzeczywista cecha theta/BN/AdamW z k910..k1000 nie jest wejściem. W szczególności moment AdamW k1000 nie może wejść w ostatnim kroku. Upływ czasu do planowanej daty snapshotu nie oznacza pojawienia się nowego pomiaru i nie resetuje automatycznie wieku dostępnej informacji.
4. Przyszłe auxilia muszą być prognozowane wspólnie albo przesuwane na podstawie dotychczasowych prognoz. Jeśli encoder uruchamia się raz i latent przechodzi przez A^h, decoder może równolegle odczytywać predicted z i predicted aux; żaden z tych odczytów nie pochodzi z przyszłego source. Jeśli projekt ponownie koduje zdekodowany stan z mechanicznym przesunięciem lagów, trzeba nazwać tę odrębną pętlę wprost, zamiast utożsamiać ją z czystym A^h.
5. Przyrosty wyprowadzane z przewidywanych z są automatycznie spójne z ruchem. Bezpośrednio przewidywane bloki historii mogą być niespójne (`predicted Δz != predicted z_next−predicted z_now`); trzeba to kontrolować lub narzucić. Nie wolno naprawiać niespójności przez dostęp do przyszłego prawdziwego z.
6. Primary target pozostaje z768. Dodatkowe bloki mają jawne scalery i wagi. Zwykłe MSE po konkatenacji wielu lagów zmieniłoby wagę błędu z zależnie od liczby lagów. Full latent loss, aux loss i RMSE w natywnych jednostkach muszą być rozdzielone w raporcie.
7. Te same ograniczenia obowiązują w walidacji origin80→90. Tylko targets ≤80 trafiają do wewnętrznego fitu; valfuture służy wyborowi. Po wyborze refit może używać prawdziwych danych do90. Test91..100 wyłącznie po freeze. Liczenie historycznych cech dla zbioru targetów nie daje prawa do ich użycia jako forecast input.

Przed złożeniem joba na Eden potrzebny jest odczyt rzeczywistego rozszerzonego kodu, manifestu cech/history oraz test z zatrutym przyszłym sufiksem. Samo przeniesienie na klaster nie rozstrzyga poprawności matematyki. Użytkownik autoryzował submit po weryfikacji; audytor nie uruchamia konkurencyjnego GPU ani nie składa drugiego joba.
