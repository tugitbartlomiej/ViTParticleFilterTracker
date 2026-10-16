# Wykonanie na Edenie i plan statystyczny

Ten dokument jest specyfikacją kolejki, nie gotowym skryptem SLURM. Nie wysłano zadań na Eden i nie zweryfikowano bieżącej dostępności GPU, limitu czasu ani kwoty dysku. Parametry wymagające pomiaru lub decyzji są jawnie niezamrożone w KAMPANIA.json. Nadrzędny dokument: [PROTOKOL.md](PROTOKOL.md); szczegółowe uzasadnienia: [audyt statystyczny](../../outputs/protokol_testow_vit_iclr_neurips_20261016/AUDYT_STATYSTYKA_EDEN.md).

## 0. Ustalony sposób pracy: obliczenia na Edenie, interpretacja lokalnie

**Preferencja użytkownika z 16 października 2026:** treningi i kosztowne pomiary wykonujemy domyślnie na Edenie. Po każdym etapie pobieramy kompletny pakiet wyników i oglądamy go lokalnie, w tym w mikroskopie tokenów. Zwykłe odczytanie wykresów, porównanie przebiegów i odtworzenie zapisanej trajektorii ma działać bez dostępu do Edena i bez lokalnego GPU. Interaktywne nowe interwencje wymagają osobnego obliczenia; istniejące wyniki nie mogą zależeć od ponownego forward.

Każdy test T00–T29 ma rozdzielone etapy: **compute → export → download/verify → analyze/view**. Compute jest bezobsługowym zadaniem uruchamianym z utrwalonej konfiguracji, bez GUI i notebooka wymagającego ręcznego klikania. Nowe pytania po przeglądzie discovery tworzą osobny wersjonowany bieg; w confirmation obowiązuje zamrożony plan.

### Pakiety po każdym etapie

| Pakiet | Obowiązkowa zawartość | Użycie lokalne |
|---|---|---|
| analysis | Surowe sparowane wyniki wszystkich zaplanowanych ramion, shams i dawek; identyfikatory seed/film/klatka/checkpoint/blok/token; GT, predykcje, statusy i powody braków; manifest, konfiguracja, snapshot kodu, środowisko, logi, koszt, agregaty i skrypty odczytu | Przeliczenie tabel, CI i wykresów bez ponownej inferencji. Nie wystarczają średnie ani PNG. |
| viewer | Trajektorie wszystkich 600 tokenów dla zadeklarowanych klatek i kroków; jawne punkty pomiaru 12 podkroków i wejścia; parametry projekcji, osie, jednostki i indeksy; obrazy referencyjne lub weryfikowalne odwołania do lokalnych kopii; GT, ramki, confidence, maska maksimów i ranking | Oś checkpointów, przejścia przez bloki i powiązanie ruchu z ramką. Projekcja ma być stała w obrębie porównania i opisana; nie traktować jej jako pełnej geometrii. |
| replay | Wszystkie zaplanowane checkpointy wag i BN, manifest kroków i hashy; pełne stany restartu na z góry wybranych punktach, obowiązkowo dla T19: optimizer, scheduler, RNG, sampler i dokładny batch/augmentacja | Zachowanie biegów do późniejszej kontroli. Nowe ciężkie pomiary można zlecić Edenowi bez ponownego treningu. |

Formaty proponowane do implementacji: JSON/JSONL dla manifestów, Parquet lub CSV dla tabel oraz porcjowane tablice NumPy/Zarr dla tensorów. Wybrać i zamrozić konkretny format oraz wersję schematu przed biegiem; czytnik Windows/Linux ma przejść próbę odczytu E0. Używać ścieżek względnych względem katalogu biegu oraz logicznych identyfikatorów danych, bez zależności od ścieżki /scratch lub litery dysku autora.

Nie pobieramy automatycznie wszystkich wielkich tensorów. Pakiet analysis pobieramy w całości; viewer i replay jako osobne, wznawialne części. **Dokładną próbkę do odtwarzania pełnego mikroskopu wybieramy przed obliczeniami.** Dla tej próbki zapisujemy stan residualu, aktywacje po GELU i wagi potrzebne do odtworzenia reflektorów, podpisanych wkładów oraz pól dla przewidzianych widoków. Dla pozostałych klatek lekki cache może zawierać tylko projekcję i metryki; aplikacja musi pokazywać zakres dostępności, zamiast obiecywać pełny widok. Dokładne lokalne J zapisujemy na zaplanowanej próbce pomiarowej; brak macierzy oznacza brak możliwości dowolnego nowego przekroju bez obliczeń.

Cache służący wizualizacji może mieć osobną kompresję/precyzję, ale surowe liczby do naukowej analizy zachowują zwalidowaną precyzję. Nie duplikować całego datasetu w każdym biegu; manifest wskazuje wymagane obrazy i ich hashe, a eksport ma jawnie wykrywać niedostępne lokalne źródła. Zakres viewer, checkpointów i pełnych stanów restartu musi zmieścić się w zmierzonym budżecie dysku i transferu; nie usuwać ich po cichu z powodu kwoty.

Osie cache viewer to seed, image_id, checkpoint_step, substep 0–12, token_id 0–599 oraz xyz. Zapisać mapowanie token–patch, RGB po preprocessing lub jego bezstratny odpowiednik, transformację współrzędnych obrazu, bazę/środek/skalę projekcji i dane użyte do jej dopasowania. Kolory i legendy muszą mieć zapisane definicje. Widoki clean/perturbed/target/sham mają osobne identyfikatory. Dla attention jawnie podać dostępność wszystkich połączeń, wybranego top-k lub tylko wektorów sumarycznych; brak zapisanego połączenia nie oznacza zerowego wpływu.

**Gęsty film uczenia ma osobny budżet.** Dwanaście checkpointów atlasu to dwanaście obserwacji, niezależnie od płynności animacji. Przed treningiem wybieramy mały stały zestaw dense_viewer_probes i gęstszą siatkę cache, wstępnie co 100 kroków, do potwierdzenia w E0. Taki cache można zbierać podczas treningu bez zapisywania pełnego stanu optymalizatora co 100 kroków; tryb eval i zapis nie mogą zmieniać RNG ani buforów treningu, co wymaga kontroli. Film rozróżnia czas treningu od 13 stanów forward; interpolację oznacza jawnie. Projekcję dopasowaną z użyciem przyszłych checkpointów oznaczamy jako retrospektywną i wykluczamy z predykcyjnego T25. Same pozycje xyz atlasu 250×12×13×600 w float32 zajmują około 281 MB na seed; pełne aktywacje wiązek wymagają osobnej wyceny.

### Dostarczenie i sprawdzenie biegu

1. Każdy ukończony shard jest atomowo zamykany z liczbą rekordów, rozmiarem i SHA-256. Eksport zawiera indeks oczekiwanych i wykonanych pomiarów, także błędów i wyników ujemnych.
2. Pakiet jest dzielony na części możliwe do wznowienia; pliki częściowe nie są oznaczane jako ukończone. Wybór narzędzia transferu zależy od rzeczywiście dostępnego dostępu do Edena, bez zakładania działającej konfiguracji SSH.
3. Po pobraniu lokalny weryfikator sprawdza hashe, schemat, liczbę rekordów, parowanie ramion, pokrycie checkpointów i próbek. Braki wymagają uzupełnienia lub jawnego statusu, nie są pomijane w raporcie.
4. Lokalny etap odtwarza wcześniej wybrane tabele/wykresy z surowych rekordów i otwiera próbkę trajektorii bez połączenia z Edenem. Ten test całej ścieżki jest warunkiem zaliczenia E0 przed dużymi biegami.
5. Kopie na Edenie zachowujemy do potwierdzenia pobrania i sprawdzenia lokalnej kopii. Dalsza retencja jest jawna; projekt nie zawiera automatycznego kasowania biegów.

Katalog lokalny: `outputs/<campaign_id>/<run_id>/`, z podkatalogami analysis, viewer, replay i manifests. Snapshot raportu i jego interpretacja mają własną wersję oraz wskazują hashe źródeł. Import danych do obecnej aplikacji wymaga osobnego adaptera i sprawdzenia zgodności; niniejszy plan nie oznacza, że aplikacja już obsługuje nowy format.

## 1. Kolejność, która ogranicza koszt nietrafionej hipotezy

| Etap | Zadania | Dane / obliczenia | Bramka |
|---|---|---|---|
| A: poprawność E0 | T00–T05, T10, T28 | s47; 3 checkpointy, 4 obrazy; referencja float64 i pomiar kosztu | Zgodny forward/dekoder, pochodne, split i reguły perturbacji |
| B: pilot | T06–T09, T11, T15–T16, T20, T26 | 24 nowe klatki z 12 VAL; najpierw mały T09 | Efekt przekracza błąd numeryczny i ma sens względem sham, dawki i dekodera |
| C: discovery | T21 część rozwojowa; atlas; wybrane T12–T19; T27 | s47 + 2 nowe treningi; 250 VAL; symulacja precyzji | Wybrana jedna historia, oszacowany koszt i wariancja, zamrożony manifest |
| D: confirmation | T09/T11/T15/T16/T20/T21/T26/T27; atlas według claimu | 3 nowe treningi jako budżet początkowy; 250 TEST + pełny ustalony TEST dla metryk detektora | Protokół zablokowany przed wynikami; raport także wyników ujemnych |
| E: zakres | T22/T23, ewentualnie T24/T25; T29 | Drugi model, drugi detection dataset i odtworzenie z czystego środowiska | Claim zawężony do rzeczywiście sprawdzonego zakresu |

Nie trzeba wykonywać wszystkich rozszerzeń w jednym artykule. Aby pisać ogólnie o mechanizmie ViT, druga architektura jest częścią dowodu; dla ogólnej lokalizacji potrzebny jest drugi zbiór detekcyjny. Sama klasyfikacja CIFAR nie potwierdza mechanizmu bbox. T19 jest potrzebny do wyjaśnienia przyczyny zmian treningowych, nie do samego opisania wyuczonego forward.

Jeśli pilot jest nierozstrzygający, sprawdzamy precyzję. Jeśli przewidziany efekt znika względem poprawnych kontroli, wracamy do wyboru hipotezy na discovery. Nie odczytujemy TEST, żeby wybrać wygodniejszą historię.

## 2. Seedy, filmy i checkpointy

Potwierdzony w dokumentacji podział roboczy to **117 TRAIN / 25 VAL / 25 TEST grup operacji**. Przed użyciem musi przejść T00. Nie oznacza automatycznie 167 różnych pacjentów. W razie wspólnych pacjentów rozdział i resampling przesuwamy na ich poziom. Rejestr wcześniejszego użycia TEST rozstrzyga, czy pozostaje odłożonym testem; nowy seed nie usuwa wcześniejszego strojenia na tych samych filmach.

- s47 jest historycznym discovery.
- Propozycja nowych seedów discovery: 1047 i 2047.
- Propozycja confirmation: 3047, 4047 i 5047.
- Sprawdzić kolizje rejestru; inicjalizacja, kolejność danych i augmentacje muszą wynikać z zapisanych RNG.
- Docelowe N confirmation wybieramy po symulacji na discovery, **przed** odczytaniem confirmation; rozważana siatka to 3/5/8. Nie wolno wliczyć dwóch seedów użytych do wyboru metody jako dwóch dodatkowych ślepych replik.

| Zakres | Klatki | Checkpointy | Dokładne Jacobiany MLP |
|---|---:|---|---:|
| E0 | 4 | 0, 33000, 65640 | 4×3×6×12 = 864 |
| Pilot atlasu | 24 z 12 VAL | 0, 1000, 5000, 15000, 33000, 55000, 65640 | 24×7×6×12 = 12 096 |
| Atlas na seed | 250, po 10/film | 0, 300, 1000, 3000, 6000, 10000, 16000, 24000, 33000, 44000, 55000, 65640 | 250×12×6×24 = 432 000 |

T09 zaczyna od 3 stanów [0,33000,65640], bloków 1/3/6 i 24 obrazów; nie musi wykonywać pełnego iloczynu atlasu i wszystkich interwencji. W kodzie bloki są indeksowane 0–5, w tekście 1–6. Manifest przechowuje oba znaczenia jawnie.

Aktualizacja po drugiej recenzji: źródłowe J dla podstawowego T09 dotyczą tylko j=i0, czyli 216 J w powyższym pilocie. Reguła i0 jest stała wewnątrz ramion seed×checkpoint×obrazu, nie między wszystkimi treningami. Dla P2 dodatkowo zamrażamy wspólną kotwicę GT i populację obrazów. Pełny atlas 432 000 J/seed jest wariantem do claimu o całej głębokości i czasie; confirmation węższego mechanizmu może ograniczyć kosztowne pomiary do wybranych bloków oraz init/late. Nie redukujemy zamiast tego niezależnych filmów, seedów i kontroli sham.

Budżet treningów dzielimy na podstawowy (s47 + 2 nowe discovery + początkowo 3 nowe confirmation) oraz warunkowy T20 (random-label/loss ablations). Drugi ma własny manifest seedów, długość okna i dopasowaną kontrolę treningu właściwego, wybierane przed startem tej części. Krótkiego null-treningu nie porównujemy jako równoważnej kontroli pełnego końcowego modelu. Pełna cena kampanii obejmuje obie części, eksport i retencję.

Dla s47 wybór zastępczego checkpointu: najbliższy rzeczywisty krok, remis → wcześniejszy, różnica najwyżej 50 aktualizacji. Poza zakresem status missing_checkpoint; decyzja o zmianie siatki przed analizą. Nowe treningi zapisują dokładną siatkę. Porównanie między datasetami wymaga także ekspozycji danych, nie wyłącznie tego samego numeru epoki.

Klatki wybieramy przed inferencją, warstwowo w czasie i z zapisem brakujących etykiet. Tokeny atlasu: po 8 z interior/boundary/far-background, jeśli dostępne; żadnego powielania dla uzupełnienia budżetu. Bbox jest przybliżeniem lokalizacji, nie maską narzędzia. Agregaty tanich wkładów obejmują wszystkie 600 tokenów; kosztowne pełne J dotyczą ustalonej próbki.

## 3. Co dokładnie benchmarkować

W tym modelu residual ma 192 wymiary, ukryte MLP 768, obraz 600 tokenów, a iloraz symetrii translacji LN 191. Cache GUI w float16 i projekcja 3D nie są źródłem do obliczania dokładnych pochodnych.

- Forward i cache 12 podkroków; pełny suffix po interwencji; noising/denoising.
- JVP/VJP pełnego attention i downstream, J192, rozkład SVD/eigen, koszt sham i zapisu.
- Rozgrzewka, co najmniej 3 pomiary; mediana, rozrzut, peak VRAM/RAM, czas I/O.
- Float32 na kampanię dopiero po porównaniu z float64; jawne TF32/autocast i deterministyczność.
- Attention: start 16 sond; zbieżność na 8/16/32. Sham: start 8 orientacji; zbieżność na 8/16/32.
- Wszystkie kwoty i GPU-godziny obejmują również null-treningi, nieudane próby i piloty.

**Nie materializujemy pełnego Jacobianu 115200×115200:** jeden float32 wymaga około 53,08 GB samych elementów. Dokładne J MLP całego atlasu to około 63,70 GB na seed. Cache 13 stanów residualu dla 250×12 obrazów to około 17,97 GB na seed. To wartości dziesiętne bez narzutu, gradientów i kopii. Liczyć agregaty strumieniowo, a macierze i pełne aktywacje zachować dla małej audytowej podpróby.

Estymacja kosztu: liczba operacji każdego typu × zmierzony czas + trening + I/O + margines restartu. Przed E0 nie deklarujemy „zajmie X godzin”. Benchmark ma wybrać rozmiar sharda mieszczący się w rzeczywistym GPU i limicie czasu.

## 4. Pierwotny endpoint i dwa dekodery analityczne

P1 mierzy RMS czterech krawędzi podzielony przez max(przekątna GT,d_min), przy **stałym kandydacie wybranym na czystym oryginalnym forward**. Kontrast ma znak:

`D = błąd po interwencji ukierunkowanej − średni błąd po sham dopasowanym normą faktycznej poprawki`.

Dodatni D oznacza większą szkodę po interwencji ukierunkowanej. Sham zachowujący widmo operatora stanowi oddzielną kontrolę wtórną; nie mieszamy obu rodzajów w jednej średniej.

Właściwa kampania zawsze raportuje też maskę maksimów, top-k, pełny decoder i IoU. Brak detekcji daje brak recall/IoU=0 według ustalonego przypisania, a nie wyłączenie klatki. Standardowe globalne AP liczymy z predykcji na ustalonym zbiorze; średnia AP po filmach nie jest jego zamiennikiem.

**Kotwica i geometria:** bbox z dodatnich exp(ltrb) zawsze zawiera kotwicę tokenu. Dla głównego fixed-candidate P1 preferujemy rodzinę nuisance (jasność albo kontrast; wybór zamrozić). Dla geometrycznego P3:
1. użyć z góry ustalonego małego zakresu transformacji, w którym ta sama kotwica leży wewnątrz obu GT z marginesem; reguła kwalifikacji zależy od clean kandydata i GT, nie wyników interwencji;
2. albo zdefiniować odrębny endpoint z deterministycznie przekształconą kotwicą i jawnym błędem kwantyzacji siatki.

Przypadki poza wsparciem głowy zachowujemy w ocenie pełnego dekodera i raportujemy ich liczbę. Nie nazywamy ich dowodem braku ekwiwariancji MLP. W podstawowym P1 nie zmieniamy po cichu indeksu kandydata.

P2 porównuje P1 późnego modelu z inicjalizacją. P3 w głównym protokole oznacza selektywność względem nuisance/geometrii; pomocniczy audyt używał roboczo P3 także dla predykcji jakości. **Tutaj predykcja jakości to wyłącznie Q1 w opcjonalnym T25**, z osobno zamrożonym zdarzeniem i metryką. Nie łączymy dwóch znaczeń.

## 5. Niepewność i moc

Dla każdej pary seed×film redukujemy wyniki do zamrożonego kontrastu D[s,v]. Zachowujemy sparowanie obrazów, perturbacji, checkpointów i sham. Główna średnia równo waży filmy, a następnie seedy.

Bootstrap jest **skrzyżowany**: w każdej replice osobno losujemy wagi seedów i wagi filmów, a ich iloczyn stosujemy do całych sparowanych trajektorii. Nie losujemy innych filmów niezależnie wewnątrz każdego seeda. Przy znanych pacjentach używamy pacjenta i filmów w nim. Jeżeli bootstrapujemy klatki, ich wybór w danym filmie jest wspólny dla wszystkich seedów/ramion.

Trzy seedy nie gwarantują wiarygodnego wniosku o populacji treningów. Obowiązkowe wyniki per seed, leave-one-seed-out, CI warunkowe po filmach oraz osobno CI skrzyżowane. 10 000 replik bootstrapu poprawia dokładność Monte Carlo; nie tworzy dodatkowych niezależnych modeli.

T27 przed lock:
- Oszacować wariancję seed/film/interakcja na discovery, z szeroką analizą wrażliwości.
- Ustalić praktyczne δ_min w jednostkach endpointu.
- Symulować 3/5/8 seedów, 12/25 filmów i 2/10 klatek, wartości efektu 0, ±0,5δ_min, ±δ_min, ±2δ_min.
- Co najmniej 2000 symulowanych kampanii na kluczowy scenariusz; sprawdzić bias, pokrycie CI, FWER, moc, ciężkie ogony i brakujące pomiary.
- Dobrać N przed confirmation do celu precyzji albo ograniczyć zakres wniosku. Nie dopisywać seedów do osiągnięcia p<0,05.

Jeden pierwotny test jest najczytelniejszy. Przy kilku równorzędnych P używamy uprzednio określonej rodziny i Holm albo jednoczesnych CI, po sprawdzeniu kalibracji. Dla map eksploracyjnych FDR z jawną etykietą discovery. Brak istotności nie oznacza równoważności; ta wymaga marginesu i odpowiedniej precyzji.

Lock obejmuje również kartę przewidywań mechanizmu, selektor tokenu, mediatorów i donorów, budżet podmian, warstwę odłożoną i agregację. Prognozy mechanizmu są zapisywane przed odczytem odpowiedzi interwencyjnej. Oglądanie pobranego confirmation nie upoważnia do ponownego wyboru tych elementów dla głównego wyniku. Dla prerejestrowanego znaku s i progu δ* reguła CI dla sP1 rozdziela wsparcie wielkości (dolna granica > δ*), odrzucenie deklarowanej wielkości (górna < δ*) i nierozstrzygnięcie.

Rzadkie checkpointy nie wystarczają do widma „drgań”. Zapisujemy rzeczywisty Δstep i nie resamplujemy swobodnie niestacjonarnego treningu. Cechy predykcyjne T25 nie mogą używać przyszłych checkpointów, GT do wyboru tokenów ani PCA dopasowanej do całej przyszłej trajektorii.

## 6. Kontrakt jobów i kompletność

DAG logiczny:
`manifest → forward/derivatives → benchmark → pilot → discovery → lock → confirmation → replication → figures/reproduce`.

Każdy shard identyfikuje campaign/config hash, kod, seed, checkpoint hash, obraz/film, blok/podkrok, tokeny, perturbację, operator, α, rodzaj i RNG sham, dtype, decoder, GT assignment, wersję analizy i środowisko. Statusy: pending/running/complete/failed/missing; ukończenie wymaga atomowego zapisu, liczby rekordów i sumy kontrolnej. Resume pomija wyłącznie zgodny, kompletny shard.

Pliki wynikowe trafiają do nowego katalogu outputs wybranego dla wykonania; nie nadpisują tego protokołu ani historycznych wyników. Surowe wyniki rozdzielone od agregatów i figur. Null, NaN, OOM, brak checkpointu i brak dopasowanego sham mają różne kody; zero jest wartością, nie znacznikiem braku.

Reguły tolerancji sham oraz próg małych norm wybieramy przed confirmation. Jeśli zabraknie dopasowanej kontroli, raportujemy coverage i status; nie przełączamy niejawnie na słabo dopasowany wariant.

Preflight Eden: działający dostęp, wersje środowiska, GPU/VRAM, quota, limit czasu, lokalizacja datasetu, dozwolona równoległość, logi i mechanizm restartu. Przechowywać tylko releasable metadane w publicznym pakiecie; prywatne obrazy operacyjne wymagają uprawnień źródłowego zbioru.

## 7. Co już istnieje i co trzeba zaimplementować

| Zasób | Co wykorzystać | Co dopisać / zweryfikować |
|---|---|---|
| model.py / trening35000.py / dane.py w eksperymencie treningu s47 | Referencyjny forward, strata, decoder, dane | Manifest, nowe seedy, pełny restart RNG/optimizera/BN |
| trening_pokretla.py | Parametry seed/lr/kontrole | Retencja i telemetria tej kampanii |
| optyka_hipoteza_20261015/ekstrakcja.py | Sumy wkładów ostatniego MLP float32 | Wszystkie bloki, pełne attention, pochodne |
| wspoldzialanie_neuronow_20261015/pomiar.py i grupy.py | Ablacje/grupy, znane kontrole | Pełny suffix, fixed-candidate vs decoder, frozen selectors |
| kontrola_przelaczenia.py / weryfikacja.py | Sprawdzenie znanego switcha | Systematyczna ocena całej populacji |
| eden/queue.py / runner.py / run_node.slurm | Wzorzec infrastruktury | Nowy kontrakt, walidacja, shardy, brak zgodności „z automatu” |
| aplikacja laserów | Wizualne pokazanie wybranych wyników | Nie używać cache 3D/float16 jako dokładnego instrumentu |

Pierwsza paczka implementacji: **T00–T05/T10/T28 + mały T09**. Jej produktami mają być forward_audit, derivative_audit, numerical benchmark, manifest i pilot target/sham. Dopiero ich poprawność uzasadnia pełny atlas i nowe długie treningi.

