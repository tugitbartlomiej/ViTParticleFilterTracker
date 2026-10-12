# Audyt NNDMD: dwie karty, wspólna paczka obrazów

Data: 2026-10-02, 16:58, Europe/Warsaw. Audytor: Astra.

**Werdykt przed uruchomieniem: PASS. Nie wykryto błędu blokującego zgłoszenie jednego pełnego zadania na dwóch A100 z obowiązkowymi kontrolami przed fit.** Wynik nie oznacza ukończenia treningu ani potwierdzenia jakości prognoz. Właściciel zgłoszeń otrzymał werdykt przed redakcją tego pliku.

Audyt jest odrębny od [wcześniejszego audytu jednej karty](AUDYT.md). Obejmuje aktualne model.py, train.py, dane.py, state.py, przygotuj_eden.py, testy distributed/reader, protokół i launcher. Nie zmieniałem implementacji wykonawców i nie używałem GPU. Odczytałem raporty testów oraz niezależnie sprawdziłem hashe i maksymalne różnice zapisane w dowodach Gloo. Zachowane wcześniejsze próby i ich błędy nie są nadpisywane tym werdyktem.

## Równoważność straty i kroku optymalizatora

W [model.py](model.py) `distributed_optimizer_step` najpierw sprawdza, czy suma liczby lokalnych obrazów jest równa globalnej liczbie obrazów paczki. Dla 32 obrazów dwa procesy otrzymują po 16, a dla ostatnich 28 po 14. Cztery procesy, objęte testem CPU, otrzymują odpowiednio po 8 lub po 7. W każdym przypadku każdy obraz zachowuje wszystkie 600 tokenów.

Każdy lokalny fragment błędu jest dzielony przez **globalne** B×600 i właściwy wymiar bloku oraz horyzont. Następnie gradienty są sumowane przez all_reduce SUM. Nie ma dodatkowego dzielenia przez liczbę GPU. To daje gradient tej samej globalnej średniej, a nie średnią lokalnych strat z błędną wagą ostatniej paczki.

Formalnie, dla rozłącznych zbiorów tokenów S_r i jednej kary R:

`L = Σ_r [Σ_(i∈S_r) ℓ_i / N] + R`, gdzie `N = B·600`.

Rank0 dodaje R raz; pozostałe procesy mnożą karę przez0. Akumulacja gradientu względem A oraz lokalne VJP przez A=exp(G) są liniowe względem gradientu wejściowego. Przy identycznym G suma lokalnych VJP jest tym samym VJP co dla sumy lokalnych gradientów A. W kodzie nie ma potrzeby uruchamiania stock DDP ani jego domyślnego uśredniania.

Po SUM następują kontrola skończoności, jeden clipping globalnego gradientu i jeden krok Adam w każdej identycznej replice. Model i optymalizator zaczynają z tego samego seedu; testy sprawdzają również zgodność parametrów po kroku, nie tylko zgodność wartości straty. Kara A−I nadal oznacza bliskość identyczności, nie ograniczenie promienia spektralnego.

## Kolejność operacji między procesami

W [train.py](train.py) wszystkie procesy przechodzą tę samą listę originów i globalnych paczek, a `np.array_split` rozdziela rozłączne indeksy. Wspólne `ensure_window` odbywa się przed lokalnymi odczytami. Po globalnym kroku telemetrię i checkpoint zapisuje wyłącznie rank0; pozostali dochodzą do tej samej kontroli błędu.

Przed walidacją wszystkie procesy przygotowują okno. Rank0 wykonuje walidację z `window_prepared=True`, zatem nie wchodzi samotnie w kolektywną ekstrakcję źródłową. Wynik walidacji jest rozsyłany do wszystkich procesów, co zachowuje zgodność wyboru epoki i warunku early stopping. Tak samo wspólne przygotowanie testowego okna następuje po zapisaniu FREEZE; tylko rank0 oblicza i zapisuje metryki testu. Moduł NN nie jest opakowany w DDP, więc samotny forward rank0 nie uruchamia ukrytych kolektywów gradientów.

Kod przekazuje błędy lokalnej akumulacji, skończoności gradientów, odczytów i operacji rank0 przez uzgodnione bramki `all_gather_object`. Testy obejmują zamierzoną awarię pojedynczego procesu w takiej bramce. Nie jest to formalny dowód odporności na każdy błąd infrastruktury: śmierć procesu, awaria CUDA lub utrata plików mogą przerwać job, a synchronizacja ma limit czasu 30 minut. Nie stwierdzono rozbieżności kolejności kolektywów na odczytanej normalnej ścieżce wykonania.

## Równoległy reader i cache

[dane.py](dane.py) ma pojedynczego autora nagłówka NPY i metadanych: rank0. Najpierw publikuje status building i tworzy pełny plik, następnie procesy synchronizują się i otwierają go przez r+. Każdy proces zapisuje indeksy `arange(rank,1500,world)`. Zbiory są rozłączne i ich suma obejmuje wszystkie 1500 obrazów. Flush i kontrola skończoności każdego wyekstrahowanego fragmentu poprzedzają wspólną bramkę ukończenia.

Dopiero po ukończeniu wszystkich fragmentów rank0 sprawdza sumę liczników, zgodność wskazanych źródłowych sond, oblicza SHA i atomowo publikuje complete. Inne procesy dostają te same metadane. W trybie wieloprocesowym `get_state` tylko odczytuje już przygotowany slot; nie rozpoczyna samotnej ekstrakcji. `get_history_batch` zachowuje wcześniejszą kolejność dwóch przeszłych stanów, origin i10 celów; model dostaje jako wejście tylko pierwsze trzy stany.

Test Gloo readera sprawdza 37 rozłącznych wierszy, jeden nagłówek, atomowe complete i propagację celowego błędu. To test mechanizmu pamięci mapowanej, nie dowód ukończenia rzeczywistej ekstrakcji 1500. Tę ostatnią sprawdzi obowiązkowy `reader_preflight.py` na źródłowym oknie 0..10 w przydzielonym zadaniu GPU.

## Kontrole przed fit i reset stanu

[Launcher dwóch kart](run_multigpu/bieg_eden_2gpu.slurm) żąda jednego węzła dgx-4 i dwóch GPU. Kolejno sprawdza pliki przez EXECUTION_FREEZE.sha256, przygotowuje dokładne obrazy, uruchamia syntetyczny test końcowego rdzenia na 600 tokenach, równoległą ekstrakcję 0..10 i dopiero driver treningowy. `set -euo pipefail` zatrzymuje sekwencję po nieudanym etapie.

`dataset_preflight` w driverze wykonuje dwa porównania pojedynczej i podwójnej karty na rzeczywistych danych: obrazy 0..31 oraz 1472..1499, czyli paczki 32 i 28, każda po 600 tokenów. Porównuje stratę, gradienty, parametry po Adam, zapis telemetrii i checkpoint. Liczniki preflight mają osobne nazwy i `actual_fit_steps=0`.

Po kontroli oba modele preflight są usuwane. Właściwy model powstaje ponownie przez `make_model()`, z ustalonym seedem, pustym nowym Adam i zerowym licznikiem kroków. Również final refit tworzy nowy model. Kontrole numeryczne na dozwolonym prefiksie nie aktualizują modelu użytego do fit ani nie wybierają hiperparametrów.

Wciąż obowiązuje inner fit do cp80, walidacja 80→90, fresh refit do cp90 i test 90→100 po freeze. Skala natywna pochodzi wyłącznie z cp0, a aux skaluje się na odpowiednim prefiksie 80/90. J128 pozostaje tylko oceną po dekodowaniu. Zmiana liczby kart nie zmienia CONFIG ani tej metody.

## Odczytane dowody

| Kontrola | Wynik i zakres |
|---|---|
| CPU Gloo, 2 procesy | PASS; maksymalna różnica gradientów 3,7253e−9, parametrów 7,4506e−9; paczki 32/28 |
| CPU Gloo, 4 procesy | PASS; maksymalna różnica gradientów 3,7253e−9, parametrów 1,4901e−8; paczki 32/28 |
| GPU job 1819570,2 A100, full768/all600 | PASS; maksymalna różnica gradientów 1,0245e−8, parametrów 7,5251e−7, decoded forecast 8,3447e−7; dwie syntetyczne paczki, nie fit danych 1500 |
| Skrócony rzeczywisty driver z Gloo i atrapą readera | Test przechodzi 8 globalnych kroków, 3 walidacje, wybór epoki 1, fresh refit, freeze i zapis pojedynczego autora |
| Docelowy runtime Eden | Wykonawca potwierdził 15 testów PASS oraz ostatni test drivera PASS po poprawce wyłącznie mocka CPU `is_current_stream_capturing`; kod produkcyjny nie zmienił się wskutek tej poprawki |
| Dokładne dane 1500 na dwóch GPU | Jeszcze niepotwierdzone w chwili werdyktu; automatyczny warunek przed właściwym fit w pełnym zadaniu |

Źródła liczb: [Gloo2](PROOF_CPU_GLOO_2.json), [Gloo4](PROOF_CPU_GLOO_4.json), [zachowany GPU smoke](run_multigpu/smoke_2gpu/RESULT_ALL600_1819570.json), [gotowość kodu](run_multigpu/CODE_READINESS.json). Maksima z plików Gloo przeliczyłem niezależnie. Nie uruchamiałem ponownie testów GPU.

Wcześniejszy GPU smoke ma własny snapshot. Końcowy rdzeń ma dodatkowe guardy i jest ponownie sprawdzany w pełnej alokacji przed treningiem; nie utożsamiamy historycznego smoke z dowodem wykonania nowych guardów.

## Hashe zatwierdzonego zakresu

| Plik | SHA256 |
|---|---|
| model.py | `97f01e514bc6a0b0a884bf5c6165ffd36e9e9ab715de5da2d3747df7759b380c` |
| train.py | `52db0c0d9f47b1c0216e737181c87273aa8b5e7184abf41dc5211c094b88401c` |
| state.py | `03ed6aa3ac63ba42f91d56a4d63826ffc1ec2d7b7504386aeed0788c279fbd99` |
| dane.py | `caec2a0cdc9094564c0cbe5c43b4ec711165201612e6125de0c6c1753accc183` |
| przygotuj_eden.py | `cd8c4a9bd77b485889b0eac54854c26b428b633762796975fab6f5771902735b` |
| CONFIG.json | `c748028121ca0710a421a168e5fbb38130aaeb2eb91c65dc432ad11f75f32dd8` |
| PLAN_MODELU.md | `73d059c7f352f7f2fe491033223108c15eff9ea186842bb5d0363ea012f67277` |
| run_multigpu/bieg_eden_2gpu.slurm | `74f24b71d7ab93eb8dc061dc36fbc4cd2473fbd0cca2247a16810ea2ff002e46` |

Hashe NN porównano z CODE_READINESS, readera i launchera obliczono z odczytanych plików. Wykonawca potwierdził zgodność 18 plików z nowym stagingiem na Edenie. Po teście należy sprawdzić realne liczniki, freeze, źródła, zapisane prognozy czterech sond i agregaty wszystkich 1500 obrazów. Wynik pozostaje eksploracyjny dla jednego wcześniej analizowanego biegu ViT; nie daje gwarancji 85 procent ani dowodu przyczynowego sterowania przez statystyki AdamW.
