# Plan rozszerzenia badań ViT: odpowiedź, tożsamość, pamięć

Data: 2026-10-16. Status: **projekt do pilota, nie wykonana kampania**. Dokument rozwija protokół v1.2 w `experiments/protokol_testow_vit_iclr_neurips_20261016/`; nie zmienia jego zamrożeń, endpointów ani istniejących plików. Przed confirmation wybrane rozszerzenia trzeba jawnie włączyć do nowej wersji prerejestracji. Dziewięć poniższych pozycji to karty wariantów/kontroli, nie dziewięć dodatkowych pełnych treningów.

## Jedna teza główna i warunki jej obalenia

Hipoteza robocza: podczas uczenia zmienia się przenoszenie semantycznych zmian obrazu do odczytu detekcji; określone bloki i kierunki wzmacniają istotne cechy oraz ograniczają wpływ zakłóceń. Sama obecność S/A, gęstość punktów ani niski rząd odczytu nie potwierdzają hipotezy.

Wybrać na discovery jeden konkretny mechanizm: np. odpowiedź na zmianę widocznej krawędzi narzędzia przy zachowanym tle albo zachowanie obiektowości przy kontrolowanej okluzji. Przewidzieć znak efektu na wybraną krawędź/logit, warunki efektu i warunki braku efektu. Po wyborze zamrozić warstwę, populację przykładów, sondę, donor policy i wielkość interwencji. Nie wybierać ich po odczycie confirmation.

Obalenie wersji mechanistycznej: efekt równie duży dla dopasowanych kierunków losowych/nuisance; brak przewidywanej selektywności dawców; brak replikacji; wyjaśnienie całego pozornego wyniku przez zmianę kandydata lub kontekst filmu. Dobry wynik geometryczny przy braku efektu zadaniowego pozostaje wynikiem geometrycznym.

## Mapa do istniejących testów

| Karta | Co sprawdzamy | Gdzie włączyć | Priorytet |
|---|---|---|---|
| F01 | Kierunkowe przywracanie, niewzajemność i pętla cyrkulacji | T02, T03, T05; interpretacja w T09 | Mała kontrola przy deklaracji niekonserwatywności |
| F02 | Transport perturbacji przez rzeczywisty suffix, zmiany osi, odczyt | T04, T07, T11 | Rdzeń pomiaru |
| F03 | Q ze śladem/bez, ścinanie, konwencja znaku i kontrprzykłady | T03, T04, T06 | Tani audyt instrumentu |
| I1 | Odczytywalna informacja o obiekcie/kategorii/tracku | T00, T08, T10, T26 | Pierwszy pilot po audycie etykiet |
| I2 | Trwałość informacji na filmie lub jawnej rodzinie augmentacji | T08/T26, nowy endpoint warunkowy | Dopiero po I1 i kontroli par danych |
| I3 | Obserwowalność, kernel W2, selektywny patching | T09, T11, T15, T16, T17 | Główna droga do mechanizmu |
| MEM1 | Historia ponad aktualny token, zegar i silny baseline | T18, T26; T25 tylko osobny cel jakości | Warunkowe, gęste nagrania |
| MEM2 | Czy historia zastępuje bieżący kontekst attention | T07, T11, T14 | Mały test domknięcia |
| MEM3 | Sparowane replay: pamięć optymalizatora i ruch tokenów | T18, T19 | Najdroższa gałąź, po wynikach pilota |

MEM1–MEM3 odpowiadają M1–M3 w załączniku `PAMIEC_MORI_ZWANZIG.md`; przedrostek MEM zapobiega pomyleniu z oznaczeniem hipotezy mechanistycznej w głównym protokole. Szczegółowe wzory, kontrole i falsyfikacje: `outputs/mosty_fizyka_tozsamosc_20261016/{OPTYKA_I_WIRY,TOZSAMOSC_I_KOOPMAN,PAMIEC_MORI_ZWANZIG}.md`.

## Etap 0: dane i znaczenie słowa tożsamość

1. Zinwentaryzować etykiety: bbox, maski, kategoria narzędzia, track, fizyczny egzemplarz, film/zabieg/pacjent. Brakujące oznaczyć jako unknown. Nie utożsamiać bbox z segmentacją.
2. Ustalić wykonalny poziom tezy. Jeśli są tylko bbox jednej zbiorczej klasy, pierwszy test dotyczy **obiektowości**, a nie rozpoznawania rodzaju ani egzemplarza.
3. Dla kategorii potrzebne są różne rodzaje w wielu filmach; dla tracka prawidłowe korespondencje i negatywy tej samej kategorii. Jeżeli film i identity są nierozdzielne, stwierdzić brak identyfikowalności.
4. Utrzymać zasady discovery/confirmation i klastrowania z T00/T21/T27. Osiemset klatek jednego zabiegu nie zastępuje niezależnych zabiegów.
5. Rozdzielić zegary: checkpoint k, podkrok l, czas filmu s oraz indeks augmentacji g. Osobne pola manifestu i osobne analizy.

## Etap 1: mały pilot na Edenie

Punktem wyjścia jest ograniczony pilot T09: 24 obrazy × 3 ustalone checkpointy × 3 bloki × 1 czysty kandydat = 216 lokalnych punktów. To **przykładowy limit techniczny**, nie próba wystarczająca do tezy publikacyjnej. Część I1 może odczytać wszystkie sześć bloków z tych samych forwardów. Obrazy dobierać warstwowo według dostępnych etykiet i filmów, a nie największego wiru/IoU.

### A. Wykonalność miernika

- Potwierdzić zgodność forward/dekodera i hooków.
- Zweryfikować finite differences/JVP oraz promień lokalności; wybrać 8–16 sond z istniejącego protokołu, jawnie oddzielając naturalne zmiany obrazu od sztucznych kierunków.
- Dla małej podpróby zrobić F01: dwie płaszczyzny zadaniowe/nuisance i losowa, trzy promienie z T05, zagęszczenie kwadratury. Nie używać ich indeksu jako fizycznego czasu.
- Policzony antysymetryczny składnik sprawdzić na gradiencie kwadratowym, czystym ścinaniu i polu `−kx+ωRx`. To testy estymatora.
- Na tych samych punktach rozłożyć J̄ na `Σ ū_c ḡ_cᵀ`, gdzie g_c jest pochodną aktywacji neuronu względem wejścia. Sprawdzić zgodność sumy z autodiff. Zapisać normy/kąty pary „kierunek reakcji, kierunek działania”, `Σ||Ā_c||²` i `||ΣĀ_c||²`; znoszenie składników oznaczyć jako algebraiczne, nie optyczne. To rozszerzenie T02/T12/T13; selektor neuronów zamrozić przed confirmation. Nie liczyć wszystkich par, jeśli nie są potrzebne w zadeklarowanym teście.
- Null W2 sprawdzić w ustalonej tolerancji FP32: interwencja po GELU z ker(W2) ma nie zmienić dalszego wyjścia. Dodać kontrolę w rowspace o tej samej normie. Nie traktować nullspace jako listy naturalnych semantycznych kierunków.

### B. Informacja i jej odczyt

- I1: pełna reprezentacja, PCA, random i kierunki stabilne wobec jawnych zmian obrazu; te same rangi/regularizacja/splity. Osobno residual192 i GELU768 oraz diagnostycznie rowspace/nullspace W2.
- Uwzględnić baseline z samego bbox, pozycji i tła. W pooled reprezentacji GT/maska jest wejściem oracle, identycznym dla wszystkich porównań.
- Obliczyć pełny suffix Jacobian-vector product do logitu i znormalizowanych krawędzi ustalonego kandydata. Dla rozkładu S/A `C_out` zaczyna się na wyjściu badanego MLP, a nie jego wejściu. Przechować `C̄δ̄`, `C̄Āδ̄`, `C̄S̄δ̄` oraz `C̄(I+J̄)δ̄`, pełne odpowiedzi i wyrazy mieszane w bazie ilorazu LN. Dokładne rozmiary i rozróżnienie G_in/G_out podaje RAPORT.md.
- Zmierzyć konkretne skończone efekty interwencji. Bez nich wynik pochodnej pozostaje lokalnym przybliżeniem.

### C. Bramka do confirmation

Przejście wymaga: poprawnego instrumentu; etykiet pozwalających odróżnić cel od tła; użytecznego zakresu perturbacji; oszacowania wariancji między filmami i seedami; zamrożonej hipotezy i progu praktycznego. Pilot nie służy do uznania tezy za potwierdzoną. Wynik ujemny lub przewaga random/PCA zamyka odpowiednią mocniejszą interpretację, nie prowadzi do dobierania nowego endpointu na confirmation.

## Etap 2: interwencje i replikacja

W I3 podmieniać małą podprzestrzeń, nie cały końcowy tensor. Zachować pełny układ dawców: zgodna identity/inny kontekst; inna identity/podobny kontekst; identyczny donor; losowa podprzestrzeń o dopasowanej randze i faktycznej normie korekty. Osobno noising i denoising; cofnięcie własnej zmiany to kontrola dodatnia, nie dowód mechanizmu.

Przy tezie o semantycznej podprzestrzeni uwzględnić LEACE jako porównanie usuwania informacji liniowej, dopasowane wyłącznie na discovery. Dopasowanie rank/norm dotyczy rzeczywistego zniekształcenia, nie samej nazwy operatora; LEACE może być rzutem ukośnym. Ocenić ponownie odczytywalność liniową i ograniczoną nieliniową oraz efekt na detektor. Nie interpretować liniowego erasure jako usunięcia całej semantyki. W related work uwzględnić Huang–Chang 2025 o rozbieżności probing/patching w liczących ViT z CLS; nasz gęsty dekoder ma inny graf zależności.

Podstawowy endpoint dotyczy stałego kandydata; pełny dekoder stanowi równoległy wynik zadaniowy. Zapisać peak mask, ranking/top10 i winner ID, żeby zmiany wyboru nie udawały poprawy regresji. Przy translacji respektować warunek osiągalności kotwicy z protokołu v1.2.

W ostatnim punktowym MLP interwencja w token inny od ustalonego kandydata nie zmienia jego ramki; może zmienić ranking. To dodatkowa kontrola ujemna, nie para do podstawowego P1.

Powtórzyć na niezależnych seedach i odłożonych filmach. Nie mieszać surowych baz seedów. Algorytm wyboru rangi/warstwy i normalizacji zamrozić; ewentualne dopasowanie baz na osobnej kalibracji musi być zdefiniowane wcześniej. Przedziały i moc liczyć po niezależnych jednostkach z zachowaniem sparowania, nie po tokenach. Progi praktyczne i kierunek efektu ustalić przed confirmation.

## Etap 3: gałęzie warunkowe

### Koopman / I2

Zamrozić checkpoint i zbierać tracki w rzeczywistym czasie filmu. Jeśli nie ma tracków, użyć par kontrolowanych augmentacji i tak nazwać operator. Estymacja oparta na C00, C01, C11 z regularyzowanym whiteningiem, ze stałą obserwablą potraktowaną osobno. Trwałość jednej funkcji oceniać bezpośrednio, a nie przez dowolną dużą singular value.

Dwa odrębne nulle, zawsze z refitem estymatora: tasowanie czasu wewnątrz tracka oraz rozrywanie relacji identity między odpowiednio dopasowanymi trackami. Pierwszy zachowuje stałą tożsamość i nie powinien automatycznie niszczyć jej odczytu. Ocenić przyrost ponad statyczny I1/PCA/persistence. Brak przyrostu oznacza pozostawienie Koopmana jako analizy pomocniczej.

### Pamięć / MEM1–MEM3

Dwanaście checkpointów atlasu nie wystarcza do dopasowania bogatego jądra. Na stałej próbce obrazów zapisać residuale na siatce kroków z planowanego cache, początkowo co 100 aktualizacji, oraz krótkie gęste okna co krok do analizy krótszej skali. Wielkość próbki i okna ustalić po pomiarze kosztu. Nie interpolować brakujących obserwacji.

MEM1 porównać z persistence, ekstrapolacją, nieliniowym modelem aktualnego stanu i zegara/LR. Strojenie lagów wyłącznie discovery; chronologiczne fit→score z przerwą eliminującą wspólne cele. Prognoza nie może używać przyszłego Δθ ani future-PCA.

W nowym seedzie dopuszczalny wariant to wcześniej zamrożony algorytm dopasowany na ustalonym prefiksie treningu i filmach kalibracyjnych VAL, a oceniany na późniejszym sufiksie i odłożonych filmach. Granice prefiksu, luki, cechy i hiperparametry zamrozić przed confirmation. To replikacja procedury estymacji, nie predictor przeniesiony między seedami bez ponownego fit.

MEM2 osobno dla głębokości: historia powinna tracić przewagę po podaniu bieżącego kontekstu, jeśli to właśnie ten kontekst zastępowała. Punktowe MLP z pełnym obecnym tokenem i wagami stanowi kontrolę deterministyczną.

MEM3 to krótkie replay/forks w ramach T19. Zapisać AdamW m/v, scheduler, BN, RNG, sampler oraz dokładne batch/augmentacje. Identyczne odtworzenie musi się zgadzać. Zmiana momentów lub kolejności danych jest jawną interwencją, nie równoważnikiem zmiany temperatury. Bez tego etapu można mówić o predykcyjnym opisie ruchu, nie o przyczynowym wyjaśnieniu trajektorii treningu.

## Eden → eksport → lokalny odczyt

Zachować istniejące trzy pakiety: `analysis`, `viewer`, `replay`. Rozszerzyć ich manifest, nie tworzyć niezależnego sposobu przechowywania. Standardowy odczyt i interpretacja po pobraniu mają działać na CPU bez ładowania checkpointów GPU. Nowe interwencje wymagające forwardu idą do następnego zlecenia na Edenie.

| Pakiet | Dodatkowe wymagane dane |
|---|---|
| analysis | Surowe sparowane wyniki przed/po, wektory odpowiedzi, sondy, Cδ/CAδ/CSδ, jednostki W, promienie, błąd Taylor/kwadratury, całe foldy i status braków; nie tylko PNG i średnia |
| identity | Etykiety kategorii/tracka i ich pochodzenie, mapowanie token↔obiekt, transformacje, donor IDs, bazy/P, zakres fit, probes i prognozy wszystkich baseline; zapisać jako część analysis |
| memory | Prawdziwe step i LR, obserwowane residuale, dostępne przeszłe historie, rolling origins, prognozy każdego modelu, identyfikatory okien i kryteria selekcji |
| viewer | 600 tokenów, 13 stanów, niezmienny identyfikator obrazu/tokenu, projekcja i jej fit-scope, źródłowy obraz/GT, ranking, wybrane wiązki; osobne osie checkpoint/depth/video |
| replay | Dokładne modele i bufory; dla MEM3 także pełny restart optymalizatora i danych. Cały replay może zostać na Edenie, lokalny pakiet analityczny musi wskazywać jego hash i ścieżkę |

Minimalne klucze rekordów: `run_id, seed, checkpoint_step, image_id, video_id, patient_if_known, frame_time, token_id, block, substep, representation_space, object_id_if_known, split, intervention_id, donor_id, probe_id, radius, fold, status`. Brak danych, zero i NaN to różne stany.

Zapis pełnych aktywacji ograniczyć do potrzebnej podpróby. Przykład kosztu surowego float32, bez narzutów: 24 obrazy × 2 widoki × 3 checkpointy × 13 stanów × 600 × 192 × 4 bajty ≈ 0,86 GB. Dodatkowo wszystkie sześć GELU768 dla tych samych przypadków ≈ 1,59 GB. To przykład pakietu pilota, **nie pomiar użycia pamięci GPU** ani całkowity koszt kampanii. Dla I1 można przechowywać tylko ustalone reprezentacje obiektowe i wybrane tokeny, zachowując podpróbę pełnych tensorów do audytu.

Gęste nagrania treningu mają osobny limit w bajtach i forwardach. Nie zapisywać pełnego attention Jacobianu; używać JVP/VJP oraz małych Gramianów na sondach. Ładować checkpoint raz i obsługiwać całą zaplanowaną paczkę pomiarów. Wznowienia muszą omijać zakończone jednostki na podstawie manifestu i hashy, nie samej obecności katalogu.

Każdy pakiet: wersja schematu, kod/config hash, model/data hash, dtype, bibliotek/CUDA, ziarna losowania, znaczniki complete, sumy SHA256 i rozmiary. Pobieranie wznawialne, weryfikacja przed lokalną agregacją; źródła Edena pozostają zachowane. E0 ma sprawdzić pełny offline readback i odtworzenie jednej figury/metyki z surowych rekordów przed uruchomieniem dużego biegu.

## Co pozostaje niewykonane

Nie napisano runnera modeli dla tych rozszerzeń, nie wybrano danych/etykiet, nie ustalono progów mocy ani nie wysłano jobów. `przeszukaj_rag.py` jest wyłącznie klientem wyszukującym w istniejącej bazie wiedzy. Przed wykonaniem: audyt etykiet → integracja wybranych kart z runnerem T00–T29 → E0 → pilot → zamrożenie konfiguracji → confirmation → pobranie i ponowny audyt rzeczywistych wyników.
