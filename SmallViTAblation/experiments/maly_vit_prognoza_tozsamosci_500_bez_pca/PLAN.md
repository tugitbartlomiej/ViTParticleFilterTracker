# Poszukiwanie prognozy tożsamości do 500 aktualizacji bez PCA

Data: 2026-10-02. Status: protokół zamknięty przed dopasowaniami; siatka i wybór poniżej. Użytkownik zlecił poszukiwanie dobrego modelu spośród Koopmana, Langevina, Kalmana i obserwatorów stanu, z celem ponad 85 procent i horyzontem do 500 kroków.

## Cel i zakres

Zachowujemy dokładnie 128 oryginalnych kanałów J z eksperymentu `maly_vit_tozsamosc90_bez_pca_langevin`. Nie wyznaczamy PCA, POD, rotacji ani autoenkodera. Każda prognozowana cząsteczka zachowuje wszystkie 128 współrzędnych; nie wybieramy łatwych kanałów ani tylko największych zmian po obejrzeniu celu.

Robocza interpretacja 85 procent zgodna z dotychczasowymi wykresami: trafność znaku skumulowanego przemieszczenia z(t+H)−z(t), uśredniona po wszystkich 128 współrzędnych, z wagami pokrycia maską wewnątrz obrazu i równą wagą obrazów. Nie jest to rozpoznanie obecności narzędzia ani procent poprawnie przewidzianych całych wektorów. Dodatkowo obowiązkowo MSE, redukcja MSE względem braku ruchu, błąd normy i cosinus zmiany. Wysoki znak przy pogorszeniu MSE nie jest sukcesem prognozy wielkości.

Zapisujemy osobno przekroczenie progu na każdym horyzoncie oraz maksimum osiągalnego horyzontu. Wysoki znak przemieszczenia po 500 krokach nie oznacza trafnego odtworzenia wszystkich 500 przyrostów po drodze. Nie przeliczamy cosinusa na procent poprawności.

## Ograniczenie danych i brak przyszłości

Znana gęsta historia ma stany 0–1000. J zidentyfikowano na checkpointcie 500. Dlatego pełny, przyczynowy test H=500 po ustaleniu J ma tylko jeden moment startu: 500, z celem1000. Krótsze horyzonty można dodatkowo sprawdzać na późniejszych startach. Jeden moment H500 nie daje przedziału niepewności po niezależnych startach czasu. Cząsteczki z tego samego obrazu i wszystkie obrazy z tego samego treningu są zależne.

Wybór modeli odbywa się wyłącznie z historii dostępnej do 500, poprzez wewnętrzne prognozy wcześniejszych odcinków. Ponieważ J określono na k500, są one retrospektywną diagnostyką historii w ustalonej bazie, nie twierdzeniem o wdrożeniu tych kanałów już w300. Dotychczasowa historia testowa była oglądana; nowa próba jest eksploracyjna i nie stanowi nietkniętego niezależnego potwierdzenia.

Krytyczne: walidacja z momentu300 nie może odejmować poziomu z500 jako kotwicy. Stan ma być surowy lub centrowany wyłącznie danymi z prefiksu właściwego dla danego dopasowania. J pozostaje stałym zbiorem indeksów, ale centra, skale, równowagi, stany obserwatora i kalibracja szumu nie mogą znać późniejszych pomiarów.

Kalman/obserwator wykorzystuje rzeczywiste pomiary tylko do chwili startu prognozy. Potem przechodzi w tryb otwarty i wykonuje samą predykcję. Nie wolno korygować kolejnych stanów prawdziwymi pomiarami w przedziale prognozy i nazywać tego prognozą H500. Filtracja wsteczna wykorzystująca cały zapis jest zabroniona jako wejście predykcji.

## Rodziny do zaprojektowania

- Punkt odniesienia: brak zmiany, ostatnia prędkość, wygaszana prędkość, trend z prefiksu, cofanie do równowagi wyznaczonej z przeszłości.
- Koopman: liniowy operator w natywnych kanałach, z jawną regularyzacją i wariantem stabilizowanym; stan ani cel nie są redukowane.
- Langevin/GLE: dryf i pamięć wspólnie dopasowane na historycznych stanach; pamięć skończona lub kilka czasów wykładniczego zaniku. Oddzielny test wspólnego ruchu narzędzia i kontekstu.
- Kalman i obserwator: stan pozycji oraz prędkości, wygaszanie trendu i szacowanie stanu z prefiksu; prognoza po starcie bez pomiarów. Szum pomiarowy jest założeniem modelowania, ponieważ zapisane reprezentacje są deterministycznym odczytem danej sieci i obrazu.

Nie poszerzamy siatki po wynikach testowych, aby przeszukiwaniem nie wytworzyć pozornego przekroczenia85 procent.

## Zamrożony protokół wykonawczy

Zamknięcie: 2026-10-02, przed pierwszym dopasowaniem. Wybrane dane zapisano w DANE_J128.npz (SHA256 bdaacfc9f3a98b09ffebbfdc5e20451c9b955e59f7e5666c881c7f04498fc790), 218 tokenów narzędzia z 12 obrazów. Podczas wyboru parametrów: dopasowanie na obrazach 0–5, walidacja na obrazach 6–7. Dwa foldy: parametry z kroków do250, start prognozy300, H=[1,10,50,100,200]; parametry do350, start400, H=[1,10,50,100]. Centra i skale dopasowania kończą się na end; stan początkowy i pamięć obserwatora mogą korzystać z rzeczywistych danych do origin. Wszystkie cele walidacyjne kończą się najpóźniej na500.

Techniczny zakres regresji liniowej: wszystkie warianty L0, finite i EMA wykorzystują te same zakończone przejścia z celami 5..end. Pierwszy cel5 zapewnia historię dla L4. Stany są q−q(0) i bg−bg(0); EMA zaczyna się od zera w stanie0. Skale i centra regresji są wyłącznie z fitu.

Agregacja: najpierw równoważna średnia po obrazach i dostępnych foldach dla każdego H, następnie równa średnia po różnych H. H200 ma tylko jeden fold. Trafienie znaku oznacza dokładną zgodność sign(predicted displacement) i sign(actual displacement), w tym jawne sign(0)=0. Nie pomijamy małych zmian. Brak ruchu ma zerowy znak, nie sztuczne 50 procent trafień. Pokazujemy też strukturę znaków i punkt odniesienia, aby nierównowaga wzrostów/spadków była widoczna.

Kandydat przechodzi bramkę przy średnim MSEskill>0 i średnim MSEskill dla H>=100 co najmniej0. Ranking wśród przechodzących: największa minimalna trafność znaku po H, następnie średnia trafność znaku, następnie średni MSEskill, na końcu leksykograficzny identyfikator konfiguracji. Jeśli żadna konfiguracja nie przechodzi, ranking diagnostyczny korzysta z tej samej kolejności bez bramki i jest jawnie oznaczony jako porażka bramki. Zachowujemy zwycięzcę globalnego, najlepszy średni MSEskill i zwycięzcę każdej rodziny wybranego tą samą regułą (brak przejścia bramki w rodzinie oznaczamy).

Wybór zostaje zapisany przed końcową oceną. Wybrane modele dopasowujemy na obrazach0–7 do500 i prognozujemy ze startu500 na H=[1,2,5,10,20,50,100,200,300,400,500]. Raportujemy oddzielnie obrazy dopasowania0–7 i transfer8–11, dodatkowo opisową średnią wszystkich12 obrazów. Wszystkie te obrazy pochodzą ze zbioru treningowego detektora; transfer dotyczy modelu dynamiki. Nie zgłaszamy niezależności czasowej ani niezależnego nowego biegu. Próg85 musi być spełniony na ocenianych horyzontach, a nie tylko w ich średniej. Maksymalny horyzont sukcesu liczymy po prefiksie całej listy H; pojedyncze późniejsze przekroczenie nie oznacza sukcesu na wszystkich wcześniejszych H.

## Siatka przed pierwszym dopasowaniem

Wszystkie stany i wyjścia zachowują128 oryginalnych kanałów. Kanały nie są mieszane przez redukcję wymiaru. Regresja może wiązać współrzędne pełną macierzą128×128. Numeryczny rozkład macierzy do przycięcia widma nie jest PCA i nie usuwa żadnego kanału.

- Natywny Koopman/L0: ridge [.1,1,10], każdy local/coupled, 6 konfiguracji.
- Skończona pamięć GLE: L=[1,2,4], ridge [1,10], local/coupled, 12 konfiguracji.
- Wykładnicza pamięć GLE: dwa ustalone banki półtrwań [2,10,50] i [10,50,200], ridge [1,10], local/coupled, 8 konfiguracji.
- Dodatkowy local Koopman z przycięciem widma przejścia do rho=[.995,1], ridge [.1,1,10], 6 konfiguracji. Po clip intercept wyznaczony ponownie jako mean_train(Y)−F mean_train(X). Jest to heurystyka, nie gwarancja stabilności długiej prognozy. Raportujemy promień widmowy i przejściowe wzmocnienie. GLE nie jest naprawiany po obejrzeniu testu.
- Brak zmiany i ostatnia prędkość: 2 kontrole.
- Wygaszana ostatnia prędkość: tau=[10,50,200,1000], 4 konfiguracje.
- Trend OLS z ostatnich [16,64,128] stanów, przyszłe wygaszanie tau=[50,200,nieskończoność], 9 konfiguracji.
- Powrót do średniego poziomu z ostatnich [50,200] stanów, tau=[50,200], 4 konfiguracje.
- Obserwator prędkości Kalmana: ratio=[.0001,.01,1], tau=[20,100,500,nieskończoność], 12 konfiguracji. F=[[1,rho],[0,rho]], Q=diag(.001*ratio,ratio), R=1, rho=exp(−1/tau). Filtruje do startu, następnie pozycja równa obserwowanemu q(start), a przyszłość to suma wygaszanej estymowanej prędkości. To jawna konstrukcja obserwatora prędkości, nie wygładzanie przyszłych pomiarów.
- Sprzężony obserwator tła i reszty: tau_bg=[100,500], tau_res=[10,50], stałe ratio_bg=.01, ratio_res=.1, 4 konfiguracje. Per-kanał beta regresji przyrostu narzędzia na przyrost tła dopasowany tylko w prefiksie na obrazach fit, ridge względny .1. Prognozuje autonomicznie zarówno tło, jak i resztę narzędzia; suma wraca do wszystkich128 kanałów.

Powyższa siatka bazowa zawiera67 konfiguracji. Nie dodajemy przyszłych rzeczywistych paczek ani gradientów. Dłuższe historie innych biegów z poolingiem ROI nie zastępują obecnych trajektorii cząsteczek.

### Uzupełnienie przed pierwszym fitem: krok modelu co10

2026-10-02, 11:19–11:21 +02:00. Worker potwierdził, że żaden fit nowego eksperymentu jeszcze nie ruszył. Dodajemy32 konfiguracje liniowe (Koopman/GLE) również z krokiem dt=10, ponieważ poprzedni ukończony eksperyment J128 wskazywał przewagę takiego kroku dla dłuższych horyzontów. Zmiana nie wynika z wyników bieżącej walidacji ani testu. **Ostateczna siatka:99 konfiguracji =32 liniowe dt1 +32 liniowe dt10 +35 obserwatorów/kontroli.**

Wrapper dt10 używa tylko stanów0,10,20,...; end i origin dzielą się przez10. Pierwszy target5 w jednostkach modelu odpowiada aktualizacji50, wspólnie dla wszystkich32 konfiguracji dt10. Opóźnienia i półtrwania pamięci są w krokach modelu: banki EMA dt10 mają półtrwania [20,100,500] lub [100,500,2000] aktualizacji sieci. Przycięcie rho=.995 lub1 dotyczy również pojedynczego kroku modelu, a więc dziesięciu aktualizacji dla dt10. H500 odpowiada50 iteracjom modelu. Dla H niedzielących się przez10 prezentujemy liniową interpolację **pomiędzy samymi prognozowanymi stanami**, nigdy rzeczywistymi przyszłymi checkpointami. To przybliżenie ruchu wewnątrz odcinka; nie dowód odtworzenia szybkiej dynamiki H1. Reguły wyboru, foldy i raportowanie pozostają takie same, a rodziny dt10 są oznaczone osobno.

Weryfikacja: test podmiany przyszłego sufiksu na NaN, zgodność pełnego i uciętego prefiksu, zgodność natywnych i standaryzowanych operatorów liniowych, stałość zbioru J, dokładny kształt wyjścia i skończoność. Dla obserwatorów dodatkowo niezmienniczość względem stałego przesunięcia współrzędnych, zachowanie stałej trajektorii i odtworzenie dokładnego trendu przez OLS.

## Zapis i odpowiedzialność

Root: ten plan, końcowe PODSUMOWANIE.md, integracja wyników. Agent plan_poprawek: na razie inwentaryzacja danych i propozycja wykonawcza, następnie kod/metryki/figury po zamrożeniu protokołu. Agent astra_koopman_wagi: LITERATURA_I_AUDYT.md, źródła pierwotne i kontrola interpretacji. Nie zmieniamy poprzednich eksperymentów, nie kasujemy danych, nie commitujemy. Używamy istniejącego F:/Python/python.exe i odczytu mmap; nie kopiujemy surowych22GB.
