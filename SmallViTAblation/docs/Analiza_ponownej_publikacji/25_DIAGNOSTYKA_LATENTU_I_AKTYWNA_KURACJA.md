# 25. Diagnostyka latentu, aktywna kuracja i uzupełnianie stanu

Data: 2026-09-11. **Osobny kierunek badawczy na prośbę autora; propozycja, bez wyników eksperymentalnych.** Rozwija [23 — połączenie z grantem](23_POLACZENIE_Z_GRANTEM_DYNAMIKA_LATENTU.md). Nie rozszerza automatycznie kolejki 51 treningów z [24 — szybkich ablacji](24_SZYBKIE_ABLACJE_I_OPTYMALIZACJA_EDEN.md).

## 1. Proponowany kierunek

**Tak, projekt grantowy można połączyć z kuracją danych. Najciekawsze pytanie brzmi: czy na podstawie diagnostyki reprezentacji umiemy wybrać działanie, które poprawi model przy ograniczonym koszcie?** Działaniem może być anotacja nowych klatek, pozyskanie innego nagrania, wydłużenie kontekstu albo dodanie brakującej obserwacji. Sam duży błąd predykcji nie rozstrzyga, które z nich pomoże.

Dokument 23 proponuje prosty selektor: pokrycie puli plus residual dynamiki. Tutaj proponuję kolejny krok: **diagnoza → wybór działania → niezależny pomiar skutku**. Potencjalnym wkładem jest trafność decyzji o rodzaju interwencji oraz korzyść w zadaniu, a nie samo połączenie DINO, Koopmana i Mori–Zwanziga.

Trzy możliwe warianty:

| Wariant | Co robi | Ocena na obecnym etapie |
|---|---|---|
| A. Diagnostyka wspiera selekcję | Zamrożone cechy i tanie predyktory wskazują klipy; etykiety służą małemu detektorowi | Najtańszy test użyteczności, najbliższy dokumentowi 23. Sam w sobie może mieć ograniczoną nowość. |
| B. Diagnostyka wybiera interwencję | Rozróżnia niedobór przykładów, zbyt ubogi predyktor, brak historii i brak obserwacji | Rekomendowany osobny problem badawczy. Wymaga wykazania, że wskazane działanie pomaga częściej niż proste reguły. |
| C. Wspólnie uczony latent i akwizycja | Enkoder, model pamięci i polityka pozyskiwania danych aktualizują się razem | Najdroższy i najtrudniejszy do interpretacji; dopiero po A i B. Nie zaczynać od pełnego JEPA. |

Roboczy temat: **„Diagnostyka brakującego stanu jako podstawa doboru danych i kontekstu w uczeniu z wideo”**. To hipoteza o użytecznej metodzie, nie potwierdzone pierwszeństwo.

## 2. Co dokładnie wnosi istniejący grant

| Materiał lokalny | Element do wykorzystania | Granica dowodu |
|---|---|---|
| [D59](F:/Studia/MCHTR/Grant/Projekt/Research/Tezy/tezy/D59.md) | Porównanie obecnego stanu z historią, kontrola ruchu, rangi i reszt | Teza ma status POSTAWIONA. Empiryczny test zależy od reprezentacji, predyktora, próby i horyzontu. |
| [E-AQ](F:/Studia/MCHTR/Grant/Projekt/Research/Tezy/zadania/E-AQ.md) | Uzupełnienie stanu przez filtry pamięci; kontrole względem opóźnień i filtrów losowych | Zadanie OTWARTE. Nie ma tu wyniku potwierdzającego skuteczność na chirurgicznym wideo. |
| [D116](F:/Studia/MCHTR/Grant/Projekt/Research/Tezy/tezy/D116.md) | Kontrola, czy przewidywalność wynika z niemal stałego identyfikatora/reżimu | Wynik dotyczący TFT nie dowodzi analogicznego zachowania DINO. Szczegóły i rozbieżności liczbowe są w dokumencie 23. |
| [F23](F:/Studia/MCHTR/Grant/Projekt/Research/Tezy/tezy/F23.md) | Rozdzielenie założeń matematycznych od empirycznej procedury | Formalizacja wybranych twierdzeń nie dowodzi poprawności selektora, estymacji z wideo ani poprawy AP. Nie wykonywano tu kompilacji Lean. |
| [miary.py](F:/Studia/MCHTR/Grant/Projekt/experiments/D59_bilans_pamieci/miary.py) | Punkt wyjścia dla tanich probe'ów NumPy | Wymaga adaptera i poprawek opisanych w 23: m.in. centrowania targetu przy MSE i respektowania granic trajektorii w korelacjach reszt. |

Plan grantowy dobrze uzasadnia badanie **jakości obserwowanego stanu**. Nie daje jeszcze gotowego algorytmu, który rozpoznaje przyczynę błędu w dowolnym filmie. Używamy jego pomysłów jako zestawu testowalnych kontrastów. [Zakres wcześniejszego odczytu i hashe źródeł](evidence_grant_2026-09-11/zrodla_lokalne.json).

## 3. Rozdzielenie, bez którego eksperyment byłby mylący

Niech `U` oznacza dostępną pulę filmów train bez ujawnionych etykiet bbox, `L_r` — etykiety pozyskane do rundy r, a `z_t = E(x_t)` — zamrożone cechy. Jeżeli predyktor dynamiki dopasowano już do wszystkich trajektorii z `U`, to **samo ujawnienie nowych bboxów nie zmienia ani z, ani danych tego predyktora**. Nie można wtedy twierdzić, że zakup etykiet zmniejszył pamięć zamrożonego latentu.

Trzeba osobno mierzyć:

1. **Wybór etykiet:** score dynamiki pomaga wybrać `L_r`, a efekt mierzymy jako jakość detektora. Score może pozostać stały we wszystkich rundach; jest wtedy statyczną kuracją z etapowym ujawnianiem etykiet, nie adaptującą się diagnozą.
2. **Uczenie reprezentacji lub stanu:** nowy enkoder albo kontekst zmienia dostępne informacje. Porównanie wymaga wspólnego zadania i niezmiennego celu diagnostycznego; spadek błędu w nowej, skurczonej przestrzeni niczego sam nie dowodzi.
3. **Pozyskiwanie obserwacji:** nowy film, kanał albo jeszcze nieobserwowana trajektoria rzeczywiście dodają informacje do modelu dynamiki. Następny latent w już dostępnym filmie nie wymaga ręcznej anotacji bbox.

W pilotażu stosować zamrożony enkoder diagnostyczny, ale normalnie trenowany mały detektor. Detektor może uczyć się z pikseli informacji pominiętej przez diagnostyczny DINO. Ograniczenia zamrożonej reprezentacji diagnostycznej nie są więc automatycznie ograniczeniami całego detektora.

Przykład: obserwacja `z = sin(theta)` może odpowiadać dwóm fazom o różnej dalszej ewolucji. Więcej par z tą samą niejednoznaczną obserwacją pozwala lepiej oszacować rozkład przyszłości, ale nie odróżnia tych faz w pojedynczym przypadku. Dodatkowa obserwacja `cos(theta)` lub odpowiednia historia może pomóc. To przykład kontrolny; chirurgiczne wideo nie musi spełniać warunków rekonstrukcji znanych z układów deterministycznych.

## 4. Diagnostyka: co porównywać, zanim nazwiemy błąd „pamięcią”

Przewidujemy ten sam target `y_t = z_(t+h)` w ustalonym horyzoncie czasu. Skalowanie, PCA i strojenie predyktorów wykonujemy wyłącznie na train. Okna respektują identyfikator operacji, PTS, przerwy i cięcia; nie łączymy końca jednego filmu z początkiem następnego.

| Predyktor | Dostępne informacje | Pytanie |
|---|---|---|
| P0: persistence | `z_t` | Czy uczenie dynamiki daje więcej niż kopiowanie stanu? |
| P1: ridge | `z_t` | Co wyjaśnia prosta funkcja bieżącej obserwacji? |
| P2: nieliniowy obecny stan | `z_t`, np. mały MLP lub ustalone cechy nieliniowe + ridge | Czy problemem jest przede wszystkim liniowa klasa modelu? |
| P3: historia | `z_t` i wcześniejsze cechy, ten sam typ predyktora co P2 | Czy przeszłość daje dodatkową moc predykcyjną poza elastycznym modelem obecnego stanu? |
| P4: dostępne wejście/kanał | Stan oraz zsynchronizowane rzeczywiście mierzone `u_t` | Czy dodatkowa obserwacja pomaga ponad stan i historię? |

Horyzonty pilotażowe, np. 0,2 / 1 / 3 sekundy, dobrać do faktycznego próbkowania, ruchu końcówki i dostępnej historii. To propozycje do kalibracji, nie ustalone optima. Wariant P4 wymaga obecności kanału; nie zastępować go GT fazy czy przyszłą pozycją narzędzia dostępną wyłącznie podczas analizy.

Dla błędu kwadratowego mierzymy różnicę ryzyka:

`Delta_hist = mean(||y - P2(z)||² - ||y - P3(z, historia)||²)`.

Predykcje do rankingu powstają poza próbą dopasowania danego predyktora, np. przez trzy foldy po operacjach wewnątrz train. Dla klipu liczymy średnią tych samych różnic, z jawnym minimalnym wsparciem. Dodatnia wartość jest **empiryczną przewagą konkretnego predyktora z historią**, a nie lokalną informacją wzajemną ani dowodem przyczynowym. Wynik może być ujemny. Nie dobieramy progu po zobaczeniu testu.

W populacji, dla optymalnych predyktorów warunkowej średniej i zagnieżdżonych informacji, historia nie zwiększa minimalnego MSE. W praktyce modele nie muszą osiągać tych optimów. Ponadto brak zysku MSE nie wyklucza zmiany warunkowej wariancji lub całego rozkładu przyszłości. Jeśli teza dotyczy rozkładowej Markowowskości, potrzebny jest odpowiedni test warunkowej niezależności albo właściwa ocena probabilistyczna; samo MSE nie wystarczy.

Kontrole konieczne do interpretacji: podobny budżet strojenia i kontrola liczby parametrów, model bieżący o zwiększonej pojemności, rzeczywiste opóźnienia kontra filtry losowe o tej samej liczbie cech, historia zamieniona między dopasowanymi klipami. Permutacja jest kontrolą negatywną, może wprowadzać nienaturalne wejścia; nie jest samodzielnym dowodem przyczyny. Przedziały niepewności liczyć po niezależnych operacjach, nie po zachodzących oknach.

### Co pozostaje z formalizmu Mori–Zwanziga

Formalizm motywuje rozdzielenie ewolucji obserwowanej części układu na wkład bieżącego stanu, pamięci i dynamiki nierozwiązanej. Istnieją już procedury uczenia takich operatorów i ich nieliniowych projekcji. [Lin et al., 2021](https://doi.org/10.1137/21M1401759), [Lin et al., 2023](https://doi.org/10.1137/22M1506146).

Nasza regresja wieloopóźnieniowa jest początkowo **modelem inspirowanym pamięcią**, nie automatycznie estymatorem dokładnego jądra MZ. Regularizacja ridge nie daje z definicji idempotentnej projekcji, a residual regresji nie jest automatycznie ortogonalną dynamiką MZ. Filmy chirurgiczne nie uzasadniają same przez się stacjonarności, ergodyczności, gaussowskości ani fizycznej relacji fluktuacja–dyssypacja. Mniejszy wymiar obserwacji niż pełnego stanu także nie wystarcza do stwierdzenia braku domknięcia: obserwowana część może ewoluować autonomicznie.

Miernik log-determinantów kowariancji można raportować jako określony przybliżony miernik liniowo-gaussowski; bez odpowiednich założeń nie nazywać go dokładną warunkową informacją wzajemną. Normalizacja cech lub SIGReg nie dowodzą spełnienia tych założeń.

## 5. Jak diagnoza miałaby sterować działaniem

Poniższe reguły są **hipotezami wyboru interwencji**, wymagającymi walidacji. Kilka problemów może występować jednocześnie; diagnoza musi dopuszczać wynik „nierozstrzygnięte”.

| Obserwacja na odłożonych danych development | Hipoteza | Działanie do sprawdzenia | Co może ją podważyć |
|---|---|---|---|
| Błędy i niezgodność modeli skupiają się w słabo pokrytym obszarze; więcej danych z tego obszaru pomaga w pilotażu | Niedobór przykładów | Wybór reprezentatywnych klatek lub nowych nagrań; anotacja | Modele zgodnie mylą się poza rozkładem; wysoka niepewność może wynikać z szumu |
| P2 poprawia wynik względem P1, a historia nie daje stabilnej dodatkowej korzyści | Zbyt ubogi predyktor | Wzbogacenie bieżącego modelu | Słaby P3 albo zły horyzont mogą ukryć przydatną historię |
| P3 poprawia wynik ponad P2 i kontrole pojemności | Brak użytecznego kontekstu w obecnym stanie | Krótkie okno lub stan pomocniczy z filtrów | Historia może kodować identyfikator filmu, rozkład faz albo zakłócenie |
| Zmierzony kanał poprawia wynik ponad historię | Brak istotnej obserwacji | Zachowanie kanału lub pozyskanie takiego pomiaru w nowych danych | Kanalizacja GT lub informacja niedostępna w zastosowaniu |
| Wysoki błąd, brak stabilnej korzyści z badanych rozszerzeń, duże niezgodności adnotatorów | Szum, niejednoznaczność lub nierozpoznana przyczyna | Kontrola anotacji, ocena niepewności; dodatkowy pomiar, jeśli wykonalny | Nie jest to dowód nieusuwalnego szumu; rodzina testowanych modeli była ograniczona |

Nie usuwać automatycznie wszystkich rozmyć, zasłonięć i rzadkich zdarzeń. Mogą być kluczowe dla zastosowania. Ocena jakości technicznej może kierować weryfikacją danych; nie powinna „poprawiać” testu przez usunięcie trudnych przypadków.

### Pierwsza wersja pętli

1. Zamrozić pulę, podział grupowy i reguły dostępności informacji. Wyliczyć cache cech jeden raz.
2. Zbudować mapę błędów i przewagi historii na train przez cross-fitting. Oceniać ją na stałym development; test pozostaje zamknięty.
3. Wyznaczyć kandydatów z ograniczeniem redundancji czasowej i reprezentatywności. Diagnostyka nadaje uzasadnienie, nie zastępuje pokrycia.
4. W wariancie A wybrać etykiety i zmierzyć efekt na detektorze. W wariancie B wybrać rodzaj interwencji według wcześniej zamrożonej reguły.
5. Ponownie ocenić zadanie i diagnostykę na tym samym odłożonym zbiorze. Zapisanie braku poprawy jest prawidłowym wynikiem.

W wersji startowej preferować prostą, jawną regułę nad sumą wielu arbitralnie ważonych score'ów. Parametry, progi i udział pokrycia wybrać na development przy równym budżecie strojenia konkurentów. Jeśli reguła wymaga modelu zadaniowego, jego aktualizacje i inferencja w każdej rundzie wchodzą do kosztu.

## 6. Najbliższa literatura i zakres możliwej nowości

| Źródło | Co już istnieje | Co wymagałoby odrębnego dowodu tutaj |
|---|---|---|
| [Abraham i Murphey, 2019](https://doi.org/10.1109/TRO.2019.2923880) | Aktywne uczenie dynamiki z reprezentacją Koopmana i informacją o zadaniu sterowania | Nie można zgłaszać jako nowości samego „Koopman + active learning”. |
| [EnKode, Li et al., 2024](https://doi.org/10.1109/LRA.2024.3486217) | Dobór punktowych pomiarów przepływu przez niezgodność zespołu modeli | Diagnoza brakującej historii i wybór rodzaju interwencji nie są jego mechanizmem. Dotyczy pomiarów pola, nie zakupu bboxów. |
| [Active operator inference, Uy et al.](https://doi.org/10.1137/21M1439729) | Aktywny dobór próbek trajektorii dla identyfikacji modelu przy szumie | Trzeba oddzielić poprawę modelu dynamiki od poprawy detekcji i kosztu ręcznych etykiet. |
| [Operator Inference of Non-Markovian Terms, Uy i Peherstorfer, 2021](https://doi.org/10.1007/s10915-021-01580-2) | Rekonstrukcja dynamiki z częściowych obserwacji i historii | Historia jako uzupełnienie modelu także ma precedensy. |
| [MZ-AE, Gupta et al., 2025](https://doi.org/10.1098/rspa.2024.0259) | Autoenkoder, przybliżenie Koopmana i poprawka pamięci w latencie | Sam moduł MZ w autoenkoderze nie stanowi nowości. Preprint pochodzi z 2023; publikacja czasopismowa z 2025. |
| [LRTD, Shi et al., 2020](https://doi.org/10.1007/s11548-020-02198-9) | Aktywny wybór klipów chirurgicznych o słabych zależnościach czasowych; etykieta ostatniej klatki | Rozpoznaje fazy, nie bboxy. Używa score'u zależności w sieci, a nie kontrastu predyktorów obecnego stanu i historii. Nie wystarczy zmienić nazwy score'u. |
| [COWAL, Wu et al., WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Wu_Correlation-Aware_Active_Learning_for_Surgery_Video_Segmentation_WACV_2024_paper.html) | Selekcja uwzględniająca reprezentację, korelację klatek i niepewność dla segmentacji wideo | Potrzebna kontrola silnej strategii łączącej reprezentatywność i niepewność. Przeniesienie na detekcję oznaczyć jako adaptację. |
| [Just et al., 2026](https://doi.org/10.1007/s11548-026-03718-9) | Strumieniowe AL w chirurgii, VeSSAL i normalizacja cech przez AAE | Samo ciągłe zbieranie danych ani normalizacja latentu nie wystarczają do nowej tezy. Autorzy badają binarne pytania klasyfikacyjne, nie pełną anotację bbox. |
| [Tang, Sloman i Kaski, AISTATS 2026](https://proceedings.mlr.press/v300/tang26d.html) | Dobór danych uwzględniający błędną specyfikację modelu, reprezentatywność i informatywność | Maksymalizacja samej niepewności nie ma ogólnej gwarancji poprawy. Ta teoria nie stanowi gotowego dowodu dla naszego detektora. |
| [Berrueta, Abraham i Murphey, rozdział książki, 2020](https://doi.org/10.1007/978-3-030-35713-9_16) | Związki Koopmana, uczenia aktywnego i eksperymentów w sterowaniu | Podstawa do dalszej lektury; w tej rundzie sprawdzono metadane i abstrakt, nie cały rozdział. |

**Wniosek z przeglądu, nie dowód pierwszeństwa:** najciekawsza luka do sprawdzenia to przewidywanie, *która interwencja* poprawi zadanie w danym reżimie obserwowalności. Nie wystarczy brak identycznego tytułu w wyszukiwarce. W szczególności dotychczasowe LRTD i COWAL ograniczają nowość prostego temporalnego rankingu, a MZ-AE ogranicza nowość dołożenia pamięci do latentu.

## 7. Tani program eksperymentalny przed dużym modelem

### Etap A — czy diagnoza rozróżnia znane przyczyny? CPU

Na syntetycznych trajektoriach przygotować pięć kontrolowanych przypadków: dobrze obserwowany układ ze słabym pokryciem próby; niejednoznaczna obserwacja wymagająca historii; obserwowany układ nieliniowy źle modelowany liniowo; brak zmiennej dostępnej dopiero w dodatkowym kanale; niezależny szum bez użytecznej historii. Przycięcie obserwacji i dodanie szumu muszą być jawne, z dostępną prawdą generatora. Rozdzielić scenariusze do ustalania reguły i nowe parametry generatorów do oceny.

Porównać działania: dodatkowe przykłady, większa klasa funkcji bieżącej, historia, dodatkowy kanał oraz brak działania. Dla każdego zapisać zmianę błędu i koszt. Sprawdzić trafność wskazania najlepszego dostępnego działania oraz regret względem najlepszego działania zmierzonego eksperymentalnie. Oracle służy wyłącznie do oceny, nie do decyzji. Raportować też macierz pomyłek diagnozy i częstość wyniku nierozstrzygniętego. Nie udawać realnego kosztu klinicznego: pokazać oddzielnie liczbę próbek, cech, parametrów i czas CPU; analizę ceny względnej działań oznaczyć jako scenariusz.

Kontrola przeciwko kolapsowi: reprezentacja może świetnie przewidywać prawie stałą cechę i jednocześnie utracić zmienną potrzebną do zadania. Dlatego dodać niezależny target zadaniowy. Spadek residualu bez zachowania tej informacji nie zalicza eksperymentu.

**Warunek przejścia:** reguła powinna rozróżniać przynajmniej niedobór danych i brak historii na niewidzianych parametrach oraz nie wybierać uporczywie szumu. Jeśli tego nie robi, wynik z wideo nie uzasadni narracji o diagnozowaniu przyczyny.

### Etap B — czy sygnał pomaga dobierać etykiety? Mały YOLO

Zacząć od zamrożonego DINO i YOLOv8n według optymalizacji dokumentu 24. Nie trenować od początku JEPA ani dużego detektora. Warianty selekcji:

| ID | Strategia | Izolowany kontrast |
|---|---|---|
| B0 | Losowanie unikalnych źródeł | Podstawowy punkt odniesienia |
| B1 | Próbkowanie równomierne po czasie i filmach | Kosztowny selektor musi uzasadnić się ponad tę kontrolę |
| B2 | Pokrycie zamrożonych cech | Czy wystarcza różnorodność? |
| B3 | Pokrycie + niepewność detektora | Standardowy konkurent AL |
| B4 | Pokrycie + residual P2 | Czy wystarcza duży błąd predykcji? |
| B5 | Pokrycie + przewaga historii P3 nad P2 | Czy rozróżnienie rodzaju błędu wnosi coś ponad residual? |

W B3 wybrać i zamrozić konkretny score detekcyjny obejmujący obecność/lokalizację obiektu. Sama entropia klas w zadaniu jednoklasowym nie wystarczy, a wysoka pewność błędnego tła może ukrywać całkiem pominięte obiekty. Nie deklarować BALD bez odpowiedniej procedury przybliżenia posterioru. Adaptację COWAL albo innego silnego selektora detekcyjnego przygotować jako dodatkową kontrolę publikacyjną, nie nazywać ogólnego B3 wierną implementacją tej pracy. LRTD jest dodatkową kontrolą mechanizmu temporalnego; przeniesienie z faz na bbox wymaga osobnej adaptacji i kosztu.

Przykładowy mały budżet: wspólne `L0 = 500` unikalnych klatek, następnie dwie rundy po 1000 nowych, czyli punkty `500, 1500, 2500`. Przy mniejszej dostępnej puli ustalić wykonalne liczby przed startem. Każdy query obejmuje anotację wszystkich obiektów docelowych na wybranej klatce; brak obiektu też jest wynikiem zapytania. Kontekst wideo może być dostępny bez dodatkowych etykiet, ale jego przetworzenie kosztuje.

Jeden pilotażowy seed to **1 wspólny trening początkowy + 6 strategii × 2 rundy = 13 treningów**. Potwierdzenie z trzema sparowanymi seedami to **3 + 6 × 2 × 3 = 39 treningów**, przy założeniu identycznego wspólnego początku w obrębie seeda. Pilotaż może wejść do 39 tylko jeśli po nim nie zmieniono protokołu; w przeciwnym razie 13 jest osobnym kosztem eksploracji. Przy zamrożonym protokole można współdzielić identyczne wyniki, ale nie traktować ich jako niezależnych powtórzeń.

B4/B5 z predyktorami dopasowanymi do całego `U` mają stały ranking; zaznaczyć to w tabeli wyników. Dopiero B3 lub reguła zależna od aktualnego detektora są adaptacyjne. Jeśli później łączymy diagnozę z niepewnością, kontrastem jest **B3 kontra B3 + diagnoza** przy tej samej reprezentatywności. Takie połączenie stanowi dodatkowy warunek, nie ukryty składnik B5.

Symulacja AL na istniejącym zbiorze: etykiety `U \ L_r` są zamaskowane dla selektora i procesu strojenia; ujawnia je wyłącznie mechanizm query. Nie używać ich do GT ROI, histogramu klas, strat nadzorowanych ani naprawiania rankingu. Początkową próbę i limity po filmach wyznaczać bez tych etykiet. Zbiory z niedopełnioną anotacją nie stają się poprawnym oracle przez samo maskowanie.

Do porównania zbiorów po każdej rundzie trenować detektor od tej samej ogólnej inicjalizacji, przy jednakowej liczbie rzeczywistych kroków optymalizatora w danym punkcie budżetu. Warm-start traktować jako osobny wariant operacyjny. Augmentacje, wybór checkpointu i strojenie są wspólne w obrębie architektury. Czas selekcji, inferencji puli, cache'u i wszystkich wcześniejszych rund wchodzi do kosztu końcowego. Równy K nie gwarantuje równego czasu anotacji; do tezy o oszczędności czasu potrzebny pomiar pracy anotatora, wraz z QA.

### Etap C — czy wskazane uzupełnienie stanu rzeczywiście pomaga?

To osobny eksperyment mechanistyczny. Minimum: dwa ustalone zbiory danych (kontrola i selekcja) × dwa warianty modelu (obecna klatka i kontekst) × trzy seedy = **12 treningów**. Ta macierz sprawdza interakcję selekcji i kontekstu; sama nie rozstrzyga wszystkich diagnoz z etapu A. Zbiory muszą zawierać etykiety klatek kotwiczących, a kontekst pochodzić wyłącznie z tego samego splitu.

Najpierw zastosować tani predyktor zadania na cache'u lub mały moduł temporalny. Detektor pojedynczej klatki nie zacznie korzystać z historii tylko dlatego, że historia posłużyła do selekcji. Zmiana architektury wymaga kontroli o podobnej pojemności bez dodatkowego kontekstu, pomiaru opóźnienia i kosztu. Przewaga historii w przewidywaniu globalnego latentu może dotyczyć tła lub fazy operacji, a nie lokalizacji końcówki.

Dla filtrów E-AQ porównać surowe opóźnienia, filtry uczone i losowe o porównywalnej liczbie cech; zdefiniować inicjalizację, rozgrzewkę oraz reset na granicy filmu. Dla tezy o nowym kanale potrzebny jest osobny test z faktycznie dostępnym pomiarem. Nie dopisywać takiej tezy na podstawie samego eksperymentu z historią.

## 8. Pomiary i warunki publikacji

Główny wynik detekcyjny: COCO AP@[.50:.95] na wszystkich obrazach odłożonych operacji, z AP50, AP75, AR i analizą małych końcówek. Predykcje puste i fałszywe alarmy pozostają w ocenie. Prógi użytkowe, NMS i checkpoint wybierać na walidacji. W AL raportować krzywą AP względem skumulowanej liczby anotacji oraz względem całkowitego kosztu; pole pod krzywą liczyć na wspólnej, jawnej osi budżetu, z tą samą interpolacją i zakresem.

Oddzielić zmienność seedów od niepewności próby pacjentów/operacji. Dla par metod można bootstrapować operacje, przeliczając AP na połączonych predykcjach w każdej replice, z konsekwentnym przemianowaniem powielonych identyfikatorów obrazów. Nie uśredniać AP po klatkach i nie traktować tysięcy okien jako tysięcy niezależnych pacjentów. Trzy seedy to praktyczny początek, nie gwarancja mocy statystycznej.

Do mocnej tezy B potrzeba wykazać łącznie:

1. Diagnoza przewiduje korzystny rodzaj interwencji poza danymi, na których ustalono regułę.
2. Wybór danych poprawia zadanie względem uniform, pokrycia, niepewności i prostego residualu przy kontrolowanym koszcie.
3. Efekt nie pochodzi wyłącznie ze zwiększenia kontekstu, parametrów, liczby kroków lub dostępu do ukrytych etykiet.
4. Wynik utrzymuje się przy dłuższym treningu i przynajmniej na drugiej rodzinie detektora; zakres twierdzenia odpowiada rzeczywiście sprawdzonym modelom i zbiorom.

Jeżeli poprawia się tylko predykcja latentu, powstaje wynik o modelowaniu dynamiki. Jeżeli poprawia się tylko szybkie uczenie, wniosek dotyczy budżetu krótkiego treningu. Jeżeli B5 przegrywa z B4, nie ma podstaw twierdzić, że właśnie diagnoza pamięci uzasadnia selekcję. Jeżeli kontekst pomaga na każdym zbiorze tak samo, jego korzyść nie dowodzi trafności polityki kuracji.

## 9. Relacja do ponownej publikacji i recenzji

[Mapa uwag recenzentów](../notatki/MAPA_UWAG_RECENZENTOW.md) pozostaje obowiązującym zestawem problemów do rozwiązania w poprawianej pracy. Ten kierunek badawczy ich nie zastępuje.

| Problem z recenzji | Jak uwzględnić w nowym kierunku |
|---|---|
| R2-m1: nowość kombinacji znanych bloków | Porównać mechanizm i dostęp do informacji z MZ-AE, LRTD, COWAL, EnKode; ostrożnie sformułować wkład w wybór interwencji |
| R1-2, R2-10: podział i historia użycia testu | Split po pacjencie/operacji przed cache'em i selekcją; nowy niezależny test dla potwierdzenia, jawna rola starego benchmarku |
| R1-7, R1-10: izolacja selekcji i transfer | Stały detektor wewnątrz kontrastu; osobna macierz kontekstu i osobny transfer zamrożonej reguły |
| R2-8, R2-m2: zasoby i wielkość zbioru | Wspólne punkty K, rzeczywiste kroki treningu i całkowity koszt, nie tylko liczba epok |
| Powtórzenia i spójna ewaluacja | Surowe wyniki seedów, predykcje, wspólny evaluator i analiza po operacjach; bez założonych z góry wartości p lub dużego zysku |

**Rekomendacja organizacyjna:** poprawianą pracę DETR/YOLO prowadzić według 22–24. Równolegle rozpocząć tutaj od etapu A na CPU. Tylko jeśli diagnoza przechodzi kontrolę znanych przyczyn, uruchamiać osobny mały pilotaż B. Pełną politykę uzupełniania stanu potraktować jako możliwą następną publikację, zamiast uzależniać od niej ponowne złożenie poprzedniej pracy.

## 10. Źródła, zakres odczytu i stan wykonania

Nowy [przegląd Undermind](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Diagnostyka%20niedomkniecia%20latentu%20jako%20sterowanie%20doborem%20danych%20i%20active%20learning) zakończył się 161 rekordami. Odczytano opis i pierwsze 15 abstraktów/metadanych, a dodatkowo wyszukano konkretne precedensy. Trzy prace — EnKode, Regression-Based Projection i LRTD — poddano lekturze pełnotekstowej przez narzędzie Undermind. Pozostałe wnioski bibliograficzne mają zakres wskazany w [rejestrze źródeł](evidence_active_learning_2026-09-11/zrodla.json). Wynik czytelnika PDF wymaga krytycznej interpretacji; nie przyjęto jego utożsamienia spadku MSE z informacją wzajemną ani automatycznego niedomknięcia przy redukcji wymiaru.

Linki do pobrania zapisano w kolekcji artykułów przez istniejący zbieracz: [lista tematyczna — 11 nowych adresów](F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/linki/diagnostyka-latentu-active-learning-i-uzupelnianie-stanu-2026-09-11.txt). Jeden adres pominięto jako obecny w innej liście. [Pełny zestaw adresów wejściowych](evidence_active_learning_2026-09-11/linki_wejsciowe.txt) zachowuje wszystkie 12 pozycji. Nie wykonano OCR, pobierania PDF-ów na lokalny dysk, nowych treningów ani implementacji proponowanej polityki. Nie zmieniono materiałów grantu ani wcześniejszych wyników.
