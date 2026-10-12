# Plan na przyszłe eksperymenty: poprawa równań Langevina i Koopmana

Data zapisu: 2026-10-01 14:06:21 (Europe/Warsaw).
Ostatnia modyfikacja: 2026-10-01 15:28:17 (Europe/Warsaw).
Status: plan badawczy po pierwszym lokalnym pilotażu identyfikacji. Nie uruchomiono nowego treningu ani Edenu.

## Cel i uzgodnione założenie

Sprawdzić, czy jawne połączenie historii stanu reprezentacji z historią klatek treningowych i ich wpływu poprawia prognozę zmian siły tożsamości narzędzia na 500 aktualizacji do przodu. Prognoza w chwili t ma korzystać wyłącznie z informacji dostępnej do t. Rzeczywiste przyszłe paczki, augmentacje, gradienty i stany wag nie są wejściem prognozy.

To zapis ustaleń z rozmowy 1 października 2026. Użytkownik zlecił utworzenie folderu planów i zapis tego planu. Parametry proponowanych pomiarów poniżej są projektem eksperymentu, a nie już wykonanymi testami.

## Co zmierzono dotychczas

- Kampania pętli 4.3: 10 ziaren 47-56, sześć ramion, 14 obrotów po 500 aktualizacji, 134 nowe klatki na obrót w ramionach dodających dane. Kierunek tożsamości wyznaczono po rozgrzewce około 21 epok, następnie zamrożono.
- LDA wyznacza kierunek w po standaryzacji reprezentacji względem negatywów. Dla każdej sondy z jest średnią 768-wymiarowych aktywacji tokenów narzędzia, a s=((z-mu)/sd) dot w. Prognozowano s, nie obrót w ani cały wektor z.
- Prognozy na 500 aktualizacji dostawały historię s. Nie dostawały osobnego opisu historycznych paczek ani ich gradientów. Wpływ klatek był obecny pośrednio przez zmianę sieci i s.
- Nowy Langevin z pamięcią: średnia trafność znaku 74,2 procent w głównym ramieniu i 75,4 procent po wszystkich ramionach. Ocena znaku obejmowała połowę sond z największą rzeczywistą zmianą w obrocie.
- Starszy selektor Koopman/Langevin: odpowiednio 70,4 i 71,0 procent. Nie są to wyniki samego Koopmana: selektor wybierał model na podstawie testu wstecznego.
- Mediana redukcji błędu kwadratowego względem braku zmiany w głównym ramieniu: nowy Langevin 28,7 procent, starszy selektor 16,2 procent. Cele nie były identyczne: nowy model oceniano na końcowych zapisach, starszy na średnich blokowych. Przed dalszym porównaniem trzeba to ujednolicić.
- Gęsta historia: jeden bieg, ziarno 47, 1001 stanów 12 obrazów, 600 osobnych tokenów po 768 wartości, float32. Dane około 22,1 GB pozostawiono na Edenie. Prognoz na tych pojedynczych tokenach jeszcze nie wykonano.
- Wariant jednego kroku uwzględniający wpływ paczki i pamięć AdamW był osobnym pomiarem. Nie dowodzi poprawy prognozy z samej przeszłości na 500 kroków.

## Co wynika z porównania planu z kodem

- `sila.wektor` zamraża nie tylko kierunek w, ale także mu i sd. Kalibracja używa średniej ostatnich pięciu zapisów rozgrzewki. Wszystkie trzy wielkości muszą być wspólne dla wariantów prognozy w danym biegu. Nie kalibrować ich z przyszłości każdego okna.
- W rzeczywistej ścieżce `sila.prognozy` stan jest skalarem s na sondę. `prognoza.py` ma również ogólne funkcje PCA dla latentu, ale pętla 4.3 nie używa ich do prognozy siły. Nowy model na całym z lub tokenach byłby dodatkową ablacją, a nie wiernym odtworzeniem tej kontroli.
- Stary Koopman jest regresją opóźnionych przyrostów, z pamięcią czterech bloków po 50 aktualizacji. Stary Langevin dodaje poziom stanu i osobną stałą sondy. `prognoza_gle.py` przewiduje zmianę bezpośrednio: 20 opóźnionych przyrostów co 10 aktualizacji i odchylenie od kroczącej średniej 50 zapisów. Współczynniki dzieli między sondami. Kod nie wymusza dodatniego hamowania ani stabilności; nazwa Langevin nie stanowi dowodu fizycznego modelu.
- `petla4.py` wydaje obie prognozy przed wyborem nowych klatek. `pchniecie_dodanych` powstaje dopiero po tym wyborze i opisuje średni hipotetyczny krok na grupach nowych klatek, nie paczki rzeczywiście wykonane podczas kolejnych 500 aktualizacji. Nie wolno dołączyć tego pola do wcześniej wydanej prognozy. Osobne prognozowanie po dołączeniu danych wymaga nowego momentu odcięcia i tej samej informacji dla wszystkich modeli.
- `Sprawdzian` bada pierwsze 10 kroków obrotu i potem co 50-ty, a do `OBROT.json` oddaje podsumowania. Sam raport nie zawiera kompletnej historii pchnięć per sonda i per krok. Wagi po obrocie nie zastępują brakującego dziennika paczek.
- Parametry kampanii (14 obrotów i odziedziczona rozgrzewka 21 epok) pochodzą z WYNIKI.md; domyślne parametry samodzielnego `petla4.py` to 7 obrotów po rozgrzewce 1000 aktualizacji. W odtworzeniu korzystać z metadanych biegu, nie tylko domyślnych stałych kodu.

Gęsta historia obejmuje wczesne uczenie 0..1000, a nie etap po rozgrzewce kampanii. Zawiera zmiany buforów BatchNorm w trybie eval. Zapis wag co 10 kroków i stanu AdamW co 100 kroków nie daje pełnej historii wpływu wszystkich paczek. Wyniki kierunków SVD dotyczą rozdzielania narzędzia i tła, nie jakości prognozy. Przy kalibracji w chwili t_cal, pierwszej dojrzałej parze uczącej na H=500 i następnym pełnym oknie oceny potrzeba co najmniej t_cal+1000 aktualizacji, a zwykle więcej dla wielu par. Obecne 1000 kroków nie daje takiego testu po kalibracji t_cal>0. Pilotaż może sprawdzić krótsze horyzonty lub wcześniej zamrożoną kalibrację, ale potwierdzenie H=500 wymaga dłuższej historii.

## Pytanie badawcze i hipoteza

Czy informacja o historycznych wymuszeniach pozwala odróżnić własną dynamikę stanu od wpływu dopływających danych i poprawia prognozę poza próbką?

Hipoteza: modele wykorzystujące stan oraz wpływ historycznych paczek mają mniejszy błąd przyszłej zmiany s niż te same rodziny modeli z samą historią s. Poprawa nie jest z góry gwarantowana. Nieznane przyszłe wymuszenia pozostają źródłem niepewności.

## Etap 1. Audyt dostępnych danych

- [ ] Sprawdzić, czy zapisano dla każdej aktualizacji identyfikatory użytych klatek, augmentacje, numery kroków, s przed i po kroku oraz stan lub mierzalny wpływ optymalizatora.
- [ ] Oddzielić embeddingi stałych sond od embeddingów rzeczywistych paczek treningowych. Pierwsze nie zastępują drugich.
- [ ] Sprawdzić zgodność czasu: paczka kroku t powoduje przejście ze stanu przed krokiem do stanu po kroku. W prognozie nie używać wpływu paczki, która jeszcze nie została wykonana.
- [ ] Jeśli brakuje zapisów paczek lub wpływów, przygotować osobny przyszły bieg telemetryczny. Nie odtwarzać ich jako faktów z samych trajektorii sond.

## Etap 2. Stan i historyczne wymuszenia

Zachować zamrożony kierunek LDA oraz tę samą definicję s dla wszystkich wariantów. Historię rozszerzać stopniowo:

1. Stan podstawowy: s oraz opóźnione przyrosty s.
2. Historyczna paczka: zagregowany opis reprezentacji klatek z faktycznie wykonanej paczki, liczba przykładów, strata przed aktualizacją oraz parametry augmentacji, jeżeli zostały zachowane.
3. Historyczny wpływ paczki: jej zmierzone lub wyliczone pchnięcie s, z oddzielnym zapisem wpływu pamięci optymalizatora. Nie utożsamiać opisu obrazu z jego wpływem na aktualizację.

Wymuszenia historyczne mają być dodatkowym wejściem, a nie zamiennikiem stanu. Liczba cech i koszt ich obliczania muszą zostać zapisane.

### Kontrakt czasowy i definicja wpływu

Stan s_k jest mierzony po ukończeniu aktualizacji k; paczka B_k odpowiada przejściu ze stanu k-1 do k. Prognoza wydana po aktualizacji t może użyć B_k i jej wpływu wyłącznie dla k <= t. Jeżeli telemetria korzysta z innej numeracji, konwersję zapisać jawnie przed łączeniem tabel.

Dla ukończonego kroku zapisać identyfikatory klatek, faktycznie zrealizowane augmentacje, cechy paczki sprzed aktualizacji, lr, liczbę przykładów, identyfikator wersji zbioru i sond oraz osobne wektory wpływu paczki i pamięci AdamW na sondy. Historyczne gradienty i wagi mogą służyć do pomiaru tych wpływów podczas rejestracji; do modelu przekazywać ich zapisane podsumowania. Prognoza nie odczytuje checkpointów ani gradientów z kroków po t. Pełny rzeczywisty przyszły ciąg pozostaje wyłącznie etykietą oceny.

Przybliżone pchnięcie paczki i pamięci wyznaczać w tej samej konwencji perturbacji co `prognoza_krok.py`, zapisując konwencję mianownika AdamW oraz koszt dodatkowych przebiegów. Reszta `delta_s - pchniecie_paczki - pchniecie_pamieci` obejmuje m.in. nieliniowość, zanik wag i bufory BatchNorm. Nie nazywać jej automatycznie szumem losowym. W pilotażu osobno zmierzyć ruch od buforów przy stałych wagach oraz błąd addytywnego przybliżenia. Diagnostyka nie może zmieniać buforów, generatorów losowania ani właściwej aktualizacji uczenia.

Sam przyrost `s_k-s_(k-1)` jest już informacją w historii stanu. Dodanie go ponownie jako „wpływu paczki” nie testuje hipotezy. Wymagane są osobne deskryptory paczki lub jawnie zmierzone składniki wpływu. Brakujący pomiar oznaczać maską; nie interpolować go z późniejszych kroków.

## Etap 3. Osobne ramiona prognoz

- [ ] Koopman z samą historią stanu.
- [ ] Koopman ze stanem i historycznymi wymuszeniami.
- [ ] Langevin z pamięcią, z samą historią stanu.
- [ ] Langevin z pamięcią, ze stanem i historycznymi wymuszeniami.
- [ ] Kontrole: brak zmiany, stały historyczny dryf i prosty model autoregresyjny.
- [ ] Raportować każdą rodzinę osobno. Ewentualny selektor modeli stanowi dodatkowe ramię, nie zastępuje wyników Koopmana i Langevina.

Porównać modele na tym samym celu: s(t+H)-s(t), z identyczną rozdzielczością zapisu. Dopasowanie i dobór regularyzacji tylko w przeszłości.

### Ablacje i proponowane równania

Dla każdej rodziny porównać cztery wejścia: stan; stan + opis paczki; stan + wpływ paczki; stan + opis + wpływ. Dopiero potem sprawdzić dodatkowy wkład pamięci optymalizatora. Każda para porównań ma te same punkty historii, horyzont, liczbę sond, zakres strojenia i regułę kalibracji. Wpływ jest zależny od sondy; deskryptor wspólnej paczki nie może przypadkiem zostać potraktowany jako wiele niezależnych paczek.

Proponowane modele dyskretne, do sprawdzenia empirycznego:

```text
v_k = s_k - s_(k-1)
x_k = s_k - srednia_kroczaca_z_danych_do_k
u_k = [opis_B_k, pchniecie_paczki_k, opcjonalnie_pchniecie_pamieci_k]

Koopman/ARX z kontrolą:
v_(k+1) = suma_j A_j v_(k-j) + suma_j C_j u_(k-j) + a
s_(k+1) = s_k + v_(k+1)

Langevin z pamięcią, prognoza bezpośrednia:
s_(k+H)-s_k = suma_j K_(H,j) v_(k-j) - kappa_H x_k
               + suma_j D_(H,j) u_(k-j) + a_H
```

Pierwsza postać rozszerza dotychczasową regresję przyrostów o kontrolę (ARX); bez dodatkowego uzasadnienia nie przypisywać jej własności dokładnego operatora Koopmana. Jeżeli zastosuje się nieliniowy lifting stanu, dobór jego wymiaru i PCA wykonuje się wyłącznie na danych uczących do danego odcięcia. Druga postać rozszerza działającą prognozę `prognoza_gle.py`; H i pamięć podawać także w liczbie aktualizacji. Wyraz stały musi mieć tę samą konwencję w modelu z wymuszeniem i w kontroli bez niego.

Wariant iteracyjnego Langevina ma H=1 i własny model przyszłych u; raportować go osobno od prognozy bezpośredniej H=500. Hamowanie nieujemne i ograniczenie stabilności dynamiki iteracyjnej to osobne ablacje wybierane na historycznej walidacji. Testować wieloetapowy rollout pod kątem eksplozji norm i nierealistycznego powrotu do stałego poziomu. Nie traktować dopasowanego kappa_H jako potwierdzonej stałej fizycznej.

## Etap 4. Nieznane przyszłe klatki

W chwili t wykorzystać wyłącznie aktualnie dostępny zbiór i jego znaną regułę losowania. Z historii oszacować średni wpływ paczek, rozrzut oraz ewentualną zależność od stanu. Prognozować oczekiwaną zmianę i rozkład możliwych zmian, zamiast wstawiać rzeczywiste przyszłe wymuszenia.

Dwa warianty porównania: prognoza bezpośrednia na docelowy horyzont oraz prognoza iteracyjna z modelowanymi wymuszeniami. Symulowane wymuszenia nie mogą pochodzić z rzeczywistych przyszłych paczek ani przyszłych wag.

Prognoza po zmianie składu zbioru jest innym warunkiem niż prognoza przy stałym zbiorze. Najpierw ocenić stały przepis losowania; następnie osobno warunek dokładania klatek w pętli. Każda prognoza musi wskazywać, jaką politykę dalszego uczenia zakłada.

## Etap 5. Ocena

- [ ] Horyzont główny: 500 aktualizacji; diagnostyczne: 1, 10, 50 i 100.
- [ ] Podział czasowy kroczący: dopasowanie do t, ocena na przyszłych stanach. Przy uczeniu prognozy bezpośredniej wolno używać tylko par, których cały cel jest już znany w chwili t.
- [ ] W ocenie między ziarnami odkładać całe biegi. Dopasowanie kierunku LDA z własnej rozgrzewki biegu jest jawnie dozwoloną kalibracją, a przyszła historia biegu nie jest dostępna.
- [ ] Miary wartości: MAE, RMSE oraz 1-MSE_model/MSE_brak_zmiany. Nie nazywać tej ostatniej procentem trafnych prognoz.
- [ ] Miary znaku: zgodność wzrostu/spadku dla wszystkich sond, osobno poprzednia konwencja największej połowy zmian, odsetek wzrostów i spadków oraz kontrola zawsze przewidująca częstszy znak, wybrany z przeszłości.
- [ ] Dla prognoz probabilistycznych: pokrycie i szerokość przedziałów oraz kalibracja. Wysoka szerokość nie może zastępować trafności.
- [ ] Różnice porównywać parami na tych samych ziarnach i momentach prognozy. Zależnych sond i nakładających się okien nie traktować jako niezależnych powtórzeń.
- [ ] Przed pełną kampanią ustalić liczbę ziaren i kryterium minimalnej użytecznej poprawy. Pierwszy bieg jest diagnostyczny, a nie potwierdzający.

### Protokół pilotażu i warunek rozpoczęcia kampanii

- [ ] Odtworzyć kontrolę z `prognoza_gle.py` na zapisach co 10 aktualizacji i wyliczyć każdy model na tym samym celu końcowym. Średnie blokowe zachować jako osobny cel pomocniczy; wyrównanie etykiet nie polega na porównaniu dwóch istniejących liczb z raportu.
- [ ] Ustalić moment kalibracji, wymagane minimum dojrzałych par i pierwszą chwilę dopuszczalnej prognozy. Przy braku historii zwracać jawny status oraz wynik kontroli braku zmiany; takich przypadków nie liczyć jako prognoz wyuczonego modelu.
- [ ] Każdy podział walidacyjny od nowa dopasowuje skalowanie, PCA, regularizację, pamięć, selektor i model rozkładu wymuszeń. Para z etykietą kończącą się po odcięciu nie należy do treningu. Przyszły zestaw sond ani filmy testowe nie służą do wyboru cech.
- [ ] Sprawdzić niezmienność prognozy po usunięciu lub podmianie całego sufiksu po t (paczki, gradienty, wagi, stany, etykiety) oraz zgodność prognozy z samego prefiksu i z pełnego pliku przy tym samym odcięciu. Osobno sprawdzić przesunięcie paczki o jeden krok i brak dokładnej historycznej augmentacji.
- [ ] Dodać kontrolę permutacji deskryptorów i wpływów wyłącznie wewnątrz historycznego fragmentu o tej samej polityce/lr. Utrata poprawy po zerwaniu związku paczka-przejście wspiera interpretację, lecz sama nie dowodzi identyfikacji przyczynowej.
- [ ] W modelu przyszłego wymuszenia sprawdzić średnią, wariancję i autokorelację reszt w historycznej walidacji. Rozkład warunkowy od stanu oraz próbki blokowe to proponowane alternatywy dla stałego średniego pchnięcia. Nie podstawiać przyszłych cech paczek nawet wtedy, gdy użyte klatki były już w zbiorze w t. Nie odtwarzać przyszłego losowania ze stanu generatora rzeczywistego treningu.

Proponowany próg użytkowy do zamrożenia po pilotażu: co najmniej 5 procent względnej redukcji MSE na H=500 wobec odpowiednika tej samej rodziny bez wymuszeń, przy dodatniej umiejętności względem braku zmiany i przewadze nad najlepszą prostą kontrolą. Nie mylić tego z pięcioma punktami procentowymi umiejętności. Kampania potwierdzająca: 10 nowych ziaren, proponowane 57-66, bez dostrajania na ich przyszłych wynikach. Historyczna adaptacja do t jest dozwolona przy wcześniej zamrożonej procedurze. Osobny wariant przenoszenia między biegami odkłada cały bieg i nie miesza go z adaptacją.

Najpierw uśrednić straty po sondach i z góry ustalonych momentach prognozy każdego biegu, następnie porównać modele parami po ziarnach. Raportować również medianę i wszystkie wyniki per ziarno. Przedziały ufności dla różnicy straty wyznaczać na poziomie całych ziaren, nie sond/tokenów; potwierdzenie wymaga przedziału wskazującego przewagę. Osobno raportować fazy lr, stały zbiór i dokładanie danych. Z góry ustalić podstawowe punkty H=500 bez nakładania okien; gęstsze okna traktować jako diagnostykę. Długość biegu ma zapewniać kilka okien oceny po pierwszej legalnej prognozie.

Artefakty przyszłego eksperymentu: `PLAN.md` (zamrożony protokół), `kod/telemetria.py` (zapis przejść), `kod/dane_prognozy.py` (odcięcia i cechy), `kod/modele.py` (ramiona), `kod/ocena.py` (miary), testy odcięć i braku ingerencji w trening, `metadata.json`, dziennik paczek i wpływów, `PROGNOZY.jsonl` (t, H, model, dostępne dane, wynik i przedział), `METRYKI.json` oraz `WYNIKI.md`. W metadanych zapisać wersję kodu, ziarna, sprzęt, lr i politykę zbioru, rozdzielczość czasu, koszt telemetrii oraz identyfikator podziału. To lista do przyszłej realizacji; żaden z tych nowych plików nie został teraz utworzony.

## Co potwierdzi lub obali hipotezę

Potwierdzenie wymaga powtarzalnego zmniejszenia błędu na 500 aktualizacji poza dopasowaniem względem modelu tej samej rodziny bez historycznych wymuszeń oraz względem prostych kontroli. Sama lepsza trafność znaku lub lepsze dopasowanie historii nie wystarcza.

Brak poprawy poza dopasowaniem, poprawa tylko na jeden krok, przewaga wyłącznie po użyciu przyszłych paczek albo zysk znikający po wyrównaniu celów nie potwierdzają hipotezy. Zachować i opisać takie wyniki.

## Kolejność realizacji i artefakty

1. Audyt istniejącej telemetrii i ujednolicenie dotychczasowych miar.
2. Osobny eksperyment w experiments/maly_vit_prognoza_stan_i_historyczne_paczki, z własnym PLAN.md, README.md, kodem, metadanymi i raportami. Nie dopisywać nowych pomiarów do zakończonej kampanii.
3. Mała próba telemetrii i kontroli przyczynowości: zmiana danych po t nie może zmieniać prognozy wydanej w t.
4. Pilotaż prognoz bez uruchamiania pętli SD.
5. Kampania wieloziarnowa dopiero po pilotażu; następnie osobny test wpływu lepszej prognozy na dobór klatek i uczenie detektora.

Pierwotnie zapisano tylko ten plan. Po rozszerzeniu zlecenia wykonano lokalny pilot w `../experiments/maly_vit_identyfikacja_langevin_koopman/`, na istniejących nagraniach s47/s48. Nie uruchomiono nowego treningu ani Edenu i nie wykonano commitów.

## Źródła do realizacji

- ../experiments/maly_vit_petla_sd_rownanie/WYNIKI.md
- ../experiments/maly_vit_petla_sd_rownanie/kod/sila.py
- ../experiments/maly_vit_petla_sd_rownanie/kod/prognoza.py
- ../experiments/maly_vit_petla_sd_rownanie/kod/prognoza_gle.py
- ../experiments/maly_vit_petla_sd_rownanie/kod/prognoza_krok.py
- ../experiments/maly_vit_petla_sd_rownanie/kod/petla4.py
- ../experiments/maly_vit_ruch_czastek_gesta_historia/README.md
- ../experiments/maly_vit_ruch_czastek_gesta_historia/WYNIKI.md
- ../docs/26_PIPELINE_TOZSAMOSC_PROGNOZA_SD.md

## Aktualizacja po lokalnym pilotażu identyfikacji

W istniejących lokalnych nagraniach znaleziono po 3001 stanów, deskryptory rzeczywistych paczek i pchnięcia paczki/AdamW. Ich przyczynowe przesunięcie to wiersz paczki k prowadzący ze stanu k do k+1; prognoza t dostaje wyłącznie paczki k<t. Zbadano H=10,50,100,500, z wyborem na walidacji s47 i transferem na s48. Dodatkowa informacja nie poprawiła H=500 względem działającego GLE. Wynik strojenia samego jądra na diagnostycznych oknach s48 to 22,65 procent redukcji MSE wobec 22,64 procent oryginału; na późnym punkcie s47 jest gorszy od braku zmiany. Nie potwierdzono hipotezy. Drugi etap był jawnie eksploracyjny po obejrzeniu pierwszego testu. Wyniki, źródła pierwotne i zakres weryfikacji Lean/SymPy: [WYNIKI.md](../experiments/maly_vit_identyfikacja_langevin_koopman/WYNIKI.md), [MODEL.md](../experiments/maly_vit_identyfikacja_langevin_koopman/MODEL.md). Dwa ziarna nie zastępują przyszłej kampanii wieloziarnowej.

## Aktualizacja 2026-10-01 14:53:54: bliższy zapis i składowe tożsamości

Priorytet użytkownika: prognoza jednego bliższego zapisu/checkpointu, bez upierania się przy 500 aktualizacjach. Każdy horyzont podawać w liczbie aktualizacji i interwale telemetrii: główna kampania miała sondy co10 i wagi po500, a istniejące nagrania siły mają stany co1. H1 nie jest tym samym zadaniem co następne wagi po500.

Cel80procent pozostaje celem do sprawdzenia, nie osiągnięciem ani gwarancją. Rozdzielić80procent zgodności znaku na wszystkich sondach, zbilansowaną trafność oraz80procent redukcji MSE względem braku zmiany; żadna z tych miar nie zastępuje pozostałych. Maska największej połowy zmian jest wyłącznie pomocnicza.

Lokalny pilot H1/3/5/10/20/50 na istniejących s47/s48: trening s47 z końcem etykiety<=1800, walidacja startów1800..2000 co5, refit po wyborze do2500, transfer diagnostyczny s48 starty2100..2950 co5. Model wybrać wyłącznie na walidacji. Dane s48 już obejrzano wcześniej, więc cały pilot jest eksploracyjny. Zachować poprzednie wyniki H500; nowe artefakty w podkatalogu `wyniki/krotki_horyzont/`. Kontrole: brak zmiany, ostatni przyrost/dryf, większość znaku z przeszłości, prosty AR/GLE. Modele: lokalny dryf/lifting Koopmana w obserwowanym stanie + jądro pamięci GLE + historyczne wpływy paczek i AdamW. Nie obiecywać globalnej dokładnej linearyzacji. Każda paczka powodująca cel pozostaje nieznana; pole nowa[t] prowadzące t->t+1 nie jest wejściem w t. Wykorzystać wyłącznie nowa[t-1] i wcześniejsze oraz prognozowany wpływ z historii. Prędkość latentu nie jest pędem optymalizatora.

Raportować MAE/RMSE/redukcję MSE, znak wszystkich sond, większość z przeszłości, zbilansowaną trafność, kalibrację przedziałów oraz korelacje/autokorelacje i wariancję reszt. Dotychczasowe wyniki warunkowe z już znaną paczką kolejnego kroku w `maly_vit_horyzont_rownania_krok` i `maly_vit_prognoza_tozsamosci_z_historii` nie spełniają aktualnego kryterium prognozy. Kierunek LDA pozostaje stały; test s nie jest prognozą pełnego wektora768.

Osobny przyszły etap: ustalić, które wektory/składowe reprezentacji tożsamości będą się zmieniać, aby dopełniać reprezentację. Zdefiniować wielowymiarowy stan w jawnej podprzestrzeni tożsamości, jej wymiar, podstawę, współrzędne i cel dopełniania. Podprzestrzeń oraz próg zmiany wybierać tylko z przeszłości/treningu; przy każdym porównaniu podstawa zamrożona i wspólna. Przewidywać zmiany współrzędnych i ranking składowych do zmiany, osobno błąd wektorowy, znaki i wykrycie aktywnych składowych. Nie utożsamiać bazowych neuronów, modów PCA i semantycznych kierunków. Wybór podprzestrzeni wymaga kontroli swoistości względem innych narzędzi, tła, odbić, krawędzi i losowych podprzestrzeni o tym samym wymiarze; wskazane kontrolowane interwencje na narzędziu przy wspólnym tle oraz na tle przy wspólnym narzędziu. Trafna prognoza ruchu sama nie dowodzi tożsamości semantycznej.

Etap wielowymiarowy na dziś tylko zaplanowany: bez pobierania22GB, nowych treningów lub Edenu. Pilot s jest diagnostyką modelowania czasowego, nie odpowiedzią na pytanie o wszystkie wektory.

### Wynik krótkiego pilota, 2026-10-01 15:00:30 Europe/Warsaw

Na H1 (jedna aktualizacja, nie500) walidacja s47 wybrała lokalny lifting Koopmana + jądroGLE20 + historyczne wpływy paczki/AdamW, ridge0.001. Test eksploracyjny s48:80,98procent znaków na wszystkich sondach,80,98procent zbilansowanej trafności,64,96procent redukcjiMSE; większość znaku z przeszłości49,73procent. Dawny bezpośredni Langevin20 na tych samych oknach:79,86procent znaków i62,40procent redukcjiMSE. Sam Koopman z liftingiem stanu daje tylko0,42procent redukcjiMSE na H1; przewagi nie ma z definicji. Późny s47 dla wybranego modelu:79,24procent znaków i59,50procent redukcjiMSE. Cel80 osiągnięty wyłącznie w mierze znaku na jednym biegu eksploracyjnym, nie w redukcjiMSE ani jako potwierdzenie między ziarnami.

Dla kolejnego zapisu sond co10 aktualizacji: znak65,07procent, redukcjaMSE22,13procent. Wyniki H3/5/20/50 i wszystkie kontrole w osobnym `wyniki/krotki_horyzont/METRYKI.json`; opis `wyniki/krotki_horyzont/WYNIKI.md`. Rzeczywiste paczki powodujące cele nie są wejściami. Dotychczasowe wynikiH500 zachowano. Do potwierdzenia potrzeba świeżego odłożonego przebiegu po zamrożeniu modelu, konkretnego horyzontu zapisu i definicji celu80. Etap podprzestrzeni wielowymiarowej tożsamości pozostaje planem.

## Rozszerzenie 2026-10-01 15:04:08: Koopmanowski dryf, obrazowe sterowanie i obserwator

Priorytetem jest GLE z dryfem estymowanym w przestrzeni Koopmana, jądrem pamięci oraz osobnym sterowaniem. Porównać EDMD z ustalonym liftingiem i learned lifting/autoenkoder Koopmana. EDMD samo nie jest autoenkoderem. Uczenie podprzestrzeni, normalizacja i dopasowanie modelu tylko na przeszłości; nie zakładać dokładnej skończenie wymiarowej globalnej linearyzacji nieliniowego treningu.

Sterowanie ma jawnie opisywać cechy obrazów faktycznie wykonanych historycznych paczek. Opis/embedding paczki nie jest embeddingiem sondy, a jej wpływ na aktualizację nie jest samą treścią obrazu. Stan obejmuje historię reprezentacji, ewentualne uzasadnione przyrosty i pamięć optymalizatora. Zdefiniować obserwowany i ukryty stan oraz zgodność czasu. W plikach aktualizacja k prowadzi s[k]->s[k+1]; paczka k jest dostępna dla prognozy dopiero po jej wykonaniu. Osobno raportować identyfikację wejście-stan-wyjście z historycznych danych i marginalną prognozę bez rzeczywistych przyszłych wejść. Przyszłe wymuszenia tylko modelowane z historii i znanej polityki.

Zbadać identyfikację nieliniową, lokalne osadzanie Koopmana, obserwowalność konkretnego poszerzonego modelu, obserwatora stanu oraz reakcję na wejścia. Rangi i wzbudzenie sprawdzać dla modelu; nie ogłaszać tych własności dla pełnego ViT na podstawie samego embeddingu. Porównać filtr Kalmana w poszerzonym stanie z pamięcią/skorelowaną resztą. EKF/UKF jako opcjonalne rozszerzenie modelu nieliniowego, z jawną dynamiką i modelem pomiaru. Hipoteza: część reszty zawiera niewydzielony sygnał. Potwierdzi ją wyłącznie przyczynowa poprawa poza dopasowaniem; nie przyjmować możliwości odzyskania prawdziwego białego szumu. Gaussowskość, białość, autokorelacje, korelacje z wejściem i heteroskedastyczność oceniać oddzielnie.

Dalsza literatura automatyki/robotyki/Koopmana/Mori-Zwanziga: paper-search i źródła pierwotne. Undermind/deep research tylko przy rzeczywistej dostępności; brak funkcji raportować, nie zastępować go udawaną usługą. Niezależny przegląd metod przez gpt-6-astra xhigh, w osobnym pliku, bez kolizji i cofania zmian innych.

Raportować krzywe znak/balanced accuracy/MAE/RMSE/redukcjaMSE vs H i maksymalny zmierzony H utrzymujący jawnie zadany próg80. Nie gwarantować monotonicznej trafności. Kryterium80 znaku nie jest80 redukcjiMSE. s48 pozostaje eksploracyjne po obejrzeniu; świeży odłożony przebieg konieczny do potwierdzenia przewagi. Kolejny etap zapisuje odrębne artefakty, nie nadpisuje wcześniejszych pilotów.

Budżet: ograniczone lokalne obliczenia CPU na już dostępnych danych; bez nowych Eden jobs, pobierania22GB, dużego treningu, commitów i zmian dawnych eksperymentów. Przed ewentualnym lokalnym GPU wykonać audyt dostępności/konfliktów; mały autoenkoder można badać na CPU w ustalonym limicie epok. Dotychczasowego krótkiego pilota nie powtarzać.

## Obowiązująca konwencja cząsteczki i układów latentnych, 2026-10-01 15:11:41 Europe/Warsaw

To terminologia uzgodniona przez użytkownika, nie dowód semantycznej tożsamości ani zmiana istniejącej implementacji.

- Cząsteczka: reprezentacja jednego tokenu obrazu z_i(t) w wspólnej przestrzeni R^768.
- Globalny układ latentny: wspólna przestrzeń współrzędnych, nie uśredniony wektor obrazu lub zbioru tokenów.
- 768 makrowektorów: wspólne wektory bazowe e_j tej przestrzeni.
- Mikrowektory cząsteczki: jej składowe z_ij(t)e_j, których suma tworzy z_i(t).
- Mikro układ cząsteczki: opis składowych/współrzędnych jednej cząsteczki względem wspólnej bazy. Nie oznacza osobnej wyuczonej bazy ani niezależnego zbioru wektorów przechowywanego w modelu.
- 600 cząsteczek na obraz wynika z tokenizacji konkretnego ViT przy używanej rozdzielczości. 768 to wymiar reprezentacji w mierzonym miejscu sieci; liczby nie są uniwersalnymi parametrami każdej tokenizacji.
- Ruch: zmiana z_i pomiędzy stanami treningu przy tym samym obrazie i tokenie. Mikroruch: zmiana odpowiadających składowych z_ij(t)e_j. Zmiana obrazu/augmentacji lub indeksu tokenu wymaga osobnego dopasowania, nie jest automatycznie ruchem tej samej cząsteczki.

Pooling tokenów i zamrożony rzut LDA s są osobnymi agregatami/obserwablami. Nie zastępować trajektorii cząsteczek ich średnią bez jawnego oznaczenia agregacji. Dotychczasowe lokalne piloty s prognozują skalar z uśrednionego latentu narzędzia, nie600 pełnych trajektorii ani768 mikroruchów każdej cząsteczki. Etap tych trajektorii pozostaje osobnym planem na dostępnych, jawnie wskazanych danych.

### Wynik odrębnego pilota EDMD/learned lifting/KF, 2026-10-01 15:28:17 Europe/Warsaw

Zakończono lokalny CPU pilot H1/3/5/10/20/50 z prawdziwym dopasowaniem całego liftingu przez K_H, learned encoder/decoder18epok oraz osobną pamięcią, PCA8 cech rzeczywistych obrazów paczek i przyczynowym filtrem reszty. Wewnętrzny podział chronologiczny: dryf do1400, reszty do1800; po wyborze refit odpowiednio2000/2500. S48 nadal eksploracyjne. Wyniki walidacyjnie wybranych modeli nie przewyższają wcześniejszego direct modelu; żaden wybranyH nie osiągnął80 w znaku ani redukcjiMSE. H1: fixedEDMD+pamięć+obrazy+wpływy+KF daje−53,08procent redukcjiMSE i57,06procent znaku; H10 fixedEDMD+pamięć20,19procent i64,04procent; H20 learnedlifting24,70procent i67,06procent. Wszystkie negatywy i ramiona bezKF zachowano.

PCA obrazów wykazuje przesunięcie rozkładu: mediana normy w8 standaryzowanych współrzędnych2,73 dla treningu s47,7,40 dla s48. To diagnostyka, nie dowód jedynej przyczyny. Reprezentacje pochodzą z osobno uczonych sieci; zgodność semantyczna współrzędnych między biegami nie jest udowodniona. Następny ograniczony test powinien użyć wspólnego zamrożonego encodera obrazów albo dopasowania reprezentacji wyłącznie z przeszłości, z kontrolą prostszych deskryptorów.

KF filtruje jedną AR1-resztę dołączoną do predyktora opóźnionego; nie jest pełnym filtrem600 cząsteczek lub wszystkich wag. Używa tylko zakończonej etykiety z początku t−H. EKF/UKF i pełna rozszerzona realizacja nadal wymagają osobnego protokołu; nie twierdzić, że zostały przetestowane. Independent Astra review rozróżnia direct regresję odEDMD oraz mechanistyczną identyfikację od umownej kolejności projekcji.

Raport, operatory, cechy/PCA, małe wagiAE, odtworzone modele, diagnoza i krzywe: `experiments/maly_vit_identyfikacja_langevin_koopman/wyniki/koopman_obserwator/`. PrzeglądAstry `PRZEGLAD_ASTRA.md`; literatura `zrodla/LITERATURA_OBSERWATOR.md` i `SOURCES_OBSERWATOR.json` (7źródeł). Undermind użyto do rzeczywiście dostępnych lookup/metadanych, bez deepresearchjobs. Bez nowego treninguViT, GPU/Edenu, pobierania22GB i commitów. Świeży test konieczny przed potwierdzeniem.

## Zmiana priorytetu: wspólna identyfikacja controlled Koopman-GLE, 2026-10-01 16:21:29 Europe/Warsaw

Obowiązujące nowe polecenie: filtr Kalmana/obserwator KF zostaje odłożony i nie wchodzi do kolejnego etapu. Istniejące wyniki KF pozostają historycznymi negatywami. Najpierw identyfikujemy wspólnie macierze dryfu w wybranej przestrzeni obserwabli, jądra pamięci stanu i historycznych wejść oraz resztę. Nie dopasowujemy autonomicznego dryfu i nie doklejamy później członu B u. Liniowa ewolucja w liftingu jest aproksymacją konkretnego modelu, nie założeniem dokładnej liniowości latentu ani globalnej linearyzacji ViT.

Źródłowa paczka u_k wywołuje S[k] -> S[k+1]. W chwili prognozy S[t] dostępne są wykonane paczki do u_(t-1). Ich cechy obrazowe pozostają jawne i odrębne od sond oraz pomiarów wpływu/gradientu. Zidentyfikować osobno kontrolowany model na ukończonych parach historycznych i marginalną prognozę bez rzeczywistych przyszłych u. W marginalnym wielohoryzontowym modelu wspólne regresory obejmują lifting bieżącego stanu, historię stanu i cechy wykonanych paczek oraz ich pamięć; wszystkie bloki estymujemy w jednym regularizowanym problemie. To operator projekcji wielohoryzontowej, nie dowód reakcji na interwencję.

Nowy odrębny etap porówna joint controlled lifting z pamięcią i wejściami, joint model bez obrazów, kontrolę autonomiczną oraz zapisane dawny GLE/direct i EDMD bez KF. Protokół ustalić przed testami: ten sam cel i H1/3/5/10/20/50, chronologiczny train/validation, wybór na walidacji, s48 eksploracyjne. Sprawdzić rangę, uwarunkowanie, współliniowość i niejednoznaczność rozdziału dryfu/pamięci/wejść. Nie ogłaszać identyfikowalności mechanizmów fizycznych na podstawie ridge ani liczby próbek sond.

Astra xhigh ponownie dopracowuje równanie i indeksowanie; Lean/SymPy weryfikują właściwe tożsamości eliminacji stanu ukrytego i osobnych jąder stanu/wejść. Jawne założenia i zakres dowodu algebraicznego nie oznaczają gwarancji prognozy. Białość, gaussowskość, zależność reszt od historii i heteroskedastyczność diagnozujemy oddzielnie; bez narzucania równowagowego FDT dla AdamW. Metryki znaku, balanced accuracy i wielkości/błędu pozostają oddzielne, próg80 celem wymagającym doprecyzowania.

Uruchomić celowany Undermind Deep Search i lekturę właściwych pełnych źródeł, z koordynacją research wag, bez nieograniczonej serii. Lokalny pilot CPU na istniejących danych; bez ViT treningu, Edenu, commitów i prawdziwych przyszłych klatek. Osobne artefakty etapu joint_controlled.

### Wynik etapu wspólnej identyfikacji bez KF, 2026-10-01 16:35:25 Europe/Warsaw

Zakończono odrębny joint controlled Koopman-GLE pilot CPU25,54s: cały lifting(z,z²),20 opóźnień jego przyrostów, pamięć wykonanych obrazów w rozłącznych oknach1/4/15 i oddzielne wpływyN/P; wszystkie bloki dopasowane JOINT. Train s47<=1800, val1800..2000, refit<=2500; s48 nadal eksploracyjne. Wybrane modele H1/3/5/10/20/50: redukcjaMSE64,12/40,38/25,67/21,86/22,95/29,74procent; znak80,92/72,84/67,20/65,19/64,73/68,17procent. Znak i balanced>=80 tylkoH1, nie redukcjaMSE. Dawny directH1 pozostaje nieznacznie lepszy64,96/80,98; brak wykazanej przewagi. Samo obrazowe wejścieH1 daje62,43/79,77 wobec samej pamięci62,31/79,85. Do80,92 doprowadziło ramię także z historycznymi pomiarami wpływu, nie same obrazy.

Oddzielny joint controlled one-step identyfikowano na ukończonych historycznych parach; przy prognozie wszystkie przyszłe obrazy zastąpiono modelem średniej20 wykonanych paczek. WynikH1:62,50redukcjiMSE i79,93znaku. Nie korzysta z rzeczywistych przyszłych klatek.36 kontroli prefiksowych przechodzi. S48 i późny fragment s47 nie stanowią świeżego niezależnego potwierdzenia. Poziom80 jest nadal celem o oddzielnych miarach.

Lean7twierdzeń nadInt i SymPy eliminacji macierzowej2x2 oraz jointblockridge przechodzą. Założona realizacja daje osobne jądra BE^jC orazBE^jF; nieznanyD*u_t nie znika przez eliminację. Dowód algebraiczny nie dowodzi trafności ani dokładnego liftingu ViT. Astra xhigh: PRZEGLAD_ASTRA_STEROWANIE.md. Celowany Undermind DeepSearch zakończył119wyników z abstraktów; przeczytano4pełnePDF przez narzędzie (Lin/Lu, Lin/Tian/Perez/Livescu, Uy/Peherstorfer, Otto/Peitz/Rowley), sprawdzono daty publikacji i ograniczenia. Raport zrodla/LITERATURA_JOINT.md rozdziela zakres odczytu i zastosowanie doAdamW. Istotne przyszłe rozszerzenie to interakcja stanu i wejścia, ponieważ addytywneBu nie opisze każdej dynamiki sterowanej.

Wyniki i krzywe: experiments/maly_vit_identyfikacja_langevin_koopman/wyniki/joint_controlled/WYNIKI.md; modele, prognozy, metryki, diagnostyka reszt/wejść i próg80 zapisane osobno. KFpozostaje odłożony, starych wyników nie usunięto. Bez treninguViT/GPU/Eden/commitów.
