# Ruch cząsteczek a uczenie detekcji na stałych klatkach i sekwencjach

Data: 2026-10-01, 16:28:13 (Europe/Warsaw).
Status: plan rozszerzenia zaakceptowany przez użytkownika; wykonanie pomiarów nie zostało jeszcze rozpoczęte. Pobranie istniejących checkpointów Eden zlecono osobno.

## Cel i pytanie

Zmierzyć, jak uczenie zmienia pełne reprezentacje tokenów tych samych obrazów, oraz czy zmiany poprzedzają lub towarzyszą poprawie detekcji narzędzia. Rozdzielić efekt zmiany stanu sieci od efektu zmiany klatki filmu. Sam duży ruch nie dowodzi uczenia tożsamości ani jego przyczynowego wpływu na detektor.

## Podstawowa terminologia

Cząsteczka z_i(x,t) jest reprezentacją tokenu i obrazu x w stanie treningu t. W badanym miejscu SmallViT ma 768 współrzędnych. Globalny układ latentny to wspólna przestrzeń R^768 z makrowektorami bazowymi e_j. Mikrowektory cząsteczki to składowe z_ij e_j; jej mikro układ oznacza opis tych składowych we wspólnej bazie, bez dodatkowej niezależnej bazy. Obraz 480 × 320 daje 600 cząsteczek dzięki tokenizacji tej architektury. Są to parametry badanego modelu.

Ruch podczas uczenia: delta z_i = z_i(x,t+h) - z_i(x,t), przy tym samym obrazie i indeksie tokenu. W filmie ten sam indeks przestrzenny nie oznacza automatycznie tego samego fragmentu materialnego narzędzia.

## Dane i aparat

- Istniejący bieg Eden: maly_vit_ruch_czastek_gesta_historia. 101 checkpointów kroków 0, 10, ..., 1000. Nie wykonywać nowego treningu w tym etapie.
- Manifest pobrania: kompletność kroków, sumy SHA-256, architektura, statystyki BatchNorm. Wagi co 10 kroków umożliwiają pomiar co 10, nie odtworzenie rzeczywistych wag wszystkich pośrednich kroków.
- Zacząć od kilku stałych klatek z przypadkami poprawnych i błędnych detekcji; użyć obrazów z filmów treningowych oraz odłożonych. Regułę doboru określić przed analizą, nie wybierać tylko efektownych trajektorii.
- Wykorzystać istniejące 12 obrazów pomiarowych jako pierwszy punkt odniesienia; następnie krótkie sekwencje. Jawnie oznaczać role train/held-out i zakres wcześniejszego wykorzystania danych.
- Pomiar eval, stałe przetwarzanie, bez augmentacji, z zachowaniem BN checkpointu. Powtórzenie tego samego forwardu jako kontrola dokładności. Zapisać warstwę: aktywacje GELU ostatniego MLP, nie utożsamiać 768 z szerokością całego backbone.
- Zapisywać pełne tokeny, identyfikatory obrazów/tokenów, maski/ramki, kroki i wyniki detekcji. Pooling oraz rzut LDA stanowią osobne pochodne, nie zastępują pełnych cząsteczek.

## Dwa rozdzielone wymiary pomiaru

1. Stała klatka, zmienny checkpoint: efekt zmiany sieci podczas uczenia.
2. Zmienna klatka sekwencji, stały checkpoint: reakcja zamrożonej sieci na film.
3. Siatka klatka × checkpoint: zestawienie obu efektów i ich interakcji. Przy śledzeniu obiektu między klatkami uwzględnić zmianę maski/położenia; nie zakładać zgodności tokenów obiektu bez dopasowania.

Odtworzenie faktycznych paczek wymaga ich indeksów, augmentacji i stanu losowania. Nie twierdzić, że kolejność klatek filmu odpowiada kolejności treningu. Jeden batch32 daje jedną aktualizację; nie 32 kolejne ruchy jednej cząsteczki.

## Analizy

- Długość przemieszczenia, względny ruch, kierunek i podobieństwo kolejnych przyrostów, oscylacje/stabilizacja; jawna normalizacja i rzeczywisty odstęp kroków.
- Porównanie tokenów narzędzia, okolicy i odległego tła. Raportować próg maski, także kontrolę większego udziału narzędzia w tokenie.
- Przy każdym checkpointcie: strata na stałym obrazie, wynik dopasowania ramek/IoU, pewność i typ błędu detekcji. AP/mAP raportować dla odpowiednich zbiorów, nie jako trafność pojedynczego obrazu.
- Związek ruchu z późniejszą poprawą detekcji, także opóźnienia. Oddzielić wspólny trend czasu treningu od dodatkowej informacji niesionej przez ruch; korelacja nie dowodzi przyczynowości.
- Wspólne kierunki/podprzestrzenie oraz kierunki zmienne w fazach uczenia. Dopasowanie na danych uczących, ocena na odłożonych filmach/czasach; kontrola losowych kierunków tej samej liczby i samej normy ruchu.
- Osobno surowy ruch i ruch po korekcie wspólnego dryfu. Zmiana skali/obrotu reprezentacji i BN mogą zmieniać tokeny bez poprawy działania. Ewentualne wyrównanie przestrzeni estymować wyłącznie z dozwolonych danych i raportować obok surowego układu; nie usuwać automatycznie badanej dynamiki.
- Sekwencje: spójność reprezentacji narzędzia oraz jakość detekcji przy ruchu, zmianach oświetlenia i zasłonięciach. Spadek zmienności sam w sobie nie dowodzi poprawy i może oznaczać utratę informacji.

## Prognozowanie i testy

Pary stan teraz / stan później dla horyzontów 10, 20, 50 i 100 aktualizacji, zgodnych z checkpointami. Porównać brak zmiany, prosty dryf, Koopmana i GLE z pamięcią. Pełny wektor: błąd wielkości, znormalizowany błąd, zgodność kierunku przyrostu z progiem dla małych zmian. Rzut LDA: osobno znak, balanced accuracy i redukcja MSE. Nie przenosić wyników skalarnej prognozy na pełne tokeny. Cel 80 procent wymaga wskazania konkretnej metryki.

Brak rzeczywistych przyszłych paczek/gradientów w prognozie. Wspólne okna i sondy dla modeli, wybór na walidacji; film/obraz/przebieg jako jednostka oceny, nie każdy token jako niezależna próba. Obecny jeden bieg i 101 checkpointów ograniczają zakres wniosków.

## Wizualizacja

Stała projekcja 2D dla wszystkich checkpointów, dopasowana na wydzielonych danych; 600 punktów i strzałki przed/po, kolor narzędzie/tło. Trajektorie wybranych tokenów i mapy ruchu na obrazie. Obok krzywe jakości detekcji i normy ruchu w pełnych 768 wymiarach. Projekcja może ukrywać ruch, więc nie jest samodzielnym dowodem.

## Kolejność i artefakty

1. Zweryfikować pobrane checkpointy i kilka powtórzonych forwardów.
2. Mały pilot stałych obrazów we wszystkich checkpointach, z kontrolą pamięci/dysku przed rozszerzeniem.
3. Połączyć ruch i detekcje; dopiero potem rozszerzać na sekwencje i prognozy.
4. Osobny katalog experiments dla realizacji, konfiguracja, manifest, surowe tokeny, metryki i raport ograniczeń. Nie pobierać automatycznie istniejących 22 GB tokenów i nie tworzyć całej dużej siatki bez oszacowania kosztów.

Źródła lokalne: experiments/maly_vit_ruch_czastek_gesta_historia/README.md i WYNIKI.md; experiments/maly_vit_ruch_czastek_maska_1px/WYNIKI.md; dotychczasowy plan poprawy równań Langevina i Koopmana. Ten plik jest planem, nie wynikiem potwierdzającym mechanizm tożsamości.
