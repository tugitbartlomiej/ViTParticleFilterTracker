# Co zawodzi i co udało się przewidzieć dla jednego wektora

Data: 2026-09-25. Wynik diagnostyczny, bez PCA, nowego treningu detektora ani generatora.

## Odpowiedź

Pełny latent 576×384 był już analizowany w poprzednim eksperymencie. Nie brakowało w nim pozostałych liczb. Zawodziło uogólnianie elastycznego prawa przejścia: bardzo niski błąd na dopasowanych parach nie przekładał się na trafną prognozę innych obrazów lub późniejszego uczenia.

W nowym teście wybrano jeden pełny wektor tokenu 299: wszystkie 384 współrzędne, pozycja w siatce 24×24 to wiersz 12, kolumna 11 przy indeksowaniu od zera. Wybrano go spośród ośmiu miejsc z największym pokryciem maską na obrazach fit. Końcowe błędy nie decydowały o wyborze. Maska lokalizuje fragment, ale nie dowodzi wyłącznej tożsamości narzędzia.

## Rzeczywisty operator

Najlepszy wariant na wcześniejszej walidacji miał jedną uczoną liczbę alpha dla całego wektora. Stan zachowuje obecne 384 liczby oraz 384 liczby poprzedniego przyrostu:

```text
v_t = z_t - z_(t-1)
v_hat_(t+1) = alpha * v_t
z_hat_(t+1) = z_t + alpha * v_t

alpha = sum(v_t * v_(t+1)) / sum(v_t * v_t)
```

Sumowanie obejmuje wszystkie współrzędne i poprawnie sparowane przykłady fit. Nie uśrednia wektora do jednej liczby: pojedynczy parametr określa wspólne tempo zmiany wszystkich współrzędnych. To silne założenie strukturalne modelu, nie PCA ani redukcja przechowywanego stanu.

Dla wektorów kolumnowych macierz 768×768 wynosi:

```text
K = [[I_384, alpha * I_384],
     [0,     alpha * I_384]]
```

Zapisane MATRIX_*.npz używają konwencji wierszowej, więc zawierają transpozycję powyższej macierzy: s_next=s*A+b. Wyraz b w tym wariancie jest zerowy. Nie znaleźliśmy dowolnej macierzy 768×768; dopasowaliśmy jej jawnie ograniczoną rodzinę. To lokalne liniowe przybliżenie Koopmana na stanie z pamięcią, bez dowodu dokładnego domknięcia rzeczywistego procesu.

Pierwsza macierz, dopasowana do historii kończącej się na kroku 210, ma alpha=0,9506574551. W 31 kolejnych lokalnych dopasowaniach alpha wynosi od 0,8312 do 0,9829, z medianą 0,9091. Współczynniki nie były przycinane do tego przedziału. Model przewiduje stopniowe spowalnianie zmiany. To tłumienie przewidywanego przyrostu, a nie zerowanie niewidzianych współrzędnych samego latentu.

## Wynik na późniejszych checkpointach i innych grupach

Token i rodzinę modelu wybrano na trzech wcześniejszych momentach, z ostatnim celem w kroku 190. Dopasowanie końcowe ma 48 obrazów z ośmiu grup i osiem znanych odcinków: 384 pary. Macierz lokalna jest aktualizowana tylko na danych fit dostępnych do początku każdej prognozy. Końcowa ocena obejmuje 24 obrazy z czterech innych grup i 31 początków prognozy od 210 do 360, czyli 744 pary.

| Horyzont | Model alpha | Ostatni przyrost | Brak zmian |
|---|---:|---:|---:|
| Jeden checkpoint, 5 aktualizacji | 0,0052561 | 0,0054058 | 0,0166273 |
| 20 checkpointów, 100 aktualizacji | 0,1324338 | 0,2192246 | 0,3074004 |

W tabeli podano RMSE całych wektorów. Metody w każdym wierszu mają te same obrazy i początki; długi horyzont obejmuje tylko trzy początki (210,230,250). W długim rollout macierz pozostaje zamrożona i nie dostaje prawdziwego przyszłego latentu.

- Jeden krok: błąd mniejszy o 68,39 procent od braku zmian oraz o 2,77 procent od powtórzenia ostatniego przyrostu. Przewaga nad ostatnim przyrostem występuje w każdej z czterech grup.
- Średni cosinus prognozowanej i rzeczywistej zmiany: 0,8849. Jest identyczny jak dla ostatniego przyrostu, bo dodatnie alpha zmienia długość, nie kierunek. Poprawa dotyczy kalibracji tempa; nie nauczyliśmy nowego kierunku.
- Błąd względem RMS rzeczywistej zmiany wynosi 0,3161, więc prognoza nie jest dokładna. Nie oznacza to procentu błędnych współrzędnych.
- Na przypadkach z pokryciem tokenu maską >=0,5: 155 par, RMSE 0,0047521 wobec 0,0049271 ostatniego przyrostu i 0,0092235 braku zmian.
- Długi horyzont: RMSE mniejsze o 39,59 procent niż ostatni przyrost; to dodatkowy wynik na tylko trzech początkach, bez niezależnego potwierdzenia.

**Ustalona bramka sukcesu pozostaje niezaliczona.** Wymagała co najmniej pięcioprocentowej przewagi w RMSE nad ostatnim przyrostem w głównym teście jednego kroku. Uzyskaliśmy 2,77 procent. Pozostałe trzy warunki przeszły. Nie zmieniamy progu po wyniku ani nie zastępujemy głównego kryterium korzystniejszym długim horyzontem.

## Jeden konkretny przykład

Po zakończeniu wybrano ilustrację według reguły niezależnej od błędu: pierwszy obraz holdout z pokryciem maską >=0,5 oraz pierwszy początek prognozy 210. To oryginalny obraz numer 66, grupa TR_0043_S1_P03, token 299, pokrycie 0,5124.

W przejściu 210→215 RMSE wynosi 0,0054224, wobec 0,0057316 ostatniego przyrostu i 0,0101066 braku zmian; cosinus 0,8819. To konkretne przybliżenie przyszłego wektora, nie idealne odtworzenie. Wszystkie jego liczby są w wyniki/WEKTOR_PRZYKLAD.npz, a przyrosty na [wykresie](wyniki/WEKTOR_PRZYKLAD.png). Ta ilustracja nie jest nowym testem potwierdzającym i nie sterowała modelem.

## Diagnoza oparta na porównaniu

Na wcześniejszej walidacji nawet najlepsze konfiguracje dużych macierzy miały MSE względem ostatniego przyrostu: operator stanu 14,17 razy większe, pełna pamięć 2,20 razy większe, operator przyrostów 2,40 razy większe. Najlepszy prosty model scalar_velocity miał 1,046 razy większe. Dopiero na późniejszym odcinku uzyskał niewielką przewagę. To sygnał zależności wyniku od fazy uczenia, a nie stabilnej przewagi wszędzie.

Zaobserwowane wyniki wspierają następującą interpretację: przy tych danych zbyt elastyczna macierz uczy zależności, które słabo przenoszą się na nowe przejścia. Krótkoterminowy kierunek jest już dobrze opisany przez ostatnią zmianę, a modelowi pomaga skorygowanie jej tempa. To wyjaśnienie empiryczne, nie dowód wyłącznej przyczyny.

Pozostaje dodatkowa hipoteza: latent obrazu nie zawiera pełnego stanu procesu uczenia, w szczególności wag, momentum optymalizatora i następnych danych. Samo powiększanie latentu nie zapewnia autonomicznej liniowej dynamiki. Literatura rozróżnia dopasowanie skończonego operatora od gwarancji prognozy: [Tu i in., 2014](https://doi.org/10.3934/jcd.2014.1.391). Szczegóły sześciu prac są w [poprzednim audycie](../koopman_gesta_historia/AUDYT_LITERATURY_6_KOOPMAN.md).

## Weryfikacja i ograniczenia

Cztery testy przeszły: macierz jawna względem formuł modeli, znane liniowe prawo 384 współrzędnych, poprawne parowanie i brak przyszłości w oknie, kontrola odstępów checkpointów. Dla celowo trudnych losowych danych różnica formy dualnej i jawnej dochodziła do 5,97e-9 względem normy wektora; kontrola normowa ma tolerancję 1e-8. Dla wszystkich 31 zapisanych macierzy wybranego modelu odtworzono prognozy dokładnie (maksymalna różnica 0).

Niezależny przegląd kodu nie wskazał błędu algebry ani przecieku przyszłych celów lub grup outer. Wskazał ograniczenia, które pozostają jawne:

1. Bramka globalnego przewidywania nie wymaga wyniku na samym narzędziu. Metryki z maską raportujemy osobno. Niektóre nieprzyjęte pozycje miałyby zerowe wsparcie foreground na holdout.
2. Wynik główny jest agregatem dla wybranej pozycji na wielu obrazach i odcinkach, nie gwarancją tolerancji każdej liczby każdego wektora. Osobno pokazujemy deterministyczny przykład.
3. Wybór pozycji korzystał z masek wszystkich grup fit, również wewnętrznej walidacji. Jest to zapisane w protokole; niezależny pozostał podział outer. Nie nazywamy walidacji wewnętrznej całkowicie odseparowaną od wyboru pozycji.
4. Ta pula była już oglądana w poprzednich analizach. Wnioski są diagnostyczne; cztery grupy to mało, a przekształcenia jednego obrazu są zależne.

Na początku Windows przerwał zapis STATUS.json podczas odczytu. Naprawiono jedynie ponawianie zapisu, przed kalibracją. Poprzedni protokół i opis wznowienia są zachowane. Nie zmieniono kryteriów statystycznych po wyniku.
