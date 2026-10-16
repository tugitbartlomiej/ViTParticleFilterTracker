# Czy przeciwne reflektory działają razem, w innych kontekstach, czy jeden po drugim?

Analiza 15 października 2026. Kontynuacja badania współdziałania neuronów ostatniego MLP. Bez zmian aplikacji.

**Wynik: występują oba pierwsze zjawiska — nierównomierne działanie partnerów w różnych tokenach oraz równoczesna kompensacja ich podpisanych wkładów. Nie znaleźliśmy silnego, powtarzalnego sygnału, że jeden z przeciwnych neuronów regularnie uruchamia później drugiego, aby poprawić jego poprzedni ruch. W samym badanym MLP taka bezpośrednia kolejność nie występuje: neurony ukryte są obliczane równolegle z tego samego wejścia.**

Najciekawsza obserwacja związana z etykietami: **na tokenach, których centra leżą w ramce narzędzia, ujemny bilans członów krzyżowych przeciwnych par jest większy niż w dalekim tle.** Efekt pozostaje również dla par wybranych wcześniej, przed ocenianym końcem treningu. Nie oznacza to, że dany neuron należy wyłącznie do narzędzia ani że zmierzono fizyczną energię światła.

## Dane i przypisanie cząsteczek do obrazu

Trzy zapisane obrazy, jeden trening, 647 istniejących checkpointów co 100 kroków od 1000 do 65600. Każdy obraz daje 600 tokenów. Mierzymy 768 neuronów ukrytych ostatniego MLP i ich wkłady do 192-wymiarowej reprezentacji rezydualnej. Nie mylimy tych dwóch wymiarów.

Przestrzenna analiza obejmuje wszystkie 600 tokenów. Nie są to niezależne powtórzenia treningu. Etykiety są prostokątnymi ramkami, a nie segmentacją narzędzia.

| Obraz | Centra patchy w bbox | Patche przecinające granicę bbox | Patche całkowicie wewnątrz bbox | Dalekie tło |
| --- | ---: | ---: | ---: | ---: |
| 1 | 18 | 18 | 0 | 526 |
| 2 | 20 | 20 | 0 | 523 |
| 3 | 72 | 24 | 60 | 444 |

Dalekie tło: centrum patcha oddalone od ramki o więcej niż 32 piksele. Kategorie z tabeli częściowo się nakładają: główne porównanie **bbox vs dalekie tło** jest rozłączne, ale kategoria granicy może zawierać centra wewnątrz bbox. Pierwsze dwie ramki są tak cienkie, że żaden patch 16×16 nie mieści się w nich całkowicie. Przypisanie „cząsteczka z obszaru narzędzia” jest więc uzasadnione; przypisanie „cząsteczka zawierająca wyłącznie narzędzie” nie jest.

Reflektory/neuronowe kierunki są wspólne dla wszystkich tokenów. To aktywacja i wkład danego reflektora zależą od cząsteczki. Dodatkowo token w ostatnim bloku może zawierać kontekst innych obszarów obrazu po wcześniejszym attention.

## Co dokładnie liczymy

Wkład neuronu c to `u_c = z_c v_c`, gdzie `v_c` jest kolumną W2, a `z_c` podpisaną aktywacją GELU. Znak ma znaczenie: dla przeciwnych `v_c` i `v_d`, ale przeciwnych znaków `z_c` i `z_d`, rzeczywiste wkłady mogą się wzmacniać.

Zbiory par:

- **844 pary końcowe:** cos(v_c,v_d)<−0,5 w kroku 65640. To wybór retrospektywny; 844 pary nie oznaczają 844 odrębnych neuronów. Wcześniejsze 53,125 procent to 408 z 768 neuronów posiadających przynajmniej jednego partnera.
- **751 par wcześniejszych:** ten sam warunek w kroku 33000. Ocena późniejsza ogranicza selekcję na końcu; nie daje jednak niezależnego treningu.
- **751 par kontrolnych:** zachowany pierwszy neuron, drugi wybrany spośród neuronów o zbliżonej normie W2, z pominięciem przeciwnych par. Mediana różnicy norm wynosi 0,62 procent, 95. percentyl 4,12 procent; 26 z 751 dopasowań różni się o więcej niż 5 procent, maksimum 25,14 procent. Jest to przybliżona kontrola norm, bez dopasowania aktywacji i semantyki.

Trzy różne miary:

1. **Współaktywacja:** `|z_c| > 0,1 × P95_token(|z_c|)` i analogicznie dla partnera. Próg ustalany osobno dla neuronu, obrazu i checkpointu. GELU nie ma naturalnego binarnego „odpalenia”. To jawna konwencja, a nie własność biologiczna.
2. **Skrócenie sumy:** `C = 1 − ||u_c+u_d||/(||u_c||+||u_d||)`. Agregujemy z wagą `||u_c||+||u_d||`, aby mikroskopijne wkłady nie dominowały. C mierzy stratę długości przy dodawaniu wektorów; również wektory prostopadłe mogą mieć C>0. Dlatego nie używamy samego C jako dowodu przeciwnego działania.
3. **Podpisany bilans krzyżowy:** `R = Σ 2<u_c,u_d> / Σ (||u_c||²+||u_d||²)`. R<0 oznacza rzeczywiste zmniejszenie kwadratu normy sum par względem ich składników diagonalnych. Sumy obejmują wybrany zbiór par, tokenów i checkpointów; neuron może pojawić się w kilku parach. **R nie jest procentem całego ruchu tokenu ani fizyczną energią w dżulach.**

Wyniki poniżej dotyczą końcowych 107 checkpointów, kroki 55000–65600, około epok 50,3–60,0.

## Czy działają jednocześnie?

Tak, ale nie jest to dominujący stan w obszarze ramki przy zadanym progu. Dla 844 końcowych par:

| Obraz | Oba aktywne w bbox | Dokładnie jeden aktywny w bbox | Oba nieaktywne w bbox | Oba aktywne w dalekim tle |
| --- | ---: | ---: | ---: | ---: |
| 1 | 27,65% | 49,56% | 22,79% | 47,41% |
| 2 | 28,56% | 48,98% | 22,46% | 53,14% |
| 3 | 33,13% | 48,60% | 18,26% | 33,82% |

Mianownik to wszystkie obserwacje token–para w danej grupie i okresie. Po ograniczeniu do przypadków, w których przynajmniej jeden partner przekroczył próg, oba przekraczają go w **35,81; 36,83; 40,54 procent** przypadków bbox. Odpowiednio w około 59–64 procent takich przypadków próg przekracza tylko jeden partner.

Nie wszystkie pary zachowują się tak samo. W bbox obrazu 1 około 25,47 procent par współaktywuje się w więcej niż połowie swoich przypadków „przynajmniej jeden aktywny”, a 22,16 procent w mniej niż jednej dziesiątej. Dla obrazów 2 i 3 te udziały wynoszą odpowiednio 27,96/21,33 oraz 33,53/6,40 procent. Rozkład par zawiera więc zarówno często wspólnie aktywnych partnerów, jak i partnerów niemal rozdzielonych kontekstowo.

Próg ma wpływ na liczby: kontrola na 11 punktach późnego treningu daje dla progu 0,05 udział „oba aktywne” w bbox 38,4–46,0 procent, a dla progu 0,2 — 18,3–20,9 procent. Dlatego najważniejszym sprawdzeniem pozostają ciągłe, podpisane wkłady, bez progowania.

![Współaktywacja w czasie](wspolaktywacja_w_czasie.png)

## Czy rzeczywiście się kompensują i czy dotyczy to narzędzia?

**Tak, bilans par jest ujemny, a w bbox wyraźniej ujemny niż w dalekim tle.** Poniżej sumowanie po wszystkich tokenach grupy i późnych checkpointach, z zachowaniem znaków GELU. Wszystkie liczby to procent sumy składników diagonalnych par.

| Zbiór par | Obraz | Bilans R w bbox | Bilans R w dalekim tle | R po przetasowaniu partnera wewnątrz bbox |
| --- | ---: | ---: | ---: | ---: |
| Końcowe 844 | 1 | −4,687% | −0,371% | −2,571% |
| Końcowe 844 | 2 | −3,275% | −0,620% | −2,336% |
| Końcowe 844 | 3 | −3,459% | −0,718% | −1,473% |
| Wcześniejsze 751 | 1 | −4,083% | −2,757% | −2,024% |
| Wcześniejsze 751 | 2 | −3,417% | −2,553% | −1,563% |
| Wcześniejsze 751 | 3 | −3,712% | −1,814% | −1,497% |
| Kontrolne 751 | 1 | −0,179% | +0,112% | +0,140% |
| Kontrolne 751 | 2 | −0,316% | +0,110% | −0,082% |
| Kontrolne 751 | 3 | +2,802% | +0,800% | +0,611% |

Przetasowanie zastępuje aktywację partnera aktywacją tego samego neuronu z innego tokenu **tej samej grupy przestrzennej i tego samego obrazu/checkpointu**. Rozkłady aktywacji pozostają te same. Większa ujemność rzeczywistego R pokazuje, że nie wystarcza sama geometria wag i marginalna aktywność: ma znaczenie, które wkłady spotykają się na tej samej cząsteczce. Jest to kontrola opisowa z jedną ustaloną permutacją na grupę, nie test istotności oparty na niezależnych obrazach.

Kompensacja i wzmacnianie współistnieją. Dla końcowych par w bbox obrazu 1 suma ujemnych członów wynosi **−6,871 procent**, dodatnich **+2,184 procent**, razem **−4,687 procent**. W obrazach 2 i 3 odpowiednio: −5,903/+2,628 oraz −6,089/+2,630 procent. Tak więc przeciwne wagi nie oznaczają wyłącznie znoszenia dla każdego tokenu.

Średnie ważone skrócenie C dla końcowych par w bbox to 10,43; 10,47; 13,05 procent, a równowaga długości `2 min(||u_c||,||u_d||)/(||u_c||+||u_d||)` tylko 20,51; 20,91; 25,56 procent. Ograniczona równowaga wyjaśnia, dlaczego przeciwne kierunki często nie dają dużego znoszenia: jeden rzeczywisty wkład jest wyraźnie silniejszy.

### Przykład na obrazie

Wybrano po wynikach parę 209/601 o największym iloczynie skumulowanej siły i skrócenia w bbox obrazu 1. Jest to ilustracja, nie niezależne potwierdzenie. W kroku 65600 cosinus kierunków wynosi −0,640, czyli kąt około 129,8 stopnia.

![Przykład przestrzenny](przyklad_przestrzenny.png)

Mapy dwóch wkładów przedstawiają rzuty na **tę samą oś** `(v_209−v_601)/||v_209−v_601||`, dlatego kolor przeciwny oznacza kierunek przeciwny na tej osi. Mapa C jest obliczona z pełnych 192 wymiarów. Ukryto w niej miejsca, gdzie suma długości obu wkładów jest mniejsza niż 5 procent jej 95. percentyla, aby duży względny stosunek przy niemal zerowych wkładach nie sugerował silnego zjawiska. Zielona ramka jest etykietą bbox; mapy nie są segmentacją ani dowodem wyłącznej specjalizacji neuronów w narzędziu.

## Czy jeden neuron działa później, aby poprawić poprzedni?

**Nie znaleziono silnego, stabilnego sygnału takiej relacji przy badanej rozdzielczości czasu.** W pojedynczym forward oba neurony są liczone z tego samego wejścia do ostatniego MLP. Nie ma bezpośredniego połączenia `c → d` między tymi dwoma neuronami ukrytymi. Następstwo w checkpointach oznacza zmiany wywołane uczeniem całej sieci, a nie kolejne odpalanie wewnątrz bloku.

Analiza czasu używa 751 par wybranych w kroku 33000, 323 późniejszych checkpointów 33400–65600 i 72 wcześniej ustalonych tokenów. W bbox pozostaje 16, 16 i 14 tokenów obrazów 1–3, w dalekim tle po 8. Aktywacje odtworzono w float32 z zapisanych stanów wejścia i właściwych wag, aby nie różnicować szumu zapisu float16.

Oba wkłady pary rzutujemy na stałą oś wyznaczoną w kroku 33000. Liczymy korelacje przyrostów oraz osobno poziomów po usunięciu gładkiego trendu. W tabeli średnia korelacja po tokenach i parach w bbox:

| Obraz | Przyrosty jednoczesne | Przyrosty: drugi po 100 krokach | Poziomy bez trendu: drugi po 100 krokach |
| --- | ---: | ---: | ---: |
| 1 | −0,0186 | +0,0084 | −0,0057 |
| 2 | −0,0151 | +0,0066 | −0,0045 |
| 3 | −0,0068 | −0,0010 | −0,0038 |

W całym badanym zakresie dodatnich opóźnień 100–500 kroków, we wszystkich trzech obrazach i obu głównych grupach, średnie korelacje przyrostów mieszczą się od −0,0024 do +0,0084. Dla poziomów bez trendu: od −0,0079 do +0,0018. Kontrolne pary również pokazują małe dodatnie korelacje przyrostów przy +100 krokach. Wśród pojedynczych par, po uśrednieniu po 72 tokenach, najsilniejsza ujemna korelacja przy +100 krokach wynosi −0,073; w kontroli można znaleźć −0,162. Nie ma tu szczególnie silnego efektu wyróżniającego przeciwnych partnerów.

Kontrola 32 cyklicznych przesunięć drugiej serii zachowuje jej autokorelację. Niektóre bardzo małe rzeczywiste średnie wychodzą poza zakres tych przesunięć; nie przedstawiamy tego jako braku jakiegokolwiek związku statystycznego. **Wielkość efektu jest mała, kierunek nie jest konsekwentnie korygujący, a obserwacja nie identyfikuje mechanizmu przyczynowego.** Nie wykonano testu niezależnych powtórzeń treningu ani wyboru najlepszego opóźnienia z późniejszym potwierdzeniem na nowym biegu.

![Opóźnienia](opoznienia.png)

### Kontrola pozornego „korygowania poprzedniego ruchu”

Sprawdzono także pełny 192-wymiarowy iloczyn skalarny pomiędzy zmianą wkładu neuronu a wcześniejszą zmianą całego tokenu po MLP. W bbox, dla neuronów należących do wcześniejszych przeciwnych par, średni ważony cosinus przy opóźnieniu 100 kroków wynosi około **−0,021; −0,024; −0,012**. Sama ujemność wyglądałaby jak słabe przeciwdziałanie wcześniejszemu ruchowi.

Ale po trzech losowych przestawieniach **całych stanów checkpointów przed różnicowaniem**, przy zachowaniu związku aktywacji, wag i tokenu w każdym stanie, wynik nadal jest ujemny: około **−0,024; −0,025; −0,016**. Oryginalna kolejność uczenia nie jest potrzebna do pojawienia się tego efektu. Dwa kolejne przyrosty mają wspólny środkowy stan z przeciwnym znakiem; różnicowanie może samo wprowadzać ujemną zależność. Nie traktujemy tego pomiaru jako dowodu opóźnionej korekty.

To jest diagnostyczna kontrola artefaktu, a nie estymacja rozkładu zerowego dla całego procesu uczenia. Nie wyklucza subtelnych relacji w niektórych parach ani zmian zachodzących wewnątrz pominiętych 100 kroków optymalizatora.

## Co można powiedzieć o mechanizmie

1. **Partnerzy są często nierównomiernie używani.** Etykiety pozwalają pokazać, że ten wzór zależy od tokenu i obszaru obrazu. To zgadza się z hipotezą obsługi odmiennych kontekstów przez przeciwne kierunki, ale nie definiuje jeszcze semantyki każdego neuronu.
2. **Część partnerów działa równocześnie i rzeczywiście zmniejsza normę wspólnego wkładu.** W obszarze bbox bilans wybranych przeciwnych par jest bardziej kompensujący niż w dalekim tle i niż po przetasowaniu partnerów między tokenami tej samej grupy.
3. **Nie ma dowodu schematu „pierwszy poruszył, drugi po chwili go poprawił”.** Jednoczesne składniki MLP i kolejne checkpointy trzeba rozdzielić. Małe efekty po różnicowaniu nie wystarczają, a wykonana kontrola potrafi je odtworzyć bez prawdziwej kolejności czasu.
4. **Kompensacja reprezentacji nie jest jeszcze korektą błędu ramki.** Ujemny iloczyn skalarny mówi o działaniu wektorów. To, czy poprawia IoU, wymaga zadaniowego odczytu lub interwencji. Wcześniejszy raport współdziałania zawiera takie ablacje; obecny eksperyment nie wykazał, że wszystkie opisane 844 pary podnoszą IoU. Również nie wykazał, że wzrost liczby przeciwnych kierunków został spowodowany wyłącznie nauką narzędzia.

## Ograniczenia i kontrola jakości

- Jeden trening i trzy różne obrazy; nie potwierdzono ich niezależności na poziomie filmu/pacjenta. Nie jest to wynik populacyjny ani próba generalizacji klinicznej.
- Bbox obejmuje tło, zwłaszcza dla cienkiego narzędzia; attention poszerza kontekst tokenu. Analiza nie lokalizuje neuronu jednoznacznie w obiekcie.
- 844 końcowe pary wybrano retrospektywnie. Wcześniejsze 751 par ograniczają ten problem, ale nadal pochodzą z tego samego treningu.
- „Aktywny” zależy od umownego progu. Główne pomiary podpisanych iloczynów skalarnych nie zależą od niego.
- Analiza opóźnień obejmuje 72 tokeny i stałą projekcję par; dodatkowy test całego ruchu używa 192 wymiarów. Nie przebadano każdej możliwej relacji w całej sieci.
- Geometryczne pary współdzielą neurony, tokeny są zależne, checkpointy są zależne. Nie wyznaczamy naiwnych p-value z milionów obserwacji token–para.
- Zapis wszystkich 600 aktywacji jest float16. Rekonstrukcja float32 w trzech checkpointach daje względny błąd RMS około 0,000227–0,000247. Dla całej późnej próbki czasowej błąd poziomów wynosi 0,000231, a błąd przyrostów 0,000869; właściwa analiza czasu używa odtworzonego float32.
- Tożsamość normy sumy dwóch wektorów sprawdzono bezpośrednio; maksymalny błąd bezwzględny próbek float32 2,38×10⁻⁷. Zgodność masek z `maska_narzedzia`, kompletność regularnych kroków i rozkład stanów aktywacji sprawdzone automatycznie. Szczegóły: `METADANE.json`, `CZAS_QC.json`, `WERYFIKACJA.json`.

## Pliki i odtworzenie

Kod i protokół: `experiments/wspoldzialanie_neuronow_20261015/przeciwne_aktywacje*`.

Z katalogu SmallViTAblation uruchomić kolejno:

```powershell
python experiments/wspoldzialanie_neuronow_20261015/przeciwne_aktywacje.py
python experiments/wspoldzialanie_neuronow_20261015/przeciwne_aktywacje_czas.py
python experiments/wspoldzialanie_neuronow_20261015/przeciwne_aktywacje_energia.py
python experiments/wspoldzialanie_neuronow_20261015/przeciwne_aktywacje_kontrola_czasu.py
python experiments/wspoldzialanie_neuronow_20261015/przeciwne_aktywacje_raport.py
```

Wymagane: numpy, pandas, torch, timm, scipy/sklearn, matplotlib, threadpoolctl; lokalne checkpointy i wcześniejszy `outputs/optyka_hipoteza_20261015/pomiar.npz`. Obliczenia wykonano w istniejącym środowisku Python z GPU. Kod zapisuje wyłącznie katalog niniejszego eksperymentu; kolejne uruchomienie zastąpi jego własne pliki wynikowe.

Najważniejsze dane: `historia.csv`, `podsumowanie_pozne.csv`, `pary_pozny_trening.csv`, `energia_podsumowanie.csv`, `opoznienia.csv`, `opoznienia_kontrola_przesuniec.csv`, `przeciwdzialanie_ruchowi.csv`, `kontrola_przemieszania_stanow.csv`, `selektywnosc_neuronow.csv`. `pary.npz` przechowuje dokładne identyfikatory par i osie źródłowe, a NPY zapewniają odtwarzalny wybór wag i aktywacji. Wykresy i CSV generowane automatycznie; niniejsza synteza interpretacyjna napisana na podstawie zweryfikowanych wyników.
