# Co oznacza wzrost „przeciwnych reflektorów” do 53 procent?

Analiza 15 października 2026. Kontynuacja pomiarów współdziałania neuronów. Nie zmieniono aplikacji.

**Na tych danych jest to przede wszystkim wskaźnik przebudowy geometrii kierunków wyjściowych MLP. Nie wykazał silnego powiązania z bieżącym IoU ani pewnością po uwzględnieniu wspólnego czasu uczenia. Wzrost zaczyna się około epoki 3, jest szybki między epokami 5 i 30, a potem przechodzi w plateau. Epoka 60 jest końcem obserwacji, nie początkiem zjawiska.**

## Odnaleziony wykres i dokładna definicja

Wykres 9 zakładki „Lasery i koloidy”: `app/zakladka_laserow.py`, `_wykresy()`, rysuje `h.pary`. W `app/lasery_dane.py`, funkcja `policz()`, miara powstaje jako:

`P = mean_c [ min_(d != c) cos(v_c, v_d) < -0.5 ]`,

gdzie `v_c = W2[:,c]` jest 192-wymiarową kolumną fc2 ostatniego MLP, a c przebiega 768 neuronów. Kod normalizuje kolumny przed obliczeniem cosinusów. Warunek oznacza kąt większy niż 120 stopni. Nie jest to wymaganie dokładnego przeciwieństwa 180 stopni ani kąt mierzony w projekcji 3D.

Na końcu: **408/768 = 53,125 procent neuronów ma co najmniej jednego takiego partnera**. To nie znaczy, że 53 procent wszystkich par jest przeciwnych. Rzeczywisty udział takich nieuporządkowanych par wynosi tylko **844/294528 = 0,28656 procent**. Jeden neuron może mieć wielu partnerów. Nie wynika stąd również, że 53 procent ruchu albo aktywności się znosi.

Miara jest niezależna od obrazu. Sprawdziłem zgodność wszystkich trzech zapisanych historii oraz ponownie obliczyłem P z wag **we wszystkich 806 checkpointach**: wyniki zgodne w granicach zapisu float32. Dane dla trzech obrazów nie są trzema niezależnymi powtórzeniami tego wskaźnika.

### Uwaga o osi aplikacji

Aplikacja rysuje x jako **indeks checkpointu**, a następnie umieszcza etykiety kroków i epok. Na początku zapis jest co 10 kroków, później głównie co 100, z dodatkowymi końcami epok. Odległości poziome nie są więc proporcjonalne do czasu treningu. Poniższe wykresy i tempa korzystają z rzeczywistych kroków. W tym biegu 1 epoka = 1094 kroki, a 60 epok = 65640 kroków.

## Kiedy zaczyna rosnąć i kiedy zwalnia

Pierwsza dodatnia wartość to 3/768, czyli 0,391 procent, w kroku 3200 — epoce 2,925. W tabeli są pierwsze obserwowane przekroczenia kolejnych progów; nie oznaczają trwałego pozostania powyżej progu.

| Krok pierwszego przekroczenia | Epoka | Udział procent |
| --- | --- | --- |
| 3200 | 2.925 | 0.391 |
| 5300 | 4.845 | 1.042 |
| 9300 | 8.501 | 10.677 |
| 15500 | 14.168 | 25.391 |
| 31100 | 28.428 | 50.391 |
| 34300 | 31.353 | 53.906 |

Tempo z regresji liniowej w ustalonych przed analizą przedziałach:

| Od epoki | Do epoki | Wzrost w p.p. na epokę |
| --- | --- | --- |
| 0 | 5 | 0.212 |
| 5 | 15 | 2.400 |
| 15 | 30 | 1.521 |
| 30 | 45 | 0.059 |
| 45 | 60 | 0.035 |

Najbardziej intensywny wzrost przypada na **epoki 5–15: około 2,40 punktu procentowego na epokę**. W epokach 15–30 spada do 1,52 p.p./epokę, a po epoce 30 wynosi tylko około 0,035–0,059 p.p./epokę, z wahaniami.

Niezależny opis całej krzywej ciągłą łamaną o trzech odcinkach lokuje załamania przy kroku 4000 (epoka 3,66) i 30500 (epoka 27,88). Zmiana siatki dopasowania z 500 na 1000 kroków daje 4000 i 30000. Są to punkty uproszczonego opisu krzywej, **nie wykryte fizyczne przemiany fazowe** ani dokładne daty biologicznej reorganizacji.

Maksimum wynosi **56,771 procent** przy kroku 60100, epoce 54,936. Koniec 60. epoki jest nieco niżej, przy 53,125 procent. Zatem „53 procent przy 60. epoce” jest poprawnym odczytem końcowej wartości, ale pomija wcześniejszy wzrost i późniejsze oscylacje.

Metadane i kod treningu wskazują stały learning rate po rozgrzewce 300 kroków. Zapisane wznowienie nastąpiło po epoce 21 (22974 kroki). Nie ma w sprawdzonym przepisie zmiany harmonogramu LR przy epoce 28–30, która sama tłumaczyłaby plateau. To nie wyklucza innych przyczyn zmiany dynamiki optymalizacji.

![Przebiegi](przebieg_w_czasie.png)

## IoU czy pewność?

Główne porównanie używa **647 dokładnie istniejących zapisów co 100 kroków**, od 1000 do 65600. W ten sposób gęste początkowe checkpointy nie dostają nadmiernej wagi. IoU i pewność dotyczą ramki o najwyższym wyniku w każdym obrazie i checkpoincie; mogą więc pochodzić od różnych tokenów. Pewność jest wyjściem sigmoid głowy, nie skalibrowanym prawdopodobieństwem trafnej detekcji.

Poniżej korelacje Pearsona. „Po usunięciu trendu” to korelacja reszt po osobnym odjęciu gładkiej funkcji czasu od obu szeregów. „Przyrosty” porównują zmiany między kolejnymi punktami rozstawionymi o wskazaną liczbę kroków.

| Miara | Poziomy | Po usunięciu trendu | Przyrosty 100 | Przyrosty 500 | Przyrosty 1000 |
| --- | --- | --- | --- | --- | --- |
| IoU obraz 1 | 0.445 | -0.023 | 0.024 | 0.026 | 0.068 |
| IoU obraz 2 | 0.523 | -0.007 | -0.074 | 0.050 | -0.065 |
| IoU obraz 3 | 0.276 | 0.016 | -0.008 | -0.138 | -0.203 |
| Pewność obraz 1 | 0.776 | 0.052 | -0.028 | 0.144 | 0.073 |
| Pewność obraz 2 | 0.747 | 0.033 | 0.045 | 0.085 | 0.180 |
| Pewność obraz 3 | 0.067 | -0.026 | 0.022 | 0.064 | -0.004 |

**Przed korektą silniejszy związek widać z pewnością na obrazach 1 i 2**: r około 0,78 i 0,75, wobec 0,44 i 0,52 dla IoU. Nie powtarza się to na obrazie 3: jego korelacja z pewnością wynosi tylko 0,067. Samo P pozostaje przecież identyczne dla wszystkich trzech obrazów.

**Po usunięciu trendu czasu wszystkie sześć korelacji mieści się między −0,027 a 0,052.** Zmiana elastyczności trendu (4, 6 i 10 węzłów spline) pozostawia wszystkie te korelacje w przedziale około −0,041 do 0,064. To nie jest wynik jednego szczególnie dobranego wygładzenia.

Przyrosty co 100 kroków również dają małe wartości: od −0,074 do 0,045. Przy 1000 krokach pojawiają się większe pojedyncze wartości, np. −0,203 dla IoU obrazu 3 i +0,180 dla pewności obrazu 2, ale nie mają wspólnego kierunku na wszystkich obrazach. Nie widać stabilnej zasady „kiedy rośnie P, poprawia się ramka/pewność”.

Porównania fazowe są zgodne z tym obrazem. W epokach 15–30 korelacja P z pewnością wynosi +0,259, +0,507 i **−0,335** dla kolejnych obrazów. W epokach 30–60, gdy P jest blisko plateau, korelacje z pewnością pozostają małe. Pełne wyniki, także Spearmana, znajdują się w `korelacje.csv`.

Usunięcie trendu nie jest dowodem braku jakiegokolwiek związku przyczynowego: może usunąć także rzeczywisty, powolny efekt uczenia. Pokazuje, że ten jeden przebieg nie pozwala oddzielić takiego efektu od czasu treningu. Nie badano tu pełnego zakresu opóźnień ani nieliniowych modeli przewidywania.

![Korelacje po kontroli czasu](korelacje_po_trendzie.png)

## Inne wielkości: geometria i miary zadania

Najsilniejsza surowa współzmienność dotyczy geometrii: liczby przeciwnych par, minimalnych cosinusów, norm wag, kąta od inicjalizacji i spadku tempa obrotów. Część tych związków również zanika po usunięciu trendu:

| Miara geometrii | Poziomy | Po usunięciu trendu | Przyrosty 100 |
| --- | --- | --- | --- |
| angular_drift | -0.977 | 0.037 | 0.032 |
| edge_fraction | 0.990 | 0.591 | 0.451 |
| effective_rank | -0.797 | -0.166 | -0.042 |
| mean_min_cos | -0.971 | -0.589 | -0.297 |
| mean_weight_norm | 0.989 | -0.002 | 0.130 |
| rotation_from_start | 0.954 | 0.025 | 0.013 |

Nazwy: `mean_min_cos` — średni cosinus najbardziej przeciwnego partnera; `edge_fraction` — udział wszystkich przeciwnych par; `effective_rank` — rząd efektywny macierzy kierunków; `mean_weight_norm` — średnia długość kolumn W2; `rotation_from_start` — zapisany kąt obrotu od startu; `angular_drift` — mediana zapisanego dryfu kątowego.

Ważne rozróżnienie: **norma wag nie może sama zmienić P**, ponieważ cosinus jest liczony po normalizacji. Mimo surowej korelacji r=0,989 z normą, po kontroli czasu zostaje r≈−0,002. Także wspólny sztywny obrót wszystkich kierunków zachowałby P. Liczą się zmiany wzajemnych kątów, nie sama długość wektorów lub obrót całego układu.

Związek z liczbą przeciwnych par i minimalnym cosinusem pozostaje wyraźniejszy również w przyrostach. To jednak miary z tej samej macierzy kątów: ich współzmienność jest częściowo konsekwencją definicji, a nie niezależnym odkryciem mechanizmu treningu.

Rząd efektywny **niecentrowanych, jednostkowych kolumn W2**, liczony jako `tr(G)^2/tr(G²)` dla `G=Vn.T@Vn`, zmienia się z **153,80 do 30,10**. Oznacza to koncentrację kierunków w mniejszej liczbie dominujących osi w tym konkretnym pomiarze; nie zmianę architektonicznego wymiaru 192 ani matematycznego rzędu macierzy. Wzrost P współwystępuje z taką przebudową geometrii, ale krótkookresowe przyrosty obu miar mają małą korelację, około −0,042.

Próg jest umowny: na końcu dla −0,4 dostajemy **87,37 procent**, dla −0,5 **53,125 procent**, a dla −0,6 **17,188 procent**. Nie ma szczególnego fizycznego znaczenia liczby 53 procent. Średni minimalny cosinus wynosi wtedy około −0,504, blisko użytego progu, więc wiele kanałów może zmieniać kwalifikację przy niewielkim ruchu kierunków.

Sprawdziłem także zapisane wyniki zadania. Strata treningowa ramki ma surowo r=−0,824 z P, po kontroli czasu +0,026, a dla przyrostów −0,005. Ocena **mAP@0,5**, według metadanych na 3000 klatek walidacyjnych, ma na 60 końcach epok r=+0,590; po trendzie −0,100, dla zmian epoka-do-epoki −0,062. To użycie istniejącego logu, nie nowa ewaluacja ani nowy audyt rozdzielenia filmów.

Największa zapisana ocena mAP@0,5 przypada na krok 26000 (około epoki 23,77): 0,6593. Na końcu wynosi 0,6084. Wysokie P nie wyznacza więc najlepszego checkpointu detektora. Nie przypisujemy spadku mAP wzrostowi P — pojedynczy szereg tego nie rozstrzyga.

## Dlaczego przeciwne reflektory nie są tym samym co kompensacja cząsteczki

Dla konkretnego tokenu aktywny wkład neuronu to `u_c=z_c v_c`. Dwa przeciwne `v_c` mogą być słabo aktywne, działać z bardzo nierównymi amplitudami albo mieć różne znaki współczynników GELU. Dla ramki znaczenie ma dodatkowo `J_b(x) u_c`, czyli przekształcenie przez bieżącą głowę. P nie zawiera z ani Jacobianu.

W poprzednim eksperymencie zmierzono `η=||sum u_c||²/sum||u_c||²`, bez biasu, dla ośmiu tokenów środkowych na każdy obraz. W epoce 1, gdy P=0, mediany η wynoszą **4,69; 5,15; 3,27**. Na końcu przy P=53,125 procent wynoszą **2,77; 3,23; 1,61**. W tej konwencji η>1 oznacza dodatni łączny wkład wyrazów krzyżowych: nadal przeważa wzmacnianie względem sumy kwadratów osobnych norm, mimo obecności wielu przeciwnych kierunków wag.

W całym zakresie od 1000 kroku η i P mają ujemne korelacje od około −0,71 do −0,84, ale po odjęciu trendu pozostaje od −0,008 do +0,068. Długookresowe zmniejszenie η może współwystępować z powstawaniem przeciwnych kierunków, lecz P nie przewiduje tutaj stabilnie bieżących zmian aktywnej kompensacji.

Nie ma sprzeczności z wcześniejszą kontrolą, w której funkcjonalnie kompensujące pary znaleziono już przed treningiem. Tam szukaliśmy przeciwnych **wpływów na cztery współrzędne ramki**, a tu liczymy próg kąta **samych wag w 192D**. Funkcjonalna kompensacja jest możliwa nawet przy P=0.

W mikroskopie warto te pomiary zestawiać osobno: (1) geometria wzajemnych kierunków W2, (2) aktywne sumowanie wkładów wybranego tokenu, (3) wpływ na krawędzie i logit oraz konkurencja tokenów wybierających ramkę. Dotychczasowe ablacje potwierdziły, że trzeci poziom może obejmować przełączenie zwycięskiej detekcji. Obecna analiza nie wykazuje, żeby sam globalny udział przeciwnych reflektorów wyjaśniał takie zdarzenia.

## Wybrane końce epok

| Epoka | P_procent | IoU1 | IoU2 | IoU3 | Pewnosc1 | Pewnosc2 | Pewnosc3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.014 | 0.014 | 0.014 |
| 1 | 0.000 | 0.588 | 0.721 | 0.264 | 0.520 | 0.569 | 0.661 |
| 5 | 1.042 | 0.814 | 0.890 | 0.655 | 0.762 | 0.830 | 0.433 |
| 10 | 13.802 | 0.812 | 0.885 | 0.337 | 0.902 | 0.905 | 0.586 |
| 15 | 26.302 | 0.866 | 0.867 | 0.394 | 0.917 | 0.870 | 0.661 |
| 20 | 36.458 | 0.805 | 0.845 | 0.384 | 0.898 | 0.833 | 0.652 |
| 30 | 50.260 | 0.896 | 0.904 | 0.565 | 0.943 | 0.949 | 0.862 |
| 45 | 51.172 | 0.954 | 0.946 | 0.394 | 0.943 | 0.941 | 0.750 |
| 60 | 53.125 | 0.886 | 0.933 | 0.589 | 0.949 | 0.914 | 0.437 |

## Odtwarzanie i ograniczenia

Skrypty: `przeciwne_reflektory.py` (806 odczytów wag, analiza i wykresy), `raport_przeciwne.py` (raport i kontrole zapisanych wyników); protokół: `PROTOKOL_PRZECIWNE.md`, wszystkie w `experiments/wspoldzialanie_neuronow_20261015/`. Wyniki w podkatalogu `outputs/wspoldzialanie_neuronow_20261015/przeciwne_reflektory/`. Powiązany wcześniejszy raport: `../RAPORT.md`.

Jedna inicjalizacja i trzy obrazy ograniczają uogólnianie. Checkpointy są zależne w czasie; nie raportujemy p-wartości liczących je jako niezależne próby. Dopasowanie trendu i punktów załamania jest opisowe, bez przedziałów ufności. Parametry i pełne korelacje są zapisane, a wyniki niespójne pomiędzy obrazami nie zostały usunięte. Nie przeprowadzono interwencji zmieniającej sam P przy zachowaniu pozostałych własności modelu, dlatego nie ma podstaw do wniosku przyczynowego.
