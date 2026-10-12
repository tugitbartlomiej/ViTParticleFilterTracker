# Wspólny ruch cząsteczek: co dało nowe równanie

Data: 2026-10-02. Obliczono 450 konfiguracji na walidacji. Wybrane wcześniej modele oceniono następnie na późniejszych aktualizacjach i na drugim obrazie bez ponownego dopasowania. **Uzyskano poprawę niektórych dłuższych prognoz, ale nie uzyskano ogólnie lepszego modelu ani skutecznego transferu długiej prognozy między obrazami.**

## Jak działa model

Każda cząsteczka narzędzia zachowuje własny stan. Osiem jego współrzędnych opisuje zamrożoną podprzestrzeń tożsamości U500. Dodajemy cztery lub osiem kierunków wyznaczonych z pozostałej części pełnego latentu. Są to kombinacje makrowektorów, a nie nowe osie dopisane do sieci. Wariant PCA szuka dominujących zmian, a wariant PLS kierunków skorelowanych na treningu z następną zmianą współrzędnych tożsamości.

Wariant sprzężony dostaje też średni stan cząsteczek narzędzia i średnią reprezentację obszaru poza jego maską. Model przewiduje wszystkie te stany samodzielnie. Po każdym kroku przelicza średnią narzędzia z własnych prognoz. Nie dostaje prawdziwego przyszłego ruchu kontekstu, tokenów ani przyszłych obrazów.

Dryf jest afiniczną aproksymacją dynamiki w rozszerzonym stanie, w duchu skończenie wymiarowego modelu Koopmana. Macierze pamięci opisują wpływ wcześniejszych przyrostów. Dopasowano je wspólnie z dryfem przez regularyzowaną regresję. To empiryczny model dyskretny; teoria Mori–Zwanziga stanowi motywację konstrukcji, ale nie oznacza, że wyznaczyliśmy dokładne jądro pamięci całej sieci.

Wspólne człony dodają taki sam przyrost wszystkim cząsteczkom. Potrafią więc opisywać wspólny ruch, lecz nie wyznaczają jeszcze różnej reakcji poszczególnych tokenów na ten sam kontekst ani wszystkich oddziaływań między nimi. Ocena dotyczy ośmiu współrzędnych tożsamości, nie rekonstrukcji całych 768 wymiarów każdej cząsteczki.

## Wynik na późniejszych aktualizacjach

W tabeli podano **redukcję średniego błędu kwadratowego względem prognozy „stan się nie zmieni”**. To nie jest odsetek poprawnych prognoz. Wartość ujemna oznacza wynik gorszy od takiej prognozy.

| Krok modelu | Horyzont w aktualizacjach | Poprzedni model | Nowy model wybrany na walidacji |
|---|---:|---:|---:|
| co 1 | 1 | 60,24% | 49,68% |
| co 1 | 10 | 9,11% | 14,88% |
| co 1 | 50 | −5,03% | 0,18% |
| co 10 | 10 | 16,12% | 15,58% |
| co 10 | 20 | 20,12% | 21,27% |
| co 10 | 50 | 13,28% | 21,07% |

Porównania w każdym wierszu mają te same początki prognoz i cele. Wiersze co1 i co10 mają różną liczbę początków prognoz; osobne porównanie obu dyskretyzacji na identycznych momentach znajduje się w METRYKI_DT1.json, pole common_co10.

**Model co1:** walidacja wybrała lokalny PLS z ośmioma dodatkowymi kierunkami, dwoma lagami pamięci i ridge1. Współrzędnych stanu jest 16 na cząsteczkę; wariant nie używa jawnego sprzężenia przez średnie. Średni wynik dla H1/H10/H50 na teście to **21,58% wobec 21,44%** poprzednio. Poprawa walidacyjna 28,90% wobec 24,92% nie przełożyła się więc na wyraźną ogólną przewagę. Na jednym kroku nowy model ma o 26,55% większy MSE. Trafność znaku wszystkich składowych wynosi 75,87% wobec wcześniejszych 78,61%.

Osobny wariant wybrany wyłącznie pod jeden krok też nie pobił wcześniejszego najlepszego modelu dla tego horyzontu: redukcja błędu 59,95% wobec 60,59%.

**Model co10:** walidacja wybrała sprzężony PLS z czterema dodatkowymi kierunkami i ridge10. Wybrała L0, czyli brak jawnego członu pamięci przyrostów w tej wersji. Nie można przedstawiać tego wyniku jako dowodu korzyści nowego jądra pamięci. Na H50 MSE jest o 8,99% mniejszy niż w poprzednim modelu. Różnica redukcji błędu to +7,79 punktu procentowego, z opisowym przedziałem bootstrap od −0,95 do +16,90 punktu. Przedział obejmuje brak przewagi.

Na teście niektóre inne wcześniej zamrożone rodziny dały lepsze wyniki na H50: coupledU8 10,58%, coupledPLS8 14,19% przy kroku co1. Nie zmieniamy na ich podstawie zwycięzcy wybranego na walidacji. Wyniki wszystkich rodzin są zachowane.

## Czy pomaga obserwacja wspólnego ruchu

Na fragmencie użytym do dopasowania, aktualizacje 501–700, **82,10% energii przyrostów w ośmiu współrzędnych tożsamości przypada na ruch wspólnej średniej cząsteczek narzędzia**, a 17,90% na ich ruch względem tej średniej. Średni cosinus między przyrostami średniej narzędzia i średniej kontekstu wynosi 0,738.

Obliczenie używa wag proporcjonalnych do udziału pikseli narzędzia w patchu, znormalizowanych do sumy jeden: w_i=coverage_i/sum_j coverage_j. Dla v_i=Δq_i i średniej v_bar zachodzi rozkład sum_i w_i||v_i||² = ||v_bar||² + sum_i w_i||v_i−v_bar||², następnie uśredniany po tych 200 przyrostach. To energia niecentrowanych przyrostów, a nie procent identyfikacji ani trafność prognozy.

**Wspólny ruch rzeczywiście istnieje i jest duży. Jego obserwacja ma zatem sens.** Nie oznacza jednak, że jawne dodanie średniej musi poprawić każdą prognozę: średnia jest funkcją stanów tokenów, a ich własna historia już zawiera część tej informacji. Dodatkowe kierunki również mogą przenosić ruch wspólny, a nie wyłącznie tożsamość.

## Drugi obraz i ograniczenia

Na drugim obrazie wykorzystano dokładnie te same operatory i bazy, tylko własną znaną historię i punkt odniesienia obrazu. Dla modelu co1 redukcja błędu wyniosła 30,52% na H1, −5,13% na H10 i −32,45% na H50. Dla modelu co10: −0,84% na H10 i −31,93% na H50. Nowe dłuższe prognozy są mniej błędne od starych, ale nadal przegrywają z utrzymaniem bieżącego stanu.

Główny obraz ma 31 cząsteczek narzędzia, drugi siedem. Oba pochodzą z jednego biegu uczenia, seed47. Drugi obraz jest odłożony dla dopasowania tego operatora dynamiki, ale pochodzi ze źródłowego zbioru treningowego. To nie jest niezależny trening ani dowód generalizacji na nowy zbiór.

Test obejmuje aktualizacje 800–1000, które były już analizowane we wcześniejszych eksperymentach. Wynik pozostaje eksploracyjny. Krótkie przedziały czasowe i silnie nakładające się horyzonty ograniczają wnioskowanie statystyczne; cząsteczki nie są niezależnymi powtórzeniami eksperymentu.

Najbardziej uzasadniony wniosek jest obecnie taki: **dodatkowy stan może pomagać w dłuższej prognozie, ale musimy poprawić przenoszenie dynamiki między obrazami i sposób wyboru modelu w czasie**. Samo dodanie wspólnej średniej i większej liczby kierunków nie rozwiązało problemu. Dzisiejszy test nie uzasadnia zastąpienia poprzedniego modelu jednym uniwersalnym nowym równaniem.

Również analiza reszt nie potwierdza domknięcia równania z białym szumem. Dla wybranego modelu co1 na głównym obrazie wszystkie 256 analizowanych szeregów reszt (cząsteczki i średnia, po osiem współrzędnych) mają nominalne p<0,05 testu Ljunga–Boxa. Są to zależne testy bez korekty wielokrotnych porównań, ale wynik wyraźnie przeczy deklaracji wybielenia reszt. Normalność sprawdzono osobno; gaussowskość i brak korelacji w czasie nie oznaczają tego samego. W modelu co10 szereg ma tylko 20 punktów, więc brak odrzucenia ma małą moc.

## Gdzie są obliczenia

- [WYNIKI.md](WYNIKI.md): pełny raport i kontrasty rodzin.
- [PLAN.md](PLAN.md): protokół zapisany przed testem nowych modeli.
- [AUDYT_ASTRA.md](AUDYT_ASTRA.md): niezależna kontrola metody i wyniku.
- WYBOR_DT1/10.json: wszystkie 450 konfiguracji walidacyjnych i zamrożony wybór.
- METRYKI_DT1/10.json: test obu obrazów, bootstrap i kontrasty z poprzednim modelem.
- MACIERZE_DT1/10.npz oraz MODELE_DT1/10.json: oszacowane operatory w oryginalnych jednostkach i ich opis.
- DANE_PROJEKCJE.npz i DANE.json: zamrożone bazy i małe projekcje, bez kopiowania pełnego pliku22GB.

Poprzedni model i jego prognozy odtworzono dokładnie, maxabs0. Kontrole ucięcia historii oraz zastąpienia przyszłych stanów NaN dały zgodne prognozy w 88 z 88 przypadków. Obejmują również dodatkowe współrzędne i kontekst. Kontrole te sprawdzają dostęp do przyszłości w procedurze prognozy; nie dowodzą ogólnej poprawności modelu fizycznego.
