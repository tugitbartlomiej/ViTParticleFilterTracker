# Współdziałanie neuronów a ramka — protokół przed obliczeniami

2026-10-15. Eksperyment na zamrożonych checkpointach; bez zmian aplikacji i treningu.

## Pytania i rozróżnienia

1. Czy pary neuronów ostatniego MLP wskazane przez kierunek ich wpływu na ramkę rzeczywiście kompensują albo wzmacniają skutki ablacji?
2. Czy przeciwne kierunki w 192-wymiarowym strumieniu residual wystarczają do przewidzenia kierunków zmian ramki?
3. Jaka część wyniku jest zwykłą sumą efektów, a jaka wynika z nieliniowości odczytu?
4. Czy te same pary działają podobnie na trzecim obrazie w późniejszych checkpointach?
5. Jak zmienia się ramka o najwyższej pewności i jej IoU, gdy wyłącza się te neurony we wszystkich 600 tokenach?

## Dane

Trzy różne obrazy z poprzedniego pomiaru `outputs/optyka_hipoteza_20261015/pomiar.npz`. Osiem tokenów najbliższych środkowi GT na obraz; 24 lokalne przypadki na checkpoint. Checkpointy najbliższe krokom 1000, 3000, ..., 65000 oraz ostatni 65640. Zapisane float32 stany zweryfikujemy ponownie przez pełny forward.

To jedna inicjalizacja i trzy obrazy, nie zbiór walidacyjny do oceny mAP. Filmy pochodzenia i niezależność obrazów nie są w tym eksperymencie ustalone. Wszystkie zakresy kroków obejmujemy bez filtrowania według jakości detekcji.

## Definicje

`u_c = z_c W2[:,c]` jest działaniem neuronu w residual 192D; `z` ma 768 składowych. Bias pozostaje w modelu. `b(x)` to cztery współrzędne xyxy w pikselach, z LayerNorm, MLP głowy i eksponentą dekodera. `a_c = J_b u_c` mierzy lokalny wpływ na ramkę. Dla ablacji siły alfa: `D_c = b(x-alfa*u_c)-b(x)`. Dla pary analogicznie `D_cd`.

- Kompensacja: `q = ||D_cd|| / (||D_c||+||D_d||)`; mała wartość oznacza maskowanie osobnych efektów.
- Interakcja nieliniowa: `I = D_cd-D_c-D_d`, względna wielkość `||I||/(||D_c||+||D_d||)`. Suma zgodnych efektów sama w sobie nie jest nieliniową synergią.
- Rozmiar przesunięcia ramki: RMS czterech współrzędnych, w pikselach.
- Jakość: zmiana IoU i RMS błędu xyxy względem GT. Duży wpływ nie oznacza poprawy jakości.
- Oddzielnie logit środka: pary kompensujące ramkę mogą zmieniać pewność.

## Dobór par i kontrole

Kandydaci: 64 neurony o największej normie `a_c`; pary o stosunku norm najwyżej 1,25. Po cztery rozłączne pary o najmniejszym/największym cosinusie `a_c,a_d` (functional_cancel/reinforce). Kontrole losowe z tej samej puli, dobrane do norm obu wpływów (do 20 procent odchylenia, przy braku — najbliższa dostępna para; zapisać odchylenie). Wykluczyć obie wybrane rodziny par z kontroli.

Porównania: pary skrajnych cosinusów `u_c,u_d` w tej samej puli, cztery pary kolejnych największych norm `a_c`, cztery pary największych norm `u_c`, cztery losowe pary z wszystkich 768 neuronów. Dobór kierunkowy nie korzysta z błędu względem GT. Kryterium gradientowe jest jawnym punktem odniesienia, a nie nową fizyką.

Dodatkowo zamrozić cztery pary każdego kierunku i ich kontrole na kroku 33000, wykorzystując wyłącznie pierwsze dwa obrazy: cosinus liczony między wektorami wpływów sklejonych po 16 tokenach. Trzeci obraz od 35000 służy do sprawdzenia transferu. Zamrożone identyfikatory nie są później dostrajane.

Siły ablacji: 0,1; 0,5; 1. Osobne i wspólne ablacje liczyć przez rzeczywistą głowę modelu. Dla globalnego wyłączenia wszystkich 600 tokenów użyć par wybranych na najbliższym środka GT tokenie; ponownie wybrać najpewniejszą detekcję. Zmiana tokenu zwycięskiego to osobna obserwacja. Wspólne pary i checkpointy nie są niezależnymi replikami; zestawiać wyniki per obraz i okres, bez p-wartości liczących pary jako niezależne próbki.

## Sprawdzenia i interpretacja

Kontrole: rekonstrukcja residual/head; porównanie interwencji z hakiem wyłączającym aktywację w pełnym forward; numeryczny JVP w float64; zgodność dekodowania najwyższej pewności z funkcją modelu. Zera i wartości niefinitywne jawnie kontrolowane. Zachować wynik także wtedy, gdy transfer nie wyjdzie.

To przyczynowe interwencje w obliczenie danego modelu. Nie dowodzą przyczyny zmian podczas treningu, istnienia fotonów/fazy optycznej ani biologicznej kooperacji. Pomiar kierunków i rzeczywistych ablacji ma sprawdzić użyteczność mikroskopu.

### Korekta techniczna przed ukończeniem biegu

Pierwszy bieg zatrzymał się przy zerowym skutku ablacji losowej pary (nieaktywny neuron). Nie odrzucamy tych interwencji. Zapisujemy ich przesunięcie, IoU i pewność, natomiast ilorazy z mianownikiem `||D_c||+||D_d|| <= 0,001 px` oznaczamy jako brak wiarygodnego pomiaru (`valid_ratio=false`). Próg chroni przed odejmowaniem niemal identycznych współrzędnych float32. Liczbę takich przypadków raportujemy osobno. Kryteria doboru par i podział danych pozostają te same.

### Kontrola precyzji po pierwszym kompletnym pomiarze

Interakcje par okazały się bardzo małe. Wszystkie lokalne odczyty głowy i Jacobiany przeliczamy więc w float64 z tymi samymi wagami i zachowanymi stanami residual/udziałami neuronów float32. Pozwala to usunąć dominujący szum odejmowania przy alfa=0,1. Zachowujemy konserwatywny próg 0,001 px. Globalny odczyt detektora pozostaje float32, zgodny z modelem użytkownika. Nie wybieramy wariantu numerycznego według wielkości efektu.

### Rozszerzenie eksploracyjne: grupy po wyniku par

Ponieważ efekty par są małe, dodatkowo sprawdzimy grupy 8, 16 i 32 neuronów na tych samych 34 checkpointach i trzech obrazach. Po 4, 8 lub 16 rozłącznych par kierunkowych wybierzemy z 128 najsilniejszych wpływów na ramkę tokenu najbliższego środkowi GT; stosunek norm w parze <=1,25. Porównamy całość grupy z osobnym usunięciem pierwszych i drugich członków par oraz z grupami największej czułości, największych norm residual i losowymi. Dobór do globalnych interwencji korzysta z tokenu środkowego, a ramka zwycięska zostaje ponownie wybrana. Wszystkie porównania grup są eksploracyjne i nie stanowią testu transferu ani optymalizacji detektora.
