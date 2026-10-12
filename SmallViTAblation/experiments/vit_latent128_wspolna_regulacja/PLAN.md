# Jedna wspólna regulacja J128

2026-10-05. Użytkownik zlecił napisanie programu i wykonanie poszukiwania na RTX 3070. Wagi i bufory ViT k1000 zamrożone. Cel: wspólna reguła poprawiająca odczyt detektora, bez osobnych uczonych parametrów dla zdjęcia lub pozycji tokenu.

## Dane i rozdział informacji

Istniejące 400 oznaczonych obrazów; zachowany historyczny podział train/val/test po filmach. Do gradientów tylko train, do wyboru ustawień tylko val. Test dopiero po zamrożeniu wyboru. Są to historycznie oglądane obrazy, więc jest to pilot transferu między filmami, nie nowy niezależny test całego programu badań. Wybrane J128 z k500 pozostają stałe przy k1000. Nie wybieramy kanałów ponownie na val/test.

Wejście 480×320, 600 tokenów po 768 składowych po GELU ostatniej MLP. Cache pre-GELU J128 i pełnego stanu po residualu liczony raz na zamrożonych wagach. Statystyki kanałów tylko z train. Inferencja regulatora nie dostaje etykiet, ID obrazu, masek ani indeksu tokenu.

## Modele wspólne

1. Baseline: brak zmiany.
2. Shift: przesunięcia przed GELU, 128 parametrów.
3. Affine: skala i przesunięcie przed GELU, 256 parametrów. Inicjalizacja identycznościowa. z'_j=GELU((1+a_j)h_j+b_j sigma_j), j w J128.
4. Warunkowy adapter rank16 jako osobne, bardziej ekspresyjne ramię: ten sam mały MLP 128→16→128 dla każdego tokenu, korekta pre-GELU zależna od jego reprezentacji. Wspólne wagi, żadnych zmiennych per obraz. To rozszerzenie, nie stały wektor kanałów; raportować oddzielnie.

GELU utrzymuje zakres aktywacji. Pozostałe 640 kanałów nietknięte. Zamrożona pełna sieć w eval, bez aktualizacji BN. Rzeczywisty pełny forward z regulatorem musi zgadzać się z przyspieszonym obliczeniem ostatniej warstwy, również gradienty.

## Wstępnie ustalone wyszukiwanie

Adam, batch16, 120 epok (około 1800 aktualizacji na konfigurację przy 240 train). Konfiguracje: shift lr0.01 lambda0.001; affine lr0.003/0.01/0.03 lambda0.001 oraz affine lr0.01 lambda0.01; adapter16 lr0.003/0.01 lambda0.001. Łącznie 7 konfiguracji. Ziarno20261005, jednakowe kolejności minibatchy. LR cosine do 10% wartości początkowej. Gradient clipping norm5. Kara średniej kwadratowej standaryzowanej zmiany pre-GELU, nie normy wag ViT.

Strata: per-image focal heatmap +2 GIoU z poprzedniego aparatu, równa masa filmów w agregacji. Co10 epok pomiar AP50, AP50:95, loss na val, również epoka0 jako kandydat. Wybór po najwyższym AP50:95, następnie AP50, następnie mniejszym loss; pełny globalny wybór spośród 7 konfiguracji przed odczytem końcowego testu. Osobno wybór najlepszego statycznego regulatora (shift/affine). Wynik negatywny albo wybór identyczności pozostaje wynikiem.

## Podłoga, dostęp i obalenie

Baseline ma te same obrazy, preprocessing, wagi, dekoder top10 i metrykę. Wszystkie uczone warianty mają ten sam split i loss. Różnią się liczbą parametrów, dlatego ocena poza próbką. Wynik oracle0.884748 jest wyłącznie historycznym punktem odniesienia, nie celem z gwarancją ani kontrolą generalizacji.

Sukces pilota: dodatnia zmiana AP50:95 i AP50 na odłożonych filmach dla wariantu wybranego na val. Raportować wszystkie wyniki i paired bootstrap całych filmów (500 powtórzeń) dla AP50, AP50:95 oraz loss. Brak przewagi oznacza niepotwierdzoną użyteczność tej regulacji. Jeden seed nie jest dowodem uniwersalności.

## Zasoby i artefakty

Lokalny RTX3070, bez zatrzymywania innych procesów; cache cech około310MB float32, bez treningu backbone. Najpierw preflight porównujący pełny forward i szybki tail, niezmienność kanałów, skończony gradient, rozłączność filmów, hash wag. Potem zaplanowany sweep. Zapisy: konfiguracja i hashe przed startem, historia po epokach, model każdej konfiguracji, zamrożony wybór, wynik testu, pełne predykcje, kanały/wzmocnienia/przesunięcia najlepszego statycznego wariantu, kod wykonany, logi i raport.

Testu nie używać do dobierania kolejnej konfiguracji po tej kampanii. Nowa iteracja wymaga jawnego oznaczenia adaptacji i nowej oceny.
