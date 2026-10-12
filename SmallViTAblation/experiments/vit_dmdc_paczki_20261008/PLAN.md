# DMDc: ruch tokenów przy rzeczywistych paczkach

Zatwierdzony zakres: nowa analiza istniejących pomiarów oraz, jeśli potrzebne,
nowy pomiar wejść. Nie zmieniamy GUI ani danych historycznych.

Stan: wspólny wektor 3 x 600 x 768 z trzech stałych sond (2, 3, 6).
Zegar: jedna aktualizacja AdamW. Analiza stałego LR: 300--1000.
POD (metoda snapshotów na GPU) uczy się wyłącznie z klatek czasu 300--700.
Przejścia train 300--699, validation 700--799, test 800--999.
Dobór rangi i regularyzacji tylko na validation, wyniki test po zamrożeniu wyboru.

Wejścia: rzeczywiste paczki k wpływają na przejście k -> k+1.
Rozważono dodatkowe ramię opisu składu paczki, augmentacji i GT (proxy, bez
gradientu). Po odnalezieniu kompletnych zapisanych sił zakres właściwego biegu
obejmuje wejścia siłowe; ramię proxy nie jest częścią wykonanego eksperymentu.
Ramię sił korzysta z uprzednio obliczonego, zgodnego czasowo pchnięcia S_nowa
(kontrfaktyczna zmiana cech z części nowego gradientu AdamW, przy stanie pre-update).
Źródło sił musi mieć własne zgodne Z; nie łączymy osobnych treningów po samym seed.
Nie używamy dz ani cech po kroku jako wejścia.

Modele: persystencja, stała prędkość, autonomiczne DMD, PyDMD.DMDc oraz
regularyzowane DMDc z rozszerzeniem stanu o ostatnią prędkość i z lagiem wejścia.
Jeżeli dostępne siły, obowiązkowe porównanie z bezpośrednim dodaniem S_nowa
i prostym modelem v_next = beta*v + b*S_nowa.

Każda prognoza zakotwiczona w pełnym stanie początkowym, następne zmiany
odtwarzane z POD do wszystkich 768 składowych każdego tokenu. Raport RMSE,
SSE skill względem persystencji, cosinus ruchu, horyzonty 1/5/10 oraz grupy
narzędzie/tło. Wielokrokowe przewidywanie z przyszłymi wejściami to prognoza
warunkowa przy znanych wejściach; przy siłach liczonych na prawdziwych stanach
także przy dostępności tych stanów. Bez twierdzenia o przewidywaniu nieznanych paczek.

Pomiary kosztu: czas, proces, wersje, pamięć GPU peak. Pilot przed pełnym biegiem.
Macierz stanu 1382400 x 1382400 nigdy nie powstaje.
