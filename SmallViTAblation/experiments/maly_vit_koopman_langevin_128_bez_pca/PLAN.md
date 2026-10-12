# Koopman i Langevin na tych samych 128 oryginalnych kanałach

Data: 2026-10-02. Rozszerzenie porównania na wyraźne polecenie użytkownika. Jedno pytanie: jak prognozują oryginalne 128 kanałów lokalny Langevin z pamięcią i liniowe przybliżenia Koopmana bez pamięci? Nie wybieramy kanałów ponownie, nie robimy PCA, POD, rotacji ani autoenkodera.

Źródło danych i zamrożonych modeli: sąsiedni eksperyment `maly_vit_tozsamosc90_bez_pca_langevin`. J zawiera dokładnie wcześniej wybrane 128 osi, wspólnie osiągające punktowo 92.83 procent rozpoznania oryginał/LaMa w znanym ROI na k500. Prognoza ruchu to inne zadanie. Lista J oraz sumy danych muszą być zgodne ze źródłem.

## Modele i informacja wejściowa

1. Langevin: wcześniej wybrany lokalny model przyrostów z pamięcią L=2, ridge=1 dla dt1 i ridge=10 dla dt10. Własna implementacja prognozy odtwarza zamrożone macierze, bez nowego strojenia.
2. Koopman przyrostowy: affine x_next = K x + b, K = I + A z wcześniej wybranego wariantu L0. To już istniejący model bez pamięci, teraz zapisany i liczony jawnie jako samodzielny operator. Nie przedstawiamy zmiany nazwy jako nowej metody. Niezależny fit z karą ku K=I musi odtworzyć macierze źródłowe.
3. Koopman bezpośredni: osobno dopasowany operator x_next = K x + b z regresją ridge ku K=0. Obserwowane funkcje to stała i wszystkie 128 oryginalnych współrzędnych. Nie zakładamy dokładnego domknięcia przestrzeni przez operator Koopmana. Siatka ridge=0.0001,0.001,0.01,0.1,1,10; wybór wyłącznie na walidacji. Każdy wariant zachowuje pełną macierz 128×128 i wyraz wolny. Dwa warianty Koopmana mają tę samą klasę funkcji, różnią się regularyzacją.
4. Kontrole: brak zmiany oraz stała prędkość. Zachowują ten sam cel, kanały, obrazy, starty i wagi pokrycia maską.

Wszystkie modele mają dostęp do zakończonej historii. Langevin używa dwóch przyrostów; Koopman bez pamięci tylko ostatniego stanu, a stała prędkość ostatniego przyrostu. Różnica pamięci jest badaną cechą, nie wyciekiem przyszłości. Nie podajemy przyszłych obrazów, batchy, gradientów ani prawdziwych przyszłych stanów. W prognozie punktowej przyszła reszta Langevina wynosi zero; białość reszt nie została wcześniej potwierdzona.

## Zamrożony protokół przed nowymi dopasowaniami

Stany x_i(t)=z_i,J(t)-z_i,J(500). Główna sonda2,31 cząsteczek; transfer sonda8,7 cząsteczek, bez refitu. Wspólne macierze dla tokenów. Wagi pokrycia maską i wszystkie 128 współrzędnych są zachowane. Obie sondy pochodzą z jednego przebiegu uczenia.

dt1: fit target501..700, validation origins700..749 i H1/10/50; po wyborze refit do799. Test origins800..950 i H1/2/5/10/20/50. dt10: fit510..700co10, validation700..740co10 i H10/50; refit do790. Test800..950co10 i H10/20/50. Kryterium Koopmana bezpośredniego: największa średnia redukcja MSE na zadeklarowanych horyzontach, remis rozstrzyga mniejsze ridge. Zamrożenie i zapis macierzy przed oceną testową.

Metryki: fizyczny MSE, redukcja MSE względem braku zmiany, znak zmian wszystkich współrzędnych, cosinus rzeczywistej i prognozowanej zmiany. Przedziały blokowe mają status opisowy z jednego krótkiego przebiegu. Test był oglądany w poprzednich analizach; nowe dopasowanie przestrzega podziału, ale porównanie jest eksploracyjne, bez nowego niezależnego potwierdzenia. Nie zmieniamy siatki po zobaczeniu wyniku.

Kontrole techniczne: dokładny zestaw128 osi; zgodność macierzy i prognozy przyrostowego Koopmana z dawnym L0; zgodność nowej implementacji Langevina z zapisanymi prognozami; prefix/poison przyszłości; pełny wymiar celu; finite; hashe. Rozkład macierzy do rozwiązania regresji nie może usuwać lub obracać współrzędnych. Nie kopiujemy źródłowych22GB, nie zmieniamy wcześniejszych wyników, bez treningu ViT i bez pobierania.

Wynik poniżej kontroli jest zachowywany i raportowany. Skończony wynik na H50 nie jest dowodem stabilności asymptotycznej ani dokładności na H500.
