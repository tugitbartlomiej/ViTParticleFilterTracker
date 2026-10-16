# Wykres przeciwnych reflektorów — plan analizy

2026-10-15. Analiza zapisanych historii i wag, bez zmian aplikacji.

Źródło: `lasery_dane.py`, `PARY = mean(min_{d!=c} cos(v_c,v_d)<-0.5)`, gdzie v to kolumny fc2 ostatniego MLP, w 192D. To udział wierzchołków mających przynajmniej jednego partnera, nie udział wszystkich par ani aktywna kompensacja tokenu. Źródło rysunku: `zakladka_laserow.py`, wykres 9; x jest indeksem checkpointu z etykietami kroków/epok. Końcowa epoka 60 odpowiada 65640 krokom, 1094 na epokę.

Analiza opisowa, jedna inicjalizacja. Główne porównania: P versus IoU i pewność najwyżej ocenionej ramki, osobno dla trzech obrazów. Pomocnicze: geometria wag, mierzone wcześniej aktywne sumowanie wkładów, strata treningowa oraz dostępne oceny mAP. Nie traktować obrazów jako trzech niezależnych powtórzeń tego samego P.

1. Zweryfikować zapisane P ponownym odczytem wag. W pełnej osi checkpointów znaleźć pierwszy dodatni wynik i przekroczenia progów, a tempo liczyć w rzeczywistych krokach.
2. Do głównych korelacji użyć dokładnie istniejących checkpointów co 100 kroków, od 1000 do 65600. Osobno pokazać start treningu, by zagęszczone zapisy początkowe nie dominowały wyniku.
3. Korelacje poziomów Pearson i Spearman; przyrosty co 100, 500 i 1000 kroków; usunięcie gładkiego trendu czasu przez spline (6 węzłów, kontrolnie 4 i 10). Brak nominalnych p-wartości traktujących szeregi jako niezależne obserwacje. Wyniki fazowe na ustalonych granicach epok 0,5,15,30,45,60.
4. Opis zmian tempa: dopasowanie ciągłej łamanej o trzech odcinkach do równomiernie próbkowanego P, siatka punktów załamania co 500 kroków, minimalny odstęp 3000. Kontrola siatki 1000. Załamania są parametrami opisu, nie dowodem przemiany fazowej.
5. Obok progu -0.5 policzyć -0.4 i -0.6, średni minimalny cosinus, udział wszystkich przeciwnych par i efektywny rząd kierunków. Pomaga to odróżnić próg umowny od zmiany geometrii.
6. Aktywne sumowanie sprawdzić na poprzednich float32 pomiarach: mediana wskaźnika `||sum u||²/sum||u||²` w ośmiu tokenach środkowych, osobno dla obrazów. Ta miara ma inny mianownik i znaczenie niż P; nie nazywać jej procentem przeciwnych reflektorów.

Raport powinien wskazać ograniczenia: współzmienność podczas jednego treningu, zależność obserwacji w czasie, arbitralny próg kąta, metryki detekcji na trzech obrazach oraz odrębność zmiany wag i aktywnych wkładów. Żadna z korelacji nie identyfikuje mechanizmu przyczynowego.
