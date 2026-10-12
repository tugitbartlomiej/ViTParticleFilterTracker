# Wiele obrazów, przejścia tylko między checkpointami

Zlecenie 2026-09-24: dla tego samego obrazu mierzyć przejścia z checkpointu na checkpoint, powtórzyć dla wielu obrazów, wyznaczyć macierz dla każdego odcinka i błędy dopasowania. Nie tworzyć przejść między różnymi obrazami.

## Dane i koszt

Lokalne 73 kolejne checkpointy odpowiadają aktualizacjom 5,10,...,365. Pomiędzy nimi są 72 odcinki. Wybieramy przed oceną po 6 wariantów z każdej z 12 grup: original, rotation90, length08, black, yellow, foreground_bright. Łącznie 72 obrazy i 5184 pary. Pozostałe warianty nie są używane w tej ograniczonej obliczeniowo analizie.

Pełny stan 576×384 float32: około 0,844 MiB na obraz. Ponowny odczyt 5256 stanów wymaga około 4,33 GiB. Dokładne współczynniki 72 operatorów w postaci dualnej wymagają około 5,7 GiB float64 plus średnie; stan dopasowania mieści się w RAM bez macierzy 221184×221184. RF-DETR pozostaje zamrożony i pracuje na lokalnym GPU z batchem 1. Dopasowanie operatorów wykonujemy na CPU, cztery wątki. Nie uruchamiamy nowych treningów RF-DETR ani FastGAN. Wersja ta jest afiniczna, nie deep Koopman.

Manifest sondy i wszystkie użyte obrazy/maski muszą być identyczne bajtowo z sondą Edena. Każdy checkpoint weryfikujemy przez SHA256. Pierwszą oryginalną sondę przy każdym checkpoincie porównujemy ze zrzutem CZASTKI z Edena; tolerancja względna 1e-4. Powtórny lokalny odczyt ma tolerancję 1e-5. Niezgodność przerywa analizę; nie zwiększamy tolerancji po zobaczeniu wyniku.

## Macierz odcinka t

z_(i,t) jest pełnym wektorem tego samego obrazu i w checkpoincie t. Dla każdego t tworzymy wiersze X_t=z_(i,t) i Y_t=z_(i,t+1) tylko z grup fit (8 grup, 48 wariantów).

Minimalizujemy ||(Y_t−X_t)_c−(X_t)_c B_t||²_F+λ_t||B_t||²_F. Operator K_t=I+B_t z poprawką stałą. λ_t=0,001 trace(Xc_t Xc_tᵀ)/48. Pełne wyprowadzenie, tożsamość dualna i granice gwarancji są w MATEMATYKA_OPERATORA_PRZYROSTOWEGO.md.

Nie wyznaczamy dowolnej osobnej macierzy z jednej pary, bo taka para jej nie identyfikuje. Wspólnie dopasowujemy K_t na wielu obrazach, ale w każdym wierszu X i Y obraz pozostaje ten sam. Nie uśredniamy surowych macierzy o dowolnej identyfikacji. Uśredniamy błędy, zachowując też każdą obserwację.

## Ocena

1. Błąd na 48 obrazach fit: dopasowanie.
2. Błąd na 24 wariantach z 4 oddzielnych grup: przeniesienie K_t na obrazy niewidziane podczas dopasowania tego operatora. Oba końce odcinka dla obrazów fit są znane. To nie jest prognoza przyszłego checkpointu.
3. Zastosowanie K_(t−1) do z_(i,t): rzeczywista prognoza kolejnego odcinka bez dopasowania do jego końca.
4. Kontrole: brak zmian, a od drugiego odcinka także powtórzenie poprzedniego przyrostu tego samego obrazu.

Dla każdego obrazu i odcinka zapisujemy MSE, RMSE, MSE z wagami maski narzędzia, cosinus przyrostu i stosunek jego długości. Nie zakładamy, że sama maska izoluje semantyczną tożsamość. Błędy skalarne wolno agregować, ale warianty tego samego źródła nie są niezależnymi filmami. Zachowujemy identyfikatory grup.

Zmianę macierzy mierzymy ||K_t−K_(t−1)||_F przez iloczyny czynników dualnych, bez materializacji pełnej macierzy. To zmiana dopasowanego operatora, nie bezpośredni pomiar zmiany „prawdziwego” operatora Koopmana. Większość kierunków pozostaje nieidentyfikowalna z 48 par.

## Wnioski dozwolone

Mały błąd na fit pokazuje możliwość odtworzenia przejścia w przyjętym modelu. Mały błąd na holdout wskazuje, że afiniczna mapa przenosi się na inne obrazy w badanym odcinku. Dobra prognoza poprzednim K wskazuje na lokalną stabilność dynamiki. Żaden z tych wyników sam nie dowodzi globalnej liniowości ani domknięcia tożsamości.

Jeśli affine zawodzi, można następnie porównać z nieliniowym/deep Koopman bez redukcji stanu. Trzeba ocenić go na tych samych podziałach, a nie ogłosić przewagi na podstawie mniejszego błędu treningowego. Ten etap nie został jeszcze uruchomiony przez niniejszy skrypt.
