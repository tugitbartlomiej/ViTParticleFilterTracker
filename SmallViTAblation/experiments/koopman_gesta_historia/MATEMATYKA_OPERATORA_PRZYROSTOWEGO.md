# Operator pełnego latentu z zachowaniem stanu i uczeniem przyrostu

Zlecenie: lokalnie policzyć przejście bez PCA, wykorzystać więcej dostępnych stanów i usunąć sztuczne tłumienie składowych słabo opisanych przez historię. Nie obiecujemy dokładnego poznania dowolnej macierzy z jednej krótkiej trajektorii.

## 1. Co zmieniamy

Stan z jest wektorem wierszowym zawierającym wszystkie 576×384=221184 współrzędne. Sąsiednie stany dzieli pięć aktualizacji optymalizatora. Nie uśredniamy cząstek. Poprzednia metoda minimalizowała błąd następnego stanu z karą na wielkość macierzy przejścia. Nowa kara dotyczy odchylenia macierzy od identyczności. Zachowanie stanu jest jawnym założeniem modelu, uzasadnionym małymi zmianami między checkpointami, a nie wnioskiem z formalizmu Koopmana.

Mając wiersze X=z_t i Y=z_(t+1), definiujemy D=Y−X, średnie μ=mean(X), d=mean(D), Xc=X−μ, Dc=D−d. Rozwiązujemy:

    min_B ||Dc − Xc B||_F² + λ ||B||_F²,  λ > 0.

Model:

    F(z) = z + d + (z−μ) B = z K + b,
    K = I + B,  b = d − μB.

To nadal afiniczny liniowy model w pełnych współrzędnych. Na rozszerzonym stanie [z,1] odpowiada mu macierz [[K,0],[b,1]]. Jest skończonym przybliżeniem działania Koopmana na obserwowanych współrzędnych; dokładna domkniętość nie jest założona jako fakt.

## 2. Rozwiązanie i obliczenia bez wielkiej macierzy

Zerowanie pochodnej funkcji celu daje:

    (XcᵀXc + λI)B = XcᵀDc.
    B = Xcᵀ (XcXcᵀ + λI)⁻¹ Dc = Xcᵀ C.

Równość wynika z (XcᵀXc+λI)Xcᵀ=Xcᵀ(XcXcᵀ+λI). Dodatnia λ daje jednoznaczne rozwiązanie. W kodzie rozwiązujemy układ liniowy dla C, bez jawnego odwracania wielkiej macierzy. Dokładność pozostaje ograniczona arytmetyką zmiennoprzecinkową.

Macierz K miałaby około 364,5 GiB w float64. Zapisujemy zamiast niej I, Xc i C, a także μ i d. To dokładna reprezentacja wybranego regularizowanego operatora, nie PCA i nie obcięcie SVD. Przy 160 parach każdy czynnik 160×221184 ma około 270 MiB. Historia float64 ma około 272 MiB; cały proces potrzebuje dodatkowych buforów, ale nie setek GiB. RAM jest kontrolowany lokalnie, obliczenia na CPU z czterema wątkami.

## 3. Co potrafimy zagwarantować

Niech perturbacja v spełnia vXcᵀ=0, czyli historia wejściowa nie dostarczyła informacji o zmianach w jej kierunku. Wówczas:

    F(z+v) − F(z) = v + vXcᵀC = v.

Zatem model dokładnie zachowuje taką perturbację wejścia zamiast ją zerować. Poprzednia regresja stanu zwracała dla niej różnicę zero. Gwarancja dotyczy zachowania różnicy wejściowej; nie oznacza, że dynamika rzeczywista w tym kierunku jest zerowa, ani że dryf nie może wnieść składowej w tym kierunku.

Regularyzacja nadal tłumi niepewne współczynniki PRZYROSTU. Nie wolno twierdzić, że usunięto każdą formę tłumienia. Rezygnacja z λ przy krótkiej skorelowanej historii może prowadzić do niestabilności i niedookreślenia. Zmieniliśmy punkt odniesienia z zerowej mapy na identyczność, a nie zwiększyliśmy informacji ponad tę obecną w danych.

## 4. Prognoza wielokrokowa

Niech q=(z−μ)Xcᵀ. Jeden krok:

    z_next = z + d + qC,
    q_next = q + dXcᵀ + q(CXcᵀ).

Sumując kolejne q otrzymujemy prognozę pełnego z po h krokach. Pozwala to iterować małą macierz pomocniczą, zachowując wszystkie współrzędne wejścia i wyjścia. Test numeryczny porównuje wynik z bezpośrednią iteracją pełnej macierzy w małym sztucznym układzie. Nie interpretujemy rozmiaru macierzy pomocniczej jako redukcji liczby współrzędnych stanu.

Wartości własne K inne niż jednostkowe pochodzą z I+CXcᵀ. Pozostałe jednostkowe wartości własne wynikają z założenia identyczności na niewyznaczonych kierunkach. NIE są odkrytymi niezmiennikami tożsamości narzędzia.

## 5. Jak wpuszczamy więcej danych bez podglądania przyszłości

W lokalnych plikach jest pełna historia jednego obrazu, 161 stanów. Używamy oddzielnie:

- stałego operatora z 100 par do aktualizacji 500;
- operatora aktualizowanego: 100 par do 500, 120 do 600, 140 do 700. Dla każdej prognozy używamy tylko operatora z danych dostępnych w jej początku;
- końcowego operatora ze wszystkich 160 par do 800. Prognozy 805 i 900 zapisujemy jako niesprawdzone.

Ocena kontrolowana: te same początki 500–700 i horyzonty 5 oraz 100 aktualizacji; brak zmian, ostatni przyrost i trend z poprzednich 100 aktualizacji. λ=0,001·trace(XcXcᵀ)/N, bez przeszukiwania po wyniku. To diagnostyczne rozwinięcie po obejrzeniu wcześniejszych rezultatów; nie jest niezależnym potwierdzeniem skuteczności metody. Wariant aktualizowany ma więcej obserwacji i nowsze obserwacje jednocześnie — nie izoluje tych dwóch przyczyn.

Więcej punktów jednej trajektorii nie zastępuje nowych obrazów ani niezależnych przebiegów uczenia. Po pobraniu materiałów można dołączać pełne stany innych obrazów, zachowując rozdzielenie obrazów fit i holdout. Wspólny operator między obrazami również jest hipotezą do sprawdzenia. Nie wolno traktować tokenów jako niezależnych pełnych stanów obrazu, żeby sztucznie zwiększyć liczbę par.

## 6. Implementacja i wyniki

- `koopman_przyrostowy.py`: operator i sześć testów numerycznych.
- `analiza_przyrostowa_full.py`: SHA256, zamrożony protokół, obliczenia i raporty.
- `wyniki/koopman_przyrostowy_full/RAPORT.md`: wyniki po ukończeniu.
- `OPERATOR_0500.npz`, `OPERATOR_0800.npz`: dokładne czynniki operatorów.
- `FUTURE_UNVERIFIED.npz`: pełne prognozy poza dostępną historię, niezweryfikowane.

Nie jest to formalna weryfikacja w Lean. Dowód dotyczy rozwiązania regresji i zachowania niewidzianych perturbacji, nie dokładności prognozy rzeczywistej sieci. Nie wykonuje się tu testu tożsamości ani generowania FastGAN.
