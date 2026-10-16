# Gdzie leży cel tokenu 226 i dlaczego nie ma go jako punktu w latencie

Cel w wyjściu głowy: logit → +∞ (p = 1); ltrb = [4.380000114440918, 0.25, 4.420000076293945, 1.1699999570846558] kratek, czyli o[1:] = ln ltrb = [1.4759999513626099, -1.3860000371932983, 1.4850000143051147, 0.15399999916553497].

## krok 1100 (|x| = 33.0)
- teraz: logit -0.18 (p = 0.4554), ltrb [4.75, 0.4399999976158142, 4.510000228881836, 1.2400000095367432]
- A) najwyższy logit osiągalny samym ruchem tokenu (ramka = cel): +1.86 (p = 0.8651), ltrb [4.369999885559082, 0.27000001072883606, 4.420000076293945, 1.1699999570846558]
- B) najbliższy punkt z ramką = cel i logitem +1: |Δx| = 6.83 (20.7% |x|), 94% w 5 kierunkach czytanych; osiągnięte: logit +0.77, ltrb [4.380000114440918, 0.36000001430511475, 4.420000076293945, 1.149999976158142]
- C) rząd jakobianu głowy 5 → zbiór punktów o tej samej odpowiedzi ma 187 wymiarów

## krok 10100 (|x| = 53.9)
- teraz: logit +2.33 (p = 0.9114), ltrb [3.869999885559082, 0.15000000596046448, 4.409999847412109, 0.9800000190734863]
- A) najwyższy logit osiągalny samym ruchem tokenu (ramka = cel): +5.31 (p = 0.9951), ltrb [4.380000114440918, 0.25999999046325684, 4.420000076293945, 1.159999966621399]
- B) najbliższy punkt z ramką = cel i logitem +1: |Δx| = 7.83 (14.5% |x|), 97% w 5 kierunkach czytanych; osiągnięte: logit +3.29, ltrb [4.349999904632568, 0.23000000417232513, 4.420000076293945, 1.100000023841858]
- C) rząd jakobianu głowy 5 → zbiór punktów o tej samej odpowiedzi ma 187 wymiarów

## krok 28444 (|x| = 60.9)
- teraz: logit +2.15 (p = 0.8961), ltrb [4.389999866485596, 0.23000000417232513, 4.46999979019165, 1.1100000143051147]
- A) najwyższy logit osiągalny samym ruchem tokenu (ramka = cel): +6.27 (p = 0.9981), ltrb [4.380000114440918, 0.25999999046325684, 4.420000076293945, 1.159999966621399]
- B) najbliższy punkt z ramką = cel i logitem +1: |Δx| = 6.20 (10.2% |x|), 98% w 5 kierunkach czytanych; osiągnięte: logit +3.11, ltrb [4.369999885559082, 0.25999999046325684, 4.409999847412109, 1.1299999952316284]
- C) rząd jakobianu głowy 5 → zbiór punktów o tej samej odpowiedzi ma 187 wymiarów

## krok 46800 (|x| = 60.6)
- teraz: logit +2.65 (p = 0.9338), ltrb [4.440000057220459, 0.3199999928474426, 4.789999961853027, 1.149999976158142]
- A) najwyższy logit osiągalny samym ruchem tokenu (ramka = cel): +6.81 (p = 0.9989), ltrb [4.380000114440918, 0.27000001072883606, 4.409999847412109, 1.159999966621399]
- B) najbliższy punkt z ramką = cel i logitem +1: |Δx| = 6.63 (10.9% |x|), 96% w 5 kierunkach czytanych; osiągnięte: logit +3.60, ltrb [4.369999885559082, 0.3199999928474426, 4.440000057220459, 1.1299999952316284]
- C) rząd jakobianu głowy 5 → zbiór punktów o tej samej odpowiedzi ma 187 wymiarów

## krok 65640 (|x| = 56.6)
- teraz: logit +2.93 (p = 0.9495), ltrb [4.5, 0.3499999940395355, 4.329999923706055, 1.1299999952316284]
- A) najwyższy logit osiągalny samym ruchem tokenu (ramka = cel): +6.89 (p = 0.9990), ltrb [4.380000114440918, 0.25999999046325684, 4.420000076293945, 1.159999966621399]
- B) najbliższy punkt z ramką = cel i logitem +1: |Δx| = 5.90 (10.4% |x|), 95% w 5 kierunkach czytanych; osiągnięte: logit +3.89, ltrb [4.380000114440918, 0.3400000035762787, 4.409999847412109, 1.1200000047683716]
- C) rząd jakobianu głowy 5 → zbiór punktów o tej samej odpowiedzi ma 187 wymiarów
