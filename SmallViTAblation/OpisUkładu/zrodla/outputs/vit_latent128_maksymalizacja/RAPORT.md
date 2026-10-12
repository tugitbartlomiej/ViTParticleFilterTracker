# Silna optymalizacja 128 kanałów: wynik

Checkpoint1000; 400 obrazów; 800 kroków Adam; lr 0.3; NVIDIA GeForce RTX 3070; czas 448.2s.

| Metoda | AP50 | AP75 | AP50:95 | Strata |
|---|---:|---:|---:|---:|
| Bez zmiany | 0.437555 | 0.021881 | 0.120881 | 2.384983 |
| Optymalizacja J128 | 0.884748 | 0.774186 | 0.770525 | 0.208414 |
| Kontrola znaków | 0.873660 | 0.752233 | 0.756869 | 0.230241 |
| Dodatkowe 64 | 0.850983 | 0.751623 | 0.736021 | 0.310655 |

Cel opisowy AP50>=0.95: nieosiągnięty.
Wynik IoU>=0.5 i confidence>=0.9 dla najpewniejszej ramki: 0.50% obrazów.
Największa bezwzględna zmiana współrzędnej: 201.511490; minimalna wartość docelowa: -0.16997123.

Wagi i bufory pozostały identyczne. Sprawdzono zachowanie640 kanałów, zakres GELU, odtworzenie zmodyfikowanego forward na obrazach0/199/399.

To indywidualna optymalizacja aktywacji400 obrazów z ich etykietami. Nie są to nowe wyuczone wagi ani wynik generalizacji. Nie znaleziono obrazów realizujących te latenty. Zakres GELU jest zachowany, lecz duże aktywacje mogą leżeć daleko poza typowym rozkładem. Maksimum globalne nie zostało udowodnione.

Wcześniejsze AP0.226→0.277 dotyczyło k500,85 obrazów i małych ograniczeń. Nie porównywać bezpośrednio z k1000/400 jako efektu samego zwiększenia liczby kroków.

TARGETS_FULL400.npz zawiera wszystkie wartości per obraz/token/kanał. KANALY128.csv to statystyki, nie uniwersalne wartości docelowe. POROWNANIE.png i PRZYKLADY.png pokazują metryki oraz stałe przykłady.

## Próba dalszej poprawy
Osobny wariant600 kroków na64 najtrudniejszych obrazach dał AP50=0.850983, wobec0.884748 wariantu podstawowego. Nie zastępuje wyniku głównego. Cele odrębne: TARGETS_REFINED400.npz.

Kontrola znaków zachowuje moduły zmian dopasowane do etykiet; znaki naruszające GELU są zastępowane dodatnimi. Nie jest to losowa interwencja niezależna od etykiet.

Obliczenia i zapis głównych artefaktów zakończyły się poprawnie. Końcowy zapis do rejestru początkowo odmówił serializacji ndarray; poprawiono wyłącznie serializację i zarejestrowano istniejący RESULT.json bez ponownego liczenia.
