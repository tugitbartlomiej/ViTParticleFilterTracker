# Dlaczego osobny Koopman nie przewiduje (diagnoza przyczyn, 2026-09-25)

Skrypt: `diagnoza_przyczyny.py` (py -3.11, numpy float64). Próbka: 72 obrazy × co 16. token = 2592 trajektorie,
ten sam podział co w PLAN.md (dopasowanie na 48 parach do kroku 245, test 24 przejść do kroku 365).

## Pomiary

| Pomiar | Wartość |
|---|---|
| mediana długości przyrostu, początek → koniec historii | 0,295 → 0,091 (uczenie zwalnia) |
| cos kolejnych przyrostów (bezwładność), początek / test | 0,964 / 0,903 |
| udział stanu testowego poza przestrzenią rozpiętą przez stany uczące | 0,04 |
| wyjście testu poza zakres uczenia wzdłuż głównej osi trajektorii | 0,15 długości zakresu |
| średni dryf okna uczenia / prawdziwy przyrost w teście | 1,33 |
| długość prognozy Koopmana / prawdziwy przyrost | 1,62 |
| cos(prognoza Koopmana, średni dryf okna uczenia) | 0,66 |

RMSE jednego kroku na teście (ta sama próbka):

| Model | RMSE |
|---|---|
| Koopman, grzbiet 0,001 | 0,0275 |
| Koopman, grzbiet 0,1 (wybierany w PLAN) | 0,0100 |
| sam średni dryf z okna uczenia | 0,0089 |
| brak zmian | 0,0080 |
| Koopman, grzbiet 10 | 0,0070 |
| Koopman na prędkościach (v_{t+1} = A v_t + c) | 0,0055 |
| ostatni przyrost | 0,0037 |
| alpha × ostatni przyrost | 0,0036 |

## Wnioski

1. **Trajektoria zwalnia, a stała macierz tego nie widzi.** Przyrosty maleją ponad trzykrotnie. Macierz uczona
   na wcześniejszym oknie przewiduje ruch zgodny ze średnim dryfem tego okna (cos 0,66) i o 62 procent za duży.
2. **Test leży za końcem trajektorii uczącej.** Trajektoria się nie powtarza, więc każdy nowy stan jest dalej niż
   wszystkie znane. Część zależna od stanu musi ekstrapolować, a grzbiet ściąga ją do średniego dryfu.
   Im silniejszy grzbiet (im mniej macierz robi), tym lepiej: 0,0275 → 0,0100 → 0,0070.
3. **Następny krok zależy od prędkości, nie od położenia.** Kolejne przyrosty są prawie równoległe (cos 0,90).
   To bezwładność optymalizatora (moment w AdamW) i powoli zmieniający się gradient. Stan złożony z samego
   położenia wektora tej informacji nie zawiera, więc model pierwszego rzędu jej nie odtworzy.
   Koopman na prędkościach od razu poprawia wynik z 0,0100 do 0,0055.
4. **Stan nie jest domknięty.** O ruchu tokenu decyduje gradient liczony z całego zbioru i wszystkich wag,
   a nie tylko 384 liczby tego tokenu. W języku Mori–Zwanziga brakuje członu pamięci; ostatni przyrost jest jego
   najprostszym zastępstwem i dlatego wygrywa.

## Co zrobić

- Stan z opóźnieniem: [z_t, z_t − z_{t−1}] (albo kilka ostatnich przyrostów), czyli Koopman na zanurzeniu
  opóźnieniowym, z dopasowaniem zaniku długości przyrostu.
- Model przyrostów z malejącą skalą, np. v_{t+1} = alpha_t v_t z alpha zależnym od czasu uczenia.
- Siatka grzbietu szersza niż {0,001; 0,1}; tu wygrywa słaba macierz.
- Porównywać zawsze z alpha × ostatni przyrost, bo to obecnie najtrudniejsza podłoga.
