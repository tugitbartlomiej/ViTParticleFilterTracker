# Dobór paczek ku staremu celowi K1000: wynik próby

## Co rzeczywiście wykonano

256 oznaczonych obrazów z manifestu35000, 32 paczki po8, równomiernie między filmami. Film sondy wykluczony. Dla każdej paczki prognoza powstała przed rzeczywistym krokiem; każdy krok zaczynał z tych samych wag K1000 i oryginalnego stanu AdamW. Po eksperymencie kopia przywrócona do K1000. Żaden źródłowy checkpoint nie został nadpisany. Brak zapisu zdjęć i latentów.

Pełne600x768 porównano ze starym celem zgodnym z K1000; zmienione128 kanałów raportowano oddzielnie od pozostałych640. Forward i gradienty FP32. Nie było podmiany aktywacji przy mierzeniu wyników rzeczywistego kroku.

## Wyniki

| Pomiar | Wynik |
|---|---:|
| Czas pilota | 33,15 s |
| Mediana redukcji błędu prognozy przyrostu J128: klasyczne rozdzielenie gradientu i pędu | 98,60% |
| Mediana redukcji błędu prognozy przyrostu J128: połączony krok z weight decay | 98,74% |
| Korelacja rang prognoza–rzeczywisty efekt | 0,9674 |
| Wybrana paczka | 29; faktycznie najlepsza z32 |
| Paczki zmniejszające odległość do celu | 0/32 |
| Początkowe MSE128 | 85,92118 |
| MSE128 po wybranej paczce | 85,93709 |
| Oczekiwane MSE128 przy losowaniu jednej z32 paczek | 85,96346 |
| Początkowe confidence / IoU | 86,79% / 85,66% |
| Confidence / IoU po wybranej paczce | 86,11% / 84,37% |

**Dobra prognoza ruchu nie oznacza istnienia paczki pchającej ku celowi.** W tej próbie ranking wybierał najmniejsze oddalenie, a nie przybliżenie. Około98,7% jest redukcją błędu przyrostu względem prognozy „bez ruchu”, nie prawdopodobieństwem trafienia kierunku, wzrostem IoU ani skutecznością dojścia do sztucznego celu.

## Dalszy skan uruchomiony

Podfolder `skan_puli35000` obejmuje34701 rekordów po wykluczeniu299 klatek filmu sondy. 4338 ustalonych paczek po maksymalnie8. Skan nie przeszukuje wszystkich możliwych składów paczek. Na końcu powstanie ranking i lista1500 unikalnych kandydatów; taka lista nie jest jeszcze zweryfikowaną kolejnością wielokrokowego treningu. Jeśli wynik nie ma paczek z dodatnim zyskiem, nie wolno opisywać najmniej szkodliwych kandydatów jako prowadzących do celu.

Prognoza połączona uwzględnia gradient, pamięć AdamW i decay, ale zamraża bufory BatchNorm. Rzeczywisty krok aktualizuje także te bufory. Cel pochodzi z jednego obrazu, nie z nowej analizy1500 obrazów. Wyniku nie należy uogólniać na całą populację celów ani na kolejne kroki bez ponownego oszacowania.
