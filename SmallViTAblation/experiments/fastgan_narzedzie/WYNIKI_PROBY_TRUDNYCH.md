# Wynik próby trudnych obrazów, 2026-09-23

**Wynik techniczny dodatni:** zamrożony FastGAN umożliwił wyszukiwanie otoczenia, w którym detektor gubi prawdziwe, niezmienione narzędzie. **Przewaga Koopmana ani poprawa uczenia nie są wykazane.** Próba trwała 88 sekund na lokalnej karcie. Żadnych wag nie douczano.

Plan: `PLAN_PROBY_TRUDNYCH.md`. Kod: `proba_trudnych.py`. Wyniki źródłowe: `wyniki/trudne_pilot_20260923/REPORT.json`, `RESULTS.json`, `CONFIG.json`, `OPERATOR.json`. Dla każdej operacji zapisano obrazy, prawdziwą maskę, wejścia generatora, jego surowe wyjścia, przebieg optymalizacji i profile poszczególnych tokenów. Widoczne obrazy są kompozycjami prawdziwych wycinków z wygładzonym tłem generatora, nie samodzielnie wygenerowanymi klatkami filmu.

## Porównanie

Jeden wycinek z każdej z 12 operacji, odrębnych od 32 operacji dopasowania operatora i 16 jego kontroli. Generator był wcześniej uczony na tych operacjach; nie są one odłożonym testem generatora. Detektor jest wstępnie uczony na 500 klatkach.

| Wariant | Trafienia z 12 | Nowe chybienia wśród 8 pierwotnych trafień | Średnie maksimum pewność × IoU |
|---|---:|---:|---:|
| Oryginał | 8 | 0 | 0,478 |
| Jedno losowe wejście FastGAN | 8 | 1 | 0,491 |
| Najtrudniejsze z 20 losowań | 3 | 5 | 0,368 |
| Sterowanie samą detekcją | 4 | 4 | 0,325 |
| Sterowanie detekcją i Koopmanem | 2 | 6 | 0,335 |

Trafienie wymaga jednocześnie pewności ≥0,5 i IoU ≥0,5. Spadek wyniku oznacza trudniejszy przykład, nie lepszy detektor. Jedno losowe wejście psuje jedno pierwotne trafienie i poprawia jedno pierwotne chybienie, dlatego łączna liczba trafień pozostaje taka sama.

Dodanie Koopmana daje dwa nowe chybienia więcej niż sterowanie samą detekcją, lecz tylko jedno więcej niż wyszukiwanie losowe. Średnia ciągła miara trudności wypada **nieco lepiej dla sterowania samą detekcją**. Przy tak małej próbie, jednym losowaniu startów i zależności wyniku od progu nie ogłaszamy przewagi metody. Budżet porównania oznacza liczbę kandydatów/kroków, nie identyczny koszt GPU: optymalizacja wymaga dodatkowo gradientów.

## Co wniósł operator

To liniowe afiniczne EDMD dla rozjaśnienia otoczenia, nie deep Koopman i nie operator czasu filmu. Wybór obejmuje sześć nietrywialnych modów bliskich +1. Bliskość wartości własnej nie jest dowodem tożsamości narzędzia.

- Błąd na innych operacjach: 0,2555 dla operatora, 0,2519 dla przewidywania „nic się nie zmienia”, 1,1788 dla dopasowania po potasowaniu par.
- Operator nie pokonuje prostej niezmienności.
- Odczyt kierunku RDZEN z tych modów na innych operacjach daje R²=0,029. To słaby odczyt i **nie uprawnia do nazywania tych modów kodem tożsamości**.
- Wykorzystano rozrzut reszt jako skalę porównania. Nie estymowano jądra pamięci Langevina ani niepewności Kalmana.

## Kontrole i ograniczenia

Wagi i bufory obu modeli mają identyczne skróty przed i po próbie. Powtórzony odczyt tokenów daje zerową różnicę. Preprocessing i indeks klasy sprawdzono względem publicznego `predict`: maksymalna pewność jest identyczna. Kanał narzędzia to indeks 0 po przemapowaniu etykiet COCO, a nie surowe category_id=1. Trzy testy CPU sprawdzają zachowanie narzędzia i gradientu, geometrię ramek oraz operator z przesunięciem afinicznym.

Kontrola wklejenia na własne tło odtwarza oryginał dokładnie. Piksele wewnątrz prawdziwej maski są identyczne także po zapisaniu PNG (`KONTROLA_PNG.json`). To nie dowodzi realizmu zmienionego tła: widać rozmycie, przenikanie struktur i miejscami obwódki przy instrumencie. Tych obrazów nie włączono do treningu.

Katalogi `trudne_smoke_20260923` i `trudne_smoke_v2_20260923` są nieważnymi próbami technicznymi: pierwsza ujawniła konflikt bufora RF-DETR z gradientami, druga błędny indeks klasy. Zachowano je dla audytu. Poprawny test techniczny to `trudne_smoke_v3_20260923`; wnioskowanie dotyczy wyłącznie pełnego pilota.

## Wniosek do dalszej pracy

Da się sterować otoczeniem, żeby ujawnić słabość detektora przy zachowanym narzędziu. Nie wiemy jeszcze, czy takie sytuacje odpowiadają naturalnym klatkom i czy ich dodanie poprawi uczenie. Najpierw trzeba odrzucić przykłady zależne od artefaktów i potwierdzić pomyłki na prawdziwych pełnych klatkach. Samej reszty Koopmana nie uznajemy na tej podstawie za sygnał wartości próbki uczącej.
