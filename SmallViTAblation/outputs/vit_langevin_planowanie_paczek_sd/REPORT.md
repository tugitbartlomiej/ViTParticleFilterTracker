# Planowanie kolejności czterech paczek SD

Poprzedni eksperyment powtarzał niezależne paczki i ekstrapolował siłę początkową; nie szukał sekwencji. Tutaj po każdej rzeczywistej aktualizacji ponownie liczymy siły czterech kandydatów i wybieramy najmniejsze przewidywane MSE J128. To greedy kontrola z horyzontem1. Bez CAV i podmiany latentu. Nie deklarujemy globalnego optimum sekwencji.

Oryginalny autentyczny checkpoint k1000/AdamW i ta sama sonda29. Prawdziwe GT tych samych czterech paczek32, naturalny odczyt sondy. Cel after jest sztuczny, odległość do niego oceniana w J128; inne640 i detektor osobno. Pęd zachowany po każdym kroku, przy przejściu100->500 również.

| Polityka | najlepszy krok | redukcja MSE128 w najlepszym stanie | MSE128 najlepsze | redukcja końcowa | IoU w najlepszym stanie |
|---|---:|---:|---:|---:|---:|
| controlled | 29 | 0.54687% | 85.451300 | 0.17072% | 0.0000 |
| random_choice | 450 | 0.93805% | 85.115194 | 0.04931% | 0.5764 |
| balanced_cycle | 424 | 0.58501% | 85.418531 | 0.46525% | 0.5963 |

Budżet: 500 aktualizacji i 16000 obrazów na politykę, razem1500 aktualizacji. Etap100 zachowany w RESULT_100.json. Liczba checkpointów to częstość zapisu, nie liczba jednostek uczenia. Brak nazw epok pełnego datasetu.

Progi redukcji MSE128:5/10/25 procent. Zgodność wartości85 procent jest osobnym progiem: clip(1-RMSE/RMS celu,0,1), a nie cosine, AP, skill prognozy lub procent redukcji MSE. null w wynikach oznacza brak osiągnięcia w skończonym budżecie; nie ekstrapolujemy czasu do celu poza budżet.

Kolejność pomiędzy paczkami zmienia pamięć AdamW i ścieżkę wag. Permutacja członków jednej paczki nie zmienia matematycznie uśrednionej symetrycznej straty ani statystyk BN bez stochastyki i przetwarzania zależnego od kolejności. Drobne różnice redukcji zmiennoprzecinkowych nie uzasadniają szukania32! permutacji.

Wybory i metryki: SCHEDULE.csv. Przed każdym wykonanym krokiem decyzja jest sealed w decisions/ oraz FORECAST_SEALS.jsonl. Rzeczywiste TRAJECTORY.npy, ACTUAL.json, siły pełnej siatki na zadeklarowanych krokach, checkpointy z pamięcią optymalizatora i naturalna weryfikacja najlepszego checkpointu są zachowane.

Koszt obliczeń: {"candidate_gradient_batches": 3004, "candidate_probe_forwards": 6008, "training_updates": 1500, "training_image_exposures": 48000}; 1503.1 sekund. Wybór kontrolowany wymaga4 gradientów kandydatów na krok, losowy/cykl1. Każda polityka ma ten sam budżet realnych aktualizacji, nie ten sam koszt prognoz. Ścisły nochange ma0 aktualizacji. Żadnych przyszłych gradientów w zapowiedzi decyzji.

Najlepszy zaobserwowany stan to osiągnięty wynik tej polityki, nie dowód osiągalnego globalnego minimum. Głowa detektora jest uczona, dlatego kierunek latentu i IoU/pewność nie są utożsamiane.

## Końcowy wniosek z porównania500 kroków

W tym biegu kontroler horizon1 nie wygrał z losowaniem ani równym cyklem pod względem najlepszego osiągniętego MSE J128. Mimo mediany skill prognozy99.84 procent kontrolera jego najlepsza redukcja MSE wyniosła0.54687 procent. To skill przewidywania przyrostu, nie zgodność z celem. Losowanie osiągnęło0.93805 procent, cykl0.58501 procent. Wszystkie progi5/10/25 procent redukcji MSE oraz85 procent jawnie zdefiniowanej zgodności wartości pozostały nieosiągnięte do500. Nie ma na tej podstawie wiarygodnej liczby dalszych kroków do celu.

Najlepszy prefiks kontrolera:2026100511 powtórzone15 razy, następnie2026100549 powtórzone12 razy, następnie2026100511 powtórzone2 razy.29 aktualizacji,928 obrazów treningowych; checkpoint k1029. Najlepsza losowa sekwencja to prefix450 zapisany w random_choice/BEST_SEQUENCE.json oraz SCHEDULE.csv, seed20261006. Cykl osiągnął minimum w424:106 pełnych obiegów kolejności11,23,37,49. To opis rzeczywiście wykonanych sekwencji, nie globalnego optimum.

| Polityka | MSE J128 w najlepszym stanie | MSE other640 | MSE pełne768 | zgodność wartości J128 | pewność sondy | IoU sondy |
|---|---:|---:|---:|---:|---:|---:|
| controlled | 85.451300 | 0.075487 | 14.304790 | 0.60849% | 0.06223 | 0.00000 |
| random_choice | 85.115194 | 0.080694 | 14.253111 | 0.80415% | 0.60921 | 0.57636 |
| balanced_cycle | 85.418531 | 0.057006 | 14.283927 | 0.62755% | 0.29210 | 0.59628 |

Przed uczeniem MSE J128=85.921177, MSE other640=0, MSE768=14.320196, confidence=0.86787, IoU=0.85659. Wszystkie najlepsze stany pogorszyły IoU sondy. Kontroler w swoim minimum J128 miał IoU0. Celem kontrolera był tylko J128; dlatego wyniku nie utożsamiamy z poprawą detekcji lub zachowaniem innych640.

Zweryfikowano wszystkie1503 naturalne stany trzech trajektorii,1500 immutable sealed decyzji, wybory argmin kontrolera, trzy optymalizatory końcowe1500 i prefiksy BEST_SEQUENCE. Naturalny odczyt z każdego najlepszego checkpointu zgadza się dokładnie z zapisanym latentem: maksymalny błąd0. FINAL_AUDIT i BEST_RELOAD_VERIFICATION zawierają dowody aparatu.500 kroków każdej polityki to16000 ekspozycji zdjęć,48000 łącznie; checkpoint sampling nie jest jednostką treningu.
