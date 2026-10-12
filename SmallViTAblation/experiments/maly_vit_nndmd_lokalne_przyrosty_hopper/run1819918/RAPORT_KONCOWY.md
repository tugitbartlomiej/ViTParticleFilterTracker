# Końcowy wynik NNDMD: lokalna historia i przyrosty

Eksperyment zakończył się technicznie poprawnie, lecz nie wykazał korzyści prognostycznej. Model przegrał z prognozą utrzymującą latent checkpointu 90 bez zmian we wszystkich dziesięciu horyzontach, zarówno dla pełnych 768, jak i wybranych 128 kanałów.

Slurm 1819918: COMPLETED, exit 0. Hopper-2: cztery pełne H200 i siedem partycji MIG. Czas zadania: 1 godzina 38 minut 22 sekundy. Zakończenie: 2 października 2026, około 22:43 czasu polskiego. 7379 aktualizacji uczenia; dwie aktualizacje kontroli wykonania zapisano osobno.

## Wynik zbiorczy

| Kanały | RMSE modelu | RMSE bez zmian | Wzrost błędu RMSE |
|---|---:|---:|---:|
| full768 | 0.159935 | 0.139319 | 14.80% |
| J128 | 0.141612 | 0.122625 | 15.48% |

RMSE zbiorcze jest pierwiastkiem ze średniego MSE, przy równych wagach obrazów, tokenów, kanałów i horyzontów. Nie jest średnią dziesięciu wartości RMSE.

## full768: wszystkie horyzonty testu

| Przejścia checkpointów od 90 | RMSE modelu | Bez zmian | Średnia historyczna | Kontynuacja ostatniej zmiany |
|---:|---:|---:|---:|---:|
| 1 | 0.111470 | 0.106950 | 0.111877 | 0.156087 |
| 2 | 0.132499 | 0.120049 | 0.113371 | 0.256637 |
| 3 | 0.163653 | 0.148533 | 0.147367 | 0.354599 |
| 4 | 0.142445 | 0.126234 | 0.119730 | 0.437980 |
| 5 | 0.160967 | 0.147487 | 0.148090 | 0.530901 |
| 6 | 0.151594 | 0.137737 | 0.132689 | 0.624561 |
| 7 | 0.171995 | 0.144279 | 0.140487 | 0.716562 |
| 8 | 0.179516 | 0.145841 | 0.137045 | 0.815808 |
| 9 | 0.176809 | 0.143833 | 0.135916 | 0.902809 |
| 10 | 0.191773 | 0.163563 | 0.157206 | 1.006080 |

## J128: wszystkie horyzonty testu

| Przejścia checkpointów od 90 | RMSE modelu | Bez zmian | Średnia historyczna | Kontynuacja ostatniej zmiany |
|---:|---:|---:|---:|---:|
| 1 | 0.099285 | 0.092504 | 0.098181 | 0.138529 |
| 2 | 0.121620 | 0.109186 | 0.101601 | 0.237096 |
| 3 | 0.139250 | 0.122809 | 0.119589 | 0.322773 |
| 4 | 0.127553 | 0.113493 | 0.107741 | 0.403442 |
| 5 | 0.143963 | 0.127325 | 0.126773 | 0.487800 |
| 6 | 0.134656 | 0.123329 | 0.116211 | 0.580504 |
| 7 | 0.147622 | 0.123034 | 0.115634 | 0.666323 |
| 8 | 0.157229 | 0.133748 | 0.121765 | 0.762839 |
| 9 | 0.159112 | 0.130338 | 0.119975 | 0.841369 |
| 10 | 0.171881 | 0.143148 | 0.133738 | 0.937122 |

## Protokół i granice wniosku

Dokładnie 1500 źródłowych obrazów ViT, wszystkie 600 tokenów każdego obrazu, globalny batch 32 i ostatni batch 28. Jedno przejście checkpointów odpowiada dziesięciu aktualizacjom źródłowego ViT. Test startuje od checkpointu 90 i autonomicznie przewiduje 91–100; prawdziwe stany przyszłe służą jedynie ocenie.

Dwa warianty learning rate macierzy: 0.0001 i 0.001, każdy zatrzymany po trzech epokach. Wybór wyłącznie na walidacji 80–90: learning rate 0.0001, jedna epoka. Następnie świeże końcowe dopasowanie na historii 50–90 i zamrożony test. Stan wewnętrzny ma 768 wymiarów, decoder jest nieliniowy, a prognoza przyrostów zachowuje stały offset stanu początkowego.

Test sprawdza przyszłość tych samych obrazów. Historia 90–100 była już oglądana we wcześniejszych eksperymentach, więc wynik jest eksploracyjny, a nie globalnie nietknięty test niezależny. Wynik dotyczy tego wariantu NNDMD i tej trajektorii uczenia; nie dowodzi nieskuteczności wszystkich metod Koopmana.

Nie uruchomiono kolejnego treningu ani nie dobrano nowych parametrów po teście. Monitor pozostaje wyłączony.

Źródła liczb: METRICS_TEST.json, SELECTION.json, status.json oraz stan Slurm. Oryginalny raport generatora zachowano jako RAPORT.md.
