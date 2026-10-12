# Wyniki: osobna macierz dla każdego wektora każdego obrazu

Policzono 41472 niezależne modele: 72 obrazy × 576 wektorów po 384 współrzędne. Bez PCA i bez łączenia obrazów w regresji.
Macierze końcowe korzystają z 72 par każdej własnej historii. Ocena dotyczy osobnych macierzy dopasowanych tylko do pierwszych 48 par.
Pojedynczy krok to 5 aktualizacji RF-DETR. Końcowa przyszłość macierzy dopasowanych do wszystkich checkpointów nie jest zweryfikowana.

| Metoda | RMSE jednego checkpointu | RMSE 100 aktualizacji |
|---|---:|---:|
| koopman | 0.01033074 | 0.23218073 |
| last_increment | 0.00385891 | 0.11696864 |
| persistence | 0.00822556 | 0.13622288 |
| scalar_velocity | 0.00371795 | 0.07107641 |

Kandydaci spełniający ustalone kryteria: 0/41472; na maskach >=0,5: 0/1458.
Błąd dopasowania pierwszych 48 par: 0.00444989; wszystkich 72 par: 0.00448472.
Te liczby są eksploracyjne. Nie wybierają automatycznie poprawnej tożsamości narzędzia ani nie stanowią korekty na wielokrotne testowanie.
Błędy każdej jednostki: wyniki/PER_VECTOR.csv. Błędy każdego odcinka: MSE_ONE_STEP.npy (obraz,token,24 odcinki,4 metody).
Macierze: C_FINAL.npy, MEAN_X_FINAL.npy, MEAN_DELTA_FINAL.npy, ABSOLUTE_RIDGE_FINAL.npy i jawna formuła w PLAN.md. Xc odtwarzane ze zweryfikowanych źródeł.
Eksport jawnej macierzy: eksport.py --image 66 --token 299. Modele testowe odtwarza się z pierwszych 48 par i RELATIVE_RIDGE.npy.