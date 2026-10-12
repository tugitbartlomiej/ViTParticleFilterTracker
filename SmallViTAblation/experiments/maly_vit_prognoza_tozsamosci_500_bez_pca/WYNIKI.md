# Prognozowanie 128 natywnych kanałów tożsamości do 500 aktualizacji

Data: 2026-10-02. Status: obliczono; wynik eksploracyjny z jednego biegu sieci.

Przeszukano **99 konfiguracji**. Wybrany przed testem model: **Powrót do historycznej średniej (okno 200, czas 200)**, `reversion_w200_t200`. Bramkę użyteczności na walidacji spełniono.

## Główny wynik: model wybrany przed testem

| Horyzont | Znak: dopasowanie | Znak: transfer | MSEskill: transfer | Norma prognozy / prawdziwa: transfer |
|---:|---:|---:|---:|---:|
| 1 | 50.00% | 50.56% | 0.12% | 0.025 |
| 2 | 52.11% | 54.05% | 0.62% | 0.025 |
| 5 | 59.32% | 62.99% | 2.95% | 0.034 |
| 10 | 65.14% | 67.06% | 5.11% | 0.054 |
| 20 | 64.31% | 66.01% | 8.94% | 0.091 |
| 50 | 62.44% | 64.62% | 13.19% | 0.184 |
| 100 | 64.24% | 62.36% | 18.13% | 0.323 |
| 200 | 61.67% | 63.08% | 17.26% | 0.435 |
| 300 | 57.82% | 57.37% | -0.81% | 0.529 |
| 400 | 57.42% | 57.77% | -0.15% | 0.566 |
| 500 | 57.11% | 56.52% | -7.52% | 0.524 |

Próg ponad 85 procent dotyczy znaku skumulowanego przemieszczenia każdego kanału. Nie jest to trafność rozpoznawania narzędzia ani procent dokładnie odtworzonych wektorów. Dodatni MSEskill oznacza mniejszy błąd niż założenie braku ruchu; ujemny oznacza pogorszenie.

- fit: ciągły zakres ≥85% od H1 kończy się na **0**; najdalsze pojedyncze przekroczenie: **brak**.
- transfer: ciągły zakres ≥85% od H1 kończy się na **0**; najdalsze pojedyncze przekroczenie: **brak**.

Osobno przed testem wybrano model według najmniejszego błędu: **Powrót do historycznej średniej (okno 50, czas 50)** (`reversion_w50_t50`). Na H500 w transferze uzyskał **62.19%** znaku i **9.82%** redukcji MSE; norma prognozowanego ruchu wynosiła 0.464 normy rzeczywistej. Nie zastępujemy nim po teście głównego zwycięzcy ustalonego kryterium.

## Najlepszy wariant każdej rodziny, wybrany wyłącznie na walidacji

Tabela przedstawia końcowy transfer na cztery obrazy. Jej kolejność nie jest rankingiem na teście.

| Rodzina | Znak H1 | Znak H50 | Znak H500 | MSEskill H500 | Norma H500 | Konfiguracja |
|---|---:|---:|---:|---:|---:|---|
| Obserwator narzędzie + tło | 57.45% | 38.76% | 40.92% | -1317.45% | 3.315 | `coupled_observer_bg100_res10` |
| Wygaszana prędkość | 75.73% | 43.98% | 46.03% | -149.78% | 1.047 | `damped_velocity_t10` |
| Langevin: wykładnicza pamięć co 1 | 58.77% | 58.31% | 56.04% | -117.70% | 1.318 | `linear_09__dt1` |
| Langevin: wykładnicza pamięć co 10 | 46.26% | 57.97% | 57.46% | -148.23% | 1.446 | `linear_11__dt10` |
| Langevin: skończona pamięć co 1 | 70.44% | 54.30% | 54.67% | -174.92% | 1.469 | `linear_07__dt1` |
| Langevin: skończona pamięć co 10 | 45.76% | 56.13% | 56.71% | -65.42% | 1.080 | `linear_04__dt10` |
| Obserwator Kalmana | 69.28% | 40.84% | 43.28% | -279377.46% | 51.358 | `kalman_q1.0_tNone` |
| Koopman co 1 | 43.86% | 49.85% | 54.34% | -424.16% | 2.183 | `linear_00__dt1` |
| Koopman co 10 | 43.90% | 53.47% | 56.02% | -170.40% | 1.497 | `linear_00__dt10` |
| Trend OLS | 46.91% | 43.80% | 50.39% | -7.98% | 0.303 | `ols_w128_t50` |
| Brak ruchu | 0.00% | 0.00% | 0.00% | 0.00% | 0.000 | `persistence` |
| Powrót do historycznej średniej (okno 200, czas 200) | 50.56% | 64.62% | 56.52% | -7.52% | 0.524 | `reversion_w200_t200` |
| Koopman: przycięte widmo co 1 | 43.77% | 49.82% | 54.62% | -414.66% | 2.165 | `linear_27__dt1` |
| Koopman: przycięte widmo co 10 | 43.83% | 53.48% | 56.64% | -159.62% | 1.465 | `linear_26__dt10` |
| Ostatnia prędkość | 75.73% | 43.98% | 46.03% | -323048.58% | 55.045 | `velocity` |

## Jak policzono

- Stan: 128 ustalonych oryginalnych kanałów, bez PCA, rotacji, autoenkodera i wybierania łatwych współrzędnych.
- Dane: 1001 stanów od 0 do 1000, 218 cząsteczek narzędzia z 12 stałych obrazów. Każdy obraz ma równą wagę; wewnątrz obrazu użyto pokrycia maską.
- Walidacja: modele dopasowane na obrazach 0–5 do 250 lub 350; obrazy 6–7 oceniane od 300 lub 400. Wszystkie cele wyboru kończą się do 500.
- Refit: obrazy 0–7 do 500. Test: start 500, pełny autonomiczny przebieg do 1000. Transfer: obrazy 8–11 nie używane do dopasowania równań.
- Wszystkie obrazy pochodzą ze zbioru treningowego detektora. Transfer dotyczy modelu dynamiki, nie nowego testu rozpoznawania narzędzia.
- Modele co 10 mają 50 iteracji doH500; pośrednie stany interpolowano wyłącznie z prognoz. Ich H1 nie jest niezależną identyfikacją dynamiki jednego kroku.
- Kalman i obserwatory otrzymują rzeczywiste dane wyłącznie do startu. Dalej nie ma pomiarów korygujących ani przyszłych obrazów czy gradientów.
- MSEskill raportujemy jako średnią wyników per obraz. Nie jest to iloraz MSE po połączeniu wszystkich tokenów.
- Wybrane modele są średnimi predyktorami. Model GLE z pamięcią nie stanowi tu dowodu dokładnego Mori–Zwanziga ani białego szumu reszt.

## Co ogranicza wniosek

Kanały ustalono na 500, więc wcześniejsze foldy są retrospektywną walidacją w tej bazie. Dla H500 po ustaleniu kanałów istnieje tylko jeden moment startu. Obrazy i tokeny obserwują ten sam trening, więc nie traktujemy ich jako niezależnych biegów i nie podajemy pozornego przedziału ufności po tysiącach współrzędnych. Ta historia była już wcześniej analizowana; potwierdzenie wyniku wymagałoby niezależnego biegu lub dalszego odcinka.

Sam wysoki znak nie potwierdza wielkości ruchu. Sprawdzać trzeba jednocześnie MSEskill, normę i przewagę nad prostym trendem. Norma bliska 0 oznacza niemal brak prognozowanego ruchu; bardzo duża norma ujawnia rozbieżność mimo trafienia części znaków.

## Pliki

- `PLAN.md`: siatka 99 modeli, foldy i reguła wyboru przed fitami.
- `WYBOR_PRZED_TESTEM.json`: cała walidacja i zamrożony wybór.
- `OCENA_H500.json` i `POROWNANIE.csv`: metryki każdego H1–500.
- `MODELE_PRZED_TESTEM.json` i `MACIERZE_PRZED_TESTEM.npz`: odtwarzalne parametry.
- `PROGNOZY_*.npz`: pełne trajektorie wybranych wariantów.
- `CO_OPISUJE_MODEL.md`: równania i interpretacja.
- `LITERATURA_I_AUDYT.md`: źródła pierwotne i niezależny audyt.

![Trafność transferu](ZNAK_TRANSFER.png)

![Błąd i amplituda](BLAD_I_AMPLITUDA.png)

![Przykładowe trajektorie](TRAJEKTORIE.png)

## Dodatkowa diagnostyka znaków

Poniższa kontrola nie zmienia wyboru modelu. Znak większości oszacowano dla każdej cząsteczki i kanału wyłącznie z historycznych zmian danego H kończących się do500. Remis daje znak0. Nie jest to prognoza wielkości; przy H500 dostępna jest tylko jedna historyczna para.

| H | Dodatnie zmiany w transferze | Ujemne zmiany w transferze | Trafność historycznego znaku większości |
|---:|---:|---:|---:|
| 1 | 52.02% | 47.98% | 50.86% |
| 2 | 52.26% | 47.74% | 50.67% |
| 5 | 53.52% | 46.48% | 46.92% |
| 10 | 52.18% | 47.82% | 46.30% |
| 20 | 48.22% | 51.78% | 49.09% |
| 50 | 46.43% | 53.57% | 49.26% |
| 100 | 44.63% | 55.37% | 54.71% |
| 200 | 40.34% | 59.66% | 53.19% |
| 300 | 38.10% | 61.90% | 52.11% |
| 400 | 40.26% | 59.74% | 51.09% |
| 500 | 40.83% | 59.17% | 43.62% |
