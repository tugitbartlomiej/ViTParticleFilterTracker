# Koopman i Langevin: te same 128 oryginalnych kanałów

Obliczenia zakończono 2026-10-02T10:51:29.639583+02:00. Bez PCA, rotacji i ponownego wyboru kanałów.

Policzono samodzielne prognozy dwóch liniowych przybliżeń Koopmana i odtworzono prognozy zamrożonego Langevina. Każda cząsteczka ma dokładnie te same 128 wybranych współrzędnych jako stan i cel. Źródłowy wynik 92.83 procent dotyczy rozpoznania obecności narzędzia w parach oryginał/LaMa na k500, a nie prognozy ruchu. Pełny latent ma nadal 768 współrzędnych.

## Równania i znaczenie porównania

Stan x_i(t) = z_i,J(t) − z_i,J(500), J zapisano w WYBOR_PRZED_TESTEM.json. Wagi cząsteczek odpowiadają pokryciu maską narzędzia.

```math
\widehat{x}_{i,t+\Delta t}=Kx_{i,t}+b \qquad\text{(Koopman)}
\widehat{x}_{i,t+\Delta t}=x_{i,t}+b+Ax_{i,t}+K_0\Delta x_{i,t}+K_1\Delta x_{i,t-\Delta t} \qquad\text{(Langevin)}
```

K, A oraz obie macierze pamięci są pełnymi macierzami 128×128. Współczynniki są wspólne dla cząsteczek. Stała w słowniku Koopmana obsługuje wyraz wolny; macierz rozszerzona ma 129×129, ale nie dodaje nowego kanału obrazu. To przybliżenie operatora Koopmana na funkcjach (1,x), bez dowodu dokładnego domknięcia dynamiki w tej przestrzeni.

Koopman przyrostowy ma K = I + A. Jest to wcześniejszy wariant bez jawnej pamięci L0, z regularyzacją ku brakowi ruchu. Niezależny fit i osobny rollout odtwarzają jego macierze i wyniki. Zmiana zapisu na K nie stanowi nowej metody ani poprawy.

Koopman bezpośredni dopasowuje następny stan z regularyzacją ku K = 0. Ma tę samą klasę funkcji affine co wariant przyrostowy, ale inną karę. Wybrano ridge 0.01 dla dt1 oraz 1 dla dt10, wyłącznie na walidacji ze stałej siatki sześciu wartości.

Langevin używa dwóch przyrostów historii i wcześniej wybranej regularyzacji 1 dla dt1 oraz 10 dla dt10. Zachowano jego macierze, ale przeliczono prognozy osobną implementacją. W prognozie punktowej nie podajemy przyszłych realizacji szumu. Poprzednia diagnostyka reszt nie potwierdziła białości, więc nie traktujemy modelu jako dokładnie domkniętego równania stochastycznego.

## Wyniki na obrazie 2

| Model, zapis co jedną aktualizację | Znak H1 | Redukcja MSE H1 | Redukcja MSE H10 | Redukcja MSE H50 |
|---|---:|---:|---:|---:|
|Langevin z pamięcią|71.68%|47.75%|-4.15%|-45.30%|
|Koopman przyrostowy|50.65%|-0.43%|1.28%|-14.34%|
|Koopman bezpośredni|50.63%|-193.68%|-105.98%|-58.97%|
|Stała prędkość|78.67%|55.92%|-289.63%|-4677.47%|
|Brak zmiany|0.00%|0.00%|0.00%|0.00%|

Redukcja MSE = 100 × (1 − MSE_model / MSE_brak_zmiany). Wynik ujemny oznacza gorszą prognozę niż brak ruchu, a nie procent błędnie rozpoznanych obrazów. Trafność znaku jest średnią po wszystkich współrzędnych z wagami maski. Nie odsiewamy małych zmian na tych wykresach. Brak zmiany nie przewiduje wzrostu ani spadku, stąd zerowa lub bliska zeru trafność znaku w tej definicji.

| Model, zapis co dziesięć aktualizacji | Znak H10 | Redukcja MSE H10 | Redukcja MSE H20 | Redukcja MSE H50 |
|---|---:|---:|---:|---:|
|Langevin z pamięcią|56.67%|4.30%|9.40%|4.15%|
|Koopman przyrostowy|57.15%|7.38%|8.59%|-0.61%|
|Koopman bezpośredni|58.98%|-65.27%|-25.47%|-32.33%|
|Stała prędkość|42.34%|-149.31%|-373.07%|-1623.95%|
|Brak zmiany|0.00%|0.00%|0.00%|0.00%|

## Drugi obraz: transfer bez refitu

| dt | Model | Redukcja MSE H1 | Redukcja MSE H10 | Redukcja MSE H50 |
|---:|---|---:|---:|---:|
|1|Langevin z pamięcią|-1.41%|-262.36%|-265.27%|
|1|Koopman przyrostowy|-8.50%|-18.19%|-162.50%|
|1|Koopman bezpośredni|-1267.51%|-643.85%|-272.66%|
|1|Stała prędkość|62.53%|-257.20%|-4543.32%|
|10|Langevin z pamięcią|n/d|-23.01%|-70.63%|
|10|Koopman przyrostowy|n/d|-33.04%|-123.96%|
|10|Koopman bezpośredni|n/d|-430.09%|-146.47%|
|10|Stała prędkość|n/d|-148.29%|-1759.55%|

## Ocena poza dopasowaniem i ograniczenia

Fit: aktualizacje do 700; walidacja do 799. Końcowy fit operatorów sięga 799 dla dt1 albo 790 dla dt10. Test: starty 800–950, horyzont do 50 aktualizacji. Obraz 2 ma 31 cząsteczek narzędzia, obraz 8 ma 7. Nie dopasowywano operatorów na drugim obrazie. Wszystkie dane pochodzą z jednego biegu sieci. Cząsteczki i nakładające się okna nie są niezależnymi powtórzeniami.

Test był już oglądany w poprzednich analizach. W tym rozszerzeniu przestrzegano zamrożonej siatki i wyboru na walidacji, ale wynik pozostaje eksploracyjny. Nie zmieniano kanałów, siatki ani wyboru po nowych wynikach. MODELE.json podaje promienie spektralne Koopmana; nie narzucono stabilności po fakcie. Wyniki co jedną i co dziesięć aktualizacji mają różne gęstości startów; METRYKI.json zawiera również ocenę dt1 na wspólnych startach co10.

Modele mają ten sam fizyczny cel, skalowanie wyłącznie na fit i ten sam dostęp do zakończonej historii. Różni się wykorzystana długość pamięci. Langevin ma 49280 parametrów, każdy Koopman 16512; porównanie jest poza dopasowaniem, nie na błędzie treningowym. Nie używamy przyszłych klatek, przyszłych minibatchy ani gradientów. Nie ma telemetrii sterowania tego biegu, więc w tych modelach nie dodano jawnego członu obrazu/batcha.

## Kontrole i artefakty

Wszystkie 36 kontrole numeryczne zapisane w AUDYT.json przeszły. Obejmują odtworzenie źródłowego L0 przez osobne dopasowanie Koopmana przyrostowego, zgodność nowej implementacji prognozy Langevina z zapisanymi prognozami, testy prefix/poison przyszłych stanów i cały wymiar128. Dodatkowo kod sprawdza zgodność wyników MSE i znaków z zamrożonymi ocenami oraz hashe źródeł. To kontrola numeryczna kodu, nie nowy niezależny audyt Astry.

MACIERZE.npz: pełne K i b obu operatorów Koopmana oraz A, K0, K1 i b Langevina. WYBOR_PRZED_TESTEM.json: kanały J, siatka, wybór i hashe zapisane przed oceną. PROGNOZY.npz: pełne prognozy128D. METRYKI.json: wszystkie metryki i opisowe przedziały blokowe. POROWNANIE.csv: płaska tabela do dalszego porównania.

![Trafność znaku](ZNAK_OBRAZ_2.png)

![Błąd](BLAD_OBRAZ_2.png)

![Transfer znak](ZNAK_OBRAZ_8.png)

![Transfer błąd](BLAD_OBRAZ_8.png)

## Odtworzenie

```powershell
& 'F:\Python\python.exe' porownanie.py
& 'F:\Python\python.exe' raport.py
```

Skrypt porównania blokuje nadpisanie istniejących metryk, aby nie powtarzać strojenia na znanym teście. Raport można regenerować bez fitu. Źródła i wcześniejsze eksperymenty pozostają niezmienione.
