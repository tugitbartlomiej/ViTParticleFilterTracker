# Kontrola obliczenia i macierz z czterech epok

2026-09-24. Analiza diagnostyczna po wyniku, bez treningu i zmiany bramki.

Ponowne obliczenie odtwarza oryginalną macierz z maksymalną różnicą 2,42e-15.
Niezależny solver najmniejszych kwadratów z dołączoną karą odtwarza współczynniki
z różnicą 1,67e-15. Względna reszta równań normalnych wynosi 1,93e-15.
Nie znaleziono błędu rozwiązywania zadania regresji ani konwencji transpozycji.
Nie dowodzi to, że wybrane zmienne i model są wystarczające dla dynamiki uczenia.

Zapisano osobne macierze dla każdej epoki oraz całej historii, w tej samej bazie.
Macierz z całych czterech epok i prognoza do kroku 500 znajdują się w
`wyniki/local_1810448/run/round_01/OPERATOR_FULL_4_EPOCHS.json`.
Brak piątej epoki oznacza brak obserwacji do sprawdzenia tej nowej prognozy.
Model dopasowany na czterech epokach nie może być uczciwie oceniany jako
prognoza odłożonego końca tych samych czterech epok.

Porównanie na tych samych odłożonych obrazach i początkach prognozy:
oryginalny model 0,218159; rosnąca historia 0,212929; ostatnie pięć zapisów
0,182088; ostatnie dziesięć 0,213512; regresja przyrostów 0,206646.
Brak zmiany nadal lepszy: 0,137424. Niższa norma błędu jest lepsza.
Okna dobrano dopiero po wyniku, więc te liczby służą diagnozie, nie wyborowi
zwycięskiego modelu z potwierdzoną generalizacją.

Wniosek: wcześniejsze stwierdzenie, że zawiodła stałość macierzy, było zbyt
kategoryczne. Spowolnienie i słaba zgodność kierunków są zmierzone; przyczyna
nie została jednoznacznie oddzielona od niewystarczającej reprezentacji,
ograniczonej historii i sposobu dopasowania. Skrócenie okna samo nie wystarczyło.

Uruchomienie: `py -3.11 -B audyt_macierzy.py --run wyniki/local_1810448/run`.
Szczegóły: `wyniki/local_1810448/run/round_01/AUDYT_MACIERZY.json`.
