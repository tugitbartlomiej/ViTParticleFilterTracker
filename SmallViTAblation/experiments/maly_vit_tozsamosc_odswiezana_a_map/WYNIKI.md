# Wynik: rozdzielenie narzędzia od tła jest pełne od 50. kroku uczenia i nie rośnie; kierunek stały się starzeje

Data: 2026-10-05. Plan: [PLAN.md](PLAN.md). Bez treningu, bez GPU, bez Edenu. Dane: latent 192 sond co 10 kroków z
kampanii pętli v2 (`../maly_vit_wektor_tozsamosci/wyniki_eden/petla_s47..s56`, Eden 1817949): 10 ziaren, rozgrzewka
1000 kroków i trzy ramiona do kroku 4500. Liczby: `wyniki/WYNIK.json` (kryteria), `wyniki/UZUPELNIENIE.json` (opis
dodany po wyniku), surowe miary `wyniki/POMIAR.npz`.

Słowa: **kierunek stały** to wektor tożsamości dopasowany raz w kroku 1000 (jak w pętli). **Kierunek odświeżany** to
ten sam przepis zastosowany od nowa w danej chwili. **Rozdzielenie** `d'` to odległość średnich rzutów narzędzia i
negatywów w jednostkach ich odchylenia; odświeżane liczone jest na wycinkach nieużytych do dopasowania.
**Siła stała** to średni rzut latentu narzędzia na kierunek stały (wielkość prognozowana w pętlach v3 i 4.3).

Kontrola poprawności: siła stała wychodzi 6,13 w kroku 1000 i 5,30 / 4,36 / 4,37 w kroku 4500 (bez / losowe /
koopman), czyli dokładnie liczby zapisane w `../maly_vit_wektor_tozsamosci/PLAN.md`.

## Werdykt

1. **Rozdzielenie odświeżane nie rośnie w uczeniu (K1 niespełnione).** Jest najwyższe na samym początku: 10,7 w
   kroku 50, gdy mAP wynosi 0,000, potem 8,5 w kroku 300, 7,3 w kroku 1000 i 6,8-7,3 w kroku 4500. Pole pod krzywą
   ROC wynosi 1,00 w każdej chwili od kroku 50. Ta definicja tożsamości (narzędzie przeciw tłu i scenie) jest w
   pełni odczytywalna z sieci prawie nieuczonej i nie mierzy postępu detektora.
2. **Między ramionami związek z mAP jest słaby i dodatni (K2 w jednym z dwóch ramion, K3 na granicy).** Ramię
   `koopman` ma wyższe rozdzielenie odświeżane niż `bez` w 9 z 10 ziaren, ramię `losowe` w 6 z 10, choć oba mają
   wyższe mAP w 10 z 10. Korelacja wewnątrz chwili: 0,12, przedział od 0,00 do 0,25.
3. **Siła wzdłuż kierunku stałego zachowuje się odwrotnie** (jak w kampanii 4.3): ramiona z dokładaniem mają niższą
   siłę w 6-7 z 10 ziaren, korelacja wewnątrz chwili −0,14 (przedział od −0,31 do 0,07).
4. **Spadek siły stałej to starzenie się kierunku, nie utrata rozdzielenia.** Rozdzielenie wzdłuż kierunku stałego
   spada z 27 (w chwili dopasowania, na tych samych danych) do 9 już 50 kroków obok, 5 po 500 krokach i 2 po 3500;
   cosinus kierunku odświeżanego z kierunkiem z kroku 1000 wynosi 0,75 po 50 krokach, 0,45 po 500 i 0,10 po 3500.
   Rozdzielenie odświeżane stoi w tym czasie w miejscu.
5. **Z zachowania tożsamości przewidywalne jest tylko samo starzenie.** Prognoza zmiany na 500 kroków z krzywej
   pozostałych ziaren: cosinus z kierunkiem z kroku 1000: 0,53; siła stała: −0,11 (gorzej niż „bez zmian”);
   rozdzielenie odświeżane: −0,11; siła każdej sondy osobno: −0,11.
6. **Kierunek jest w każdym ziarnie inny, a podobieństwo układu sond między ziarnami maleje w uczeniu.** Cosinus
   kierunków między ziarnami: 0,01, 0,00 i −0,01 w krokach 1000, 2500 i 4500. Liniowe CKA układu 180 sond między
   ziarnami: 0,78, 0,55 i 0,39.

## Kryteria z planu

| kryterium | wynik | spełnione |
|---|---|---|
| K1: w ramieniu `bez` rozdzielenie odświeżane w kroku 4500 wyższe niż w 1000 w co najmniej 8 z 10 ziaren | 4 z 10 (średnio 7,26 → 6,78) | nie |
| K2: średnie rozdzielenie odświeżane z kroków 3500-4500 wyższe niż w `bez` w co najmniej 8 z 10 ziaren, osobno dla `losowe` i `koopman` (mAP wyższe w 10 z 10 w obu) | `losowe` 6 z 10 (6,99 wobec 6,88); `koopman` 9 z 10 (7,29) | tylko `koopman` |
| K3: dolna granica przedziału korelacji wewnątrz chwili powyżej zera | rozdzielenie odświeżane 0,12 [0,00; 0,25]; wersja prosta 0,18 [0,04; 0,31]; rozdzielenie stałe 0,13 [−0,01; 0,27]; siła stała −0,14 [−0,31; 0,07] | odświeżane tak (granica 0,002), stałe nie |
| K4, K5 | opisowe, tabele niżej | opis |

Warunek rozstrzygający z planu („jeśli K2 i K3 spełnione dla rozdzielenia odświeżanego, a nie dla siły stałej, to
tożsamość ma sens jako cel, ale trzeba ją mierzyć kierunkiem odświeżanym”) jest spełniony tylko połowicznie i przy
bardzo małych efektach: korelacja 0,12 to 1,5 procent zmienności. Razem z K1 wynik czytam tak: ta definicja
tożsamości nie jest tym, czego brakuje detektorowi.

## Tabela 1. Przebieg w czasie (średnie z 10 ziaren)

| krok | mAP@0,5 (`bez` / `losowe` / `koopman`) | rozdzielenie odświeżane (`bez`) | rozdzielenie wzdłuż kierunku stałego | siła stała (`bez` / `losowe` / `koopman`) | cosinus z kierunkiem z kroku 1000 | cosinus z kierunkiem sprzed 500 kroków |
|---:|---|---:|---:|---|---:|---:|
| 50 | 0,000 | 10,7 | 3,0 | 3,41 | 0,01 | |
| 100 | 0,02 | 9,5 | 2,1 | 4,03 | 0,07 | |
| 300 | 0,10 | 8,5 | 3,0 | 5,18 | 0,15 | |
| 500 | 0,20 | 7,4 | 4,4 | 5,21 | 0,35 | |
| 950 | 0,32 | 7,3 | 9,2 | 5,70 | 0,74 | 0,28 |
| 1000 | 0,32 | 7,3 | 27,4 (dopasowanie) | 6,13 | 1,00 | 0,35 |
| 1050 | 0,32 | 7,3 | 8,6 | 5,92 | 0,77 | 0,38 |
| 1500 | 0,34 / 0,38 / 0,37 | 7,0 | 5,0 | 5,18 / 5,41 / 5,44 | 0,45 | 0,45 |
| 2500 | 0,40 / 0,43 / 0,44 | 7,2 | 3,1 | 5,06 / 4,65 / 4,77 | 0,25 | 0,52 |
| 3500 | 0,40 / 0,44 / 0,47 | 7,0 | 2,5 | 5,07 / 4,49 / 5,01 | 0,14 | 0,48 |
| 4500 | 0,42 / 0,49 / 0,50 | 6,8 | 1,8 | 5,30 / 4,36 / 4,37 | 0,12 | 0,50 |

Kolumny bez podziału na ramiona podają ramię `bez` (do kroku 1000 wspólne). mAP w krokach 50, 950 i 1050 to
najbliższa ocena (co 100 kroków).

## Tabela 2. Prognoza zmiany na 500 kroków z krzywej pozostałych dziewięciu ziaren (K4)

Umiejętność `1 − SSE/SSE „bez zmian”`. Starty co 500 kroków. Start w kroku 1000 pomijam w drugiej kolumnie, bo tam
kierunek stały jest dopasowany na tych samych danych i sam spadek po nim daje pozorną przewidywalność.

| wielkość | starty 1000-4000 | starty 1500-4000 | prognoza „powtórzy się ostatnia zmiana” |
|---|---:|---:|---:|
| mAP@0,5 | 0,21 | 0,11 | −1,5 |
| siła stała, średnia po sondach | −0,03 | −0,11 | −1,8 |
| siła stała, każda sonda osobno | −0,08 | −0,11 | −1,6 |
| rozdzielenie odświeżane | −0,12 | −0,11 | −1,8 |
| rozdzielenie wzdłuż kierunku stałego | 0,99 | 0,31 | −81 |
| cosinus z kierunkiem z kroku 1000 | 0,92 | 0,53 | −3,8 |

Siła stała zmienia się średniokwadratowo o 0,54 na 50 kroków i o 0,93 na 500 kroków, przy rozrzucie między ziarnami
1,3-1,8 w kroku 4500. To wielkość, która skacze, a nie płynie.

mAP też skacze: między ocenami odległymi o 100 kroków zmienia się średniokwadratowo o 0,038-0,040, a przez 500 kroków
rośnie średnio o 0,013 (`bez`) i 0,021-0,022 (ramiona z dokładaniem). Rozrzut między ziarnami to 0,032-0,034.
Pojedyncza ocena mAP ma więc błąd tego samego rzędu co różnica między ramionami; do porównań nadaje się tylko średnia
z wielu ocen (pole pod krzywą), jak w analizach kampanii.

## Tabela 3. Powtarzalność między ziarnami i między ramionami (K5)

| krok | CKA układu sond między ziarnami (`bez`) | cosinus kierunków między ziarnami | CKA `bez` wobec `losowe`, to samo ziarno | cosinus kierunków `bez` wobec `losowe`, to samo ziarno |
|---:|---:|---:|---:|---:|
| 1000 | 0,78 | 0,01 | 1 (ten sam stan) | 1 |
| 2500 | 0,55 | 0,00 | 0,69 | 0,32 |
| 4500 | 0,39 | −0,01 | 0,54 | 0,22 |

CKA układu sond z tym samym biegiem na końcu (krok 4500): 0,50 w kroku 500, 0,51 w kroku 1000, 0,59 w kroku 2500,
0,62 w kroku 3500. Układ sond względem siebie zmienia się przez całe uczenie i nie zbiega do wspólnego między ziarnami.

## Co z tego wynika

1. **Wektor tożsamości z pętli jest migawką jednej chwili.** Dopasowany na 384 próbkach w 768 wymiarach daje w
   chwili dopasowania rozdzielenie 27, a 50 kroków dalej 9, czyli jest w dwóch trzecich dopasowany do chwilowego
   stanu drgań. Potem starzeje się powoli i powtarzalnie.
2. **Prognozowanie siły wzdłuż tego wektora nie ma przedmiotu.** Siła skacze o 0,5 w 50 kroków, nie wiąże się z mAP
   i nie da się jej przewidzieć nawet z krzywej innych przebiegów. Wynik 28,7 procent z kampanii 4.3 (Langevin z
   jądrem na 500 kroków) to prognoza powrotu do średniej, zgodnie z obrazem drgań wokół środka.
3. **Tożsamość w sensie „narzędzie czy tło” nie jest wąskim gardłem.** Jest pełna od początku, także gdy detektor
   nic nie wykrywa. Wyrocznia z 4 października pokazała, czego brakuje: dokładnej ramki (AP75 z 0,02 do 0,77).
   Celem doboru powinno być to, czego potrzebuje głowa, a nie rozdzielenie, które już jest.
4. **Jeśli tożsamość ma zostać celem, to w definicji parowej i odświeżanej.** Pary „z narzędziem i bez” (J128 albo 8
   kierunków z par) kontrolują scenę; tych zapisy tej kampanii nie pozwalają policzyć. Da się to sprawdzić samą
   inferencją na zapisanych wagach końcowych 100 biegów iteracji 2 (propozycja Z7 w docs/28).

## Zastrzeżenia

- Definicja `stary` (narzędzie przeciw tłu i scenie); 12 wycinków w 16 wersjach; podział 9 na 3 wycinki.
- Kierunek LDA w 768 wymiarach z 384 próbek jest szumnym estymatorem: część „obrotu” na krótkim odstępie (cosinus
  0,75 po 50 krokach) to szum estymacji i drgania, a nie zmiana reprezentacji.
- Trzy ramiona jednej kampanii, kroki do 4500; korelacje, nie przyczyny.
- CKA liczone na 180 sondach z 12 wycinków ma dużą część struktury wspólną dla wersji jednego wycinka.

## Czego to NIE dowodzi

- Nie dowodzi, że tożsamość w definicji parowej (J128) nie wiąże się z mAP; tej definicji tu nie mierzono.
- Nie dowodzi, że sieć nie używa kierunku narzędzie-tło; mierzona jest odczytywalność, nie użycie przez głowę.
- Nie dowodzi, że inne miary niezmienne (na przykład po dopasowaniu ortogonalnym między chwilami) też są
  nieprzewidywalne; sprawdzono rozdzielenie, cosinus kierunków i CKA.
