# Wynik: pchnięcie jednej paczki narasta przez pęd do około 3,2 pierwszego kroku, potem w większości wraca

Data: 2026-10-05. Plan: [PLAN.md](PLAN.md). Poprawka po wyniku: [AMENDMENT_PROJEKCJA_LOKALNA.md](AMENDMENT_PROJEKCJA_LOKALNA.md).
Bez treningu, bez GPU, bez Edenu. Dane: nagrania `../maly_vit_rownanie_latentu/nagranie_s47_sily` i
`nagranie_s48_sily` (latent 180 sond z narzędziem po każdym kroku i siły `S_nowa`, `S_ped`, `S_zanik`, docs/23).
Liczby: `wyniki/ODPOWIEDZ.json` (plan), `wyniki/ANALIZA_GRAMOW.json` i `wyniki/dluga/ODPOWIEDZ_DLUGA.json` (poprawka).
Macierze do dalszych regresji bez czytania nagrań: `wyniki/GRAMY.npz`, `wyniki/dluga/GRAMY_DLUGA.npz`.

Słowa: **pchnięcie** paczki to przesunięcie latentu sond, jakie daje sama część kroku AdamW pochodząca z gradientu
tej paczki (`S_nowa`). **Pęd** to część kroku pochodząca z pamięci poprzednich gradientów (`S_ped`). **Umiejętność**
to `1 − SSE/SSE prognozy „bez zmian”`. **Odchylenie** to różnica między latentem a jego średnią z ostatnich 101 kroków.

## Werdykt

1. **Jeden krok jest opisany dokładnie.** Pchnięcie i pęd wchodzą ze współczynnikami 0,98 i 0,98, umiejętność 0,954
   na późniejszych krokach ziarna 47 i 0,956 na ziarnie 48 bez ponownego dopasowania. To potwierdza równanie z docs/23.
2. **Skutek pchnięcia jest przejściowy.** Przez trzy kroki pchnięcie powtarza się przez pęd dokładnie tak, jak
   wynika z β1 = 0,9 (1; 2,0; 2,7). Szczyt wypada w piątym kroku i wynosi 3,2 pierwszego kroku. Potem skutek maleje:
   2,4 po 10 krokach, 1,2 po 20, 0,64 po 50, 0,5 po 100 i około 0,4 po 500. Wniosek z docs/25, że skutek paczki
   narasta do 5,8 pierwszego kroku, się nie potwierdza.
3. **Przyczyną powrotu jest sprzężenie zwrotne.** Pchnięcia z następnych 2-4 kroków są skierowane przeciwnie do
   kroku, który właśnie zaszedł (korelacja od −0,30 do −0,33), przeciwnie do bieżącego pędu (−0,31 i −0,35 po 1 i 2
   krokach) i przeciwnie do odchylenia od średniej (−0,21). Sieć sama cofa większość tego, co pchnięcie przesunęło.
4. **Jedna paczka rozstrzyga o ruchu przez około 10 kroków.** Znajomość bieżącej paczki podnosi umiejętność prognozy
   o 0,29-0,30 na 1 krok, o 0,22 na 5 kroków, o 0,06 na 10 kroków, o 0,01 na 20 kroków i o 0,002 na 50 kroków.
5. **Pchnięć nie da się składać liniowo ze stałą funkcją odpowiedzi.** Model A (wszystkie pchnięcia w oknie znane,
   każde policzone w swoim prawdziwym stanie) daje na 10 kroków 0,63, a nie 0,85.

## Kryteria z planu

| kryterium | wynik | spełnione |
|---|---|---|
| K1: `R(1)` odczytane z okna 50 kroków między 0,9 a 1,1 | −0,73, przedział 95 procent od −0,81 do −0,65; z okna 1 kroku 0,98 | nie |
| K2: kształt odpowiedzi według `R(1)`, `R(10)`, `R(20)` z okna 50 kroków | 1,43 i 0,48 przy ujemnym `R(1)`; kryterium zakładało K1 i w tej postaci nic nie rozstrzyga | nierozstrzygalne |
| K3: model A na 10 kroków co najmniej 0,85 na obu testach | 0,627 (ziarno 47, kroki 2000-2940) i 0,630 (ziarno 48) | nie |
| K4: przyrost umiejętności modelu C ponad B (opisowo) | 1 krok +0,286 i +0,301; 5 kroków +0,220 i +0,220; 10 kroków +0,063 i +0,061 | opis |

K1 nie jest spełnione z powodu opisanego w poprawce: współczynniki modelu A przy późnych pchnięciach niosą sprzężenie
zwrotne, a nie odpowiedź. Dla ostatniego pchnięcia w oknie współczynnik wynosi 0,98; 0,98; 0,88; 0,50; −0,32; −0,91;
−0,73 przy oknach 1, 2, 3, 5, 10, 20 i 50 kroków. Układ liniowy o stałej odpowiedzi dałby wszędzie tę samą liczbę.

## Tabela 1. Umiejętność trzech modeli (dopasowanie: ziarno 47, starty 1000-1949)

Pierwsza liczba: ziarno 47, starty 2000-2940. Druga: ziarno 48, te same kroki, bez ponownego dopasowania.

| horyzont | A: wszystkie pchnięcia w oknie | B: uczciwa (pęd, zanik, odchylenie) | C: B i paczka bieżąca | C − B |
|---:|---|---|---|---|
| 1 | 0,954 / 0,956 | 0,669 / 0,655 | 0,954 / 0,956 | +0,286 / +0,301 |
| 2 | 0,945 / 0,947 | 0,547 / 0,524 | 0,870 / 0,870 | +0,324 / +0,346 |
| 3 | 0,906 / 0,911 | 0,420 / 0,396 | 0,733 / 0,728 | +0,313 / +0,332 |
| 5 | 0,818 / 0,823 | 0,263 / 0,250 | 0,484 / 0,470 | +0,220 / +0,220 |
| 10 | 0,627 / 0,630 | 0,213 / 0,221 | 0,276 / 0,282 | +0,063 / +0,061 |
| 20 | 0,449 / 0,450 | 0,255 / 0,260 | 0,267 / 0,271 | +0,012 / +0,011 |
| 50 | 0,397 / 0,406 | 0,286 / 0,286 | 0,288 / 0,289 | +0,002 / +0,002 |

Ten sam model A dla ruchu wspólnego wszystkich sond (około 30 procent energii ruchu): 0,969 / 0,963 na 5 kroków,
0,899 / 0,882 na 10 kroków, 0,700 / 0,680 na 20. Dla ruchu własnego sondy (około 70 procent energii): 0,780 / 0,789,
0,583 / 0,589 i 0,423 / 0,424. Ruch wspólny składa się z pchnięć liniowo dużo lepiej niż ruch własny.

Dla porównania z wcześniejszych pomiarów: prawdziwa pełna siła każdego kroku (pchnięcie, pęd i zanik policzone w
prawdziwym stanie) daje na 10 kroków 0,985 (`../maly_vit_sufit_przewidywalnosci_100/WYNIKI_ROWNANIE_Z_PEDEM.md`).
Różnica między 0,985 a 0,63 to pęd policzony w prawdziwym stanie każdego kroku: nie da się go odtworzyć z samych
pchnięć paczek i stałych współczynników.

## Tabela 2. Funkcja odpowiedzi z projekcji lokalnej (poprawka)

Współczynnik przy pchnięciu bieżącej paczki w modelu C. Przedział: losowanie bloków po 50 startów, ziarno 47 razem
(starty 1000-2499). „Sam pęd” to wartość, jaką dałoby powtarzanie pchnięcia przez pamięć pędu bez żadnego powrotu:
`(1 − 0,9^h) / 0,1`. „Równanie docs/23” to `0,98 · (1 − 0,83^h) / 0,17`.

| horyzont | ziarno 47, kroki 1000-1749 | ziarno 47, kroki 1750-2499 | ziarno 48, kroki 2000-2499 | ziarno 47 razem i przedział 95 procent | sam pęd | równanie docs/23 |
|---:|---:|---:|---:|---|---:|---:|
| 1 | 0,98 | 0,98 | 0,98 | 0,98 [0,98; 0,98] | 1,00 | 0,98 |
| 2 | 1,99 | 1,95 | 1,98 | 1,97 [1,95; 1,99] | 1,90 | 1,79 |
| 3 | 2,74 | 2,70 | 2,73 | 2,72 [2,69; 2,76] | 2,71 | 2,47 |
| 4 | 3,11 | 3,14 | 3,12 | 3,13 [3,10; 3,17] | 3,44 | 3,03 |
| 5 | 3,16 | 3,28 | 3,17 | 3,22 [3,17; 3,27] | 4,10 | 3,49 |
| 7 | 2,74 | 3,00 | 2,80 | 2,86 [2,77; 2,96] | 5,22 | 4,20 |
| 10 | 2,34 | 2,46 | 2,41 | 2,40 [2,30; 2,49] | 6,51 | 4,87 |
| 15 | 1,57 | 1,80 | 1,68 | 1,68 [1,58; 1,76] | 7,94 | 5,41 |
| 20 | 1,20 | 1,27 | 1,32 | 1,23 [1,14; 1,32] | 8,78 | 5,63 |
| 30 | 0,97 | 0,98 | 0,98 | 0,97 [0,91; 1,04] | 9,58 | 5,74 |
| 50 | 0,60 | 0,69 | 0,66 | 0,64 [0,56; 0,71] | 9,95 | 5,76 |
| 100 | 0,57 | 0,44 | 0,47 | 0,51 [0,44; 0,58] | 10,0 | 5,76 |
| 200 | 0,50 | 0,37 | 0,45 | 0,44 [0,37; 0,50] | 10,0 | 5,76 |
| 500 | 0,43 | 0,34 | 0,30 | 0,39 [0,31; 0,45] | 10,0 | 5,76 |

Trzy zbiory dają tę samą krzywą do 50 kroków z dokładnością do 0,1-0,3. Powyżej 50 kroków okna celów zachodzą na
siebie, więc przedziały są zaniżone; rozrzut między trzema zbiorami (0,30-0,43 na 500 kroków) lepiej oddaje niepewność.

Ta sama odpowiedź osobno dla dwóch części ruchu (ziarno 47 razem): ruch wspólny sond 3,30 po 5 krokach, 0,78 po 50,
0,71 po 500 (przedział od 0,50 do 0,89); ruch własny sondy 3,17 po 5 krokach, 0,58 po 50, 0,24 po 500 (od 0,18 do
0,30). Część pchnięcia wspólna dla wszystkich sond zostaje na dłużej niż część właściwa jednej sondzie.

Pchnięcie pędu z chwili `t` ma podobny przebieg: 0,98; 1,65; 1,99; 2,06 (szczyt, 4 kroki); 1,49 po 10; 0,61 po 20;
około 0,4 od 40 do 500 kroków.

Kontrola placebo (pchnięcie paczki sprzed 100 kroków): współczynnik od −0,06 do +0,03 dla horyzontów do 10 kroków.
Dla dłuższych horyzontów jest ujemny (od −0,06 do −0,35 na 15-500 kroków, w obu częściach ziarna 47). Na długim
horyzoncie to nie jest czyste placebo: stare pchnięcie nadal powoli wygasa, więc w przyszłym ruchu widać jego dalszy
zanik.

## Tabela 3. Skąd wiadomo, że to sprzężenie zwrotne (ziarno 47, starty 1000-1949; w nawiasie ziarno 48)

| wielkość | wartość |
|---|---|
| korelacja pchnięcia z krokiem, który ono tworzy | 0,47 (0,46) |
| korelacja pchnięć z kroków `t+1`, `t+2`, `t+3`, `t+5`, `t+10` z krokiem wykonanym w `t` | −0,14; −0,30; −0,33; −0,14; −0,05 (−0,13; −0,28; −0,30; −0,12; −0,04) |
| korelacja pędu z chwili `t` z pchnięciami w `t`, `t+1`, `t+2`, `t+3`, `t+5` | −0,13; −0,31; −0,35; −0,27; −0,05 (−0,12; −0,29; −0,32; −0,25; −0,04) |
| korelacja odchylenia od średniej z pchnięciami w `t`, `t+1`, `t+2`, `t+3`, `t+5` | −0,21; −0,19; −0,13; −0,07; 0,00 (−0,19; −0,17; −0,12; −0,06; 0,00) |
| autokorelacja pchnięć przy opóźnieniu 1, 2, 3, 4, 5, 7, 10 | 0,23; 0,01; −0,14; −0,19; −0,15; 0,02; 0,02 (0,22; 0,02; −0,13; −0,18; −0,14; 0,02; 0,03) |
| korelacja pędu z chwili `t` z przesunięciem po 1, 5, 10, 20, 50 krokach | 0,80; 0,37; 0,16; 0,02; −0,01 |
| korelacja odchylenia z przesunięciem po 1, 5, 10, 20, 50 krokach | −0,04; −0,29; −0,42; −0,50; −0,53 |
| część pchnięcia wyznaczona przez pęd, zanik i odchylenie z tej samej chwili | 5,5 procent (4,6) |
| część pchnięcia wyznaczona przez poprzednie pchnięcie i stan | 14,8 procent (13,2) |
| część pchnięcia wyznaczona przez 9 albo 49 wcześniejszych pchnięć i stan | 11,0 i 10,9 procent (9,6 i 9,5) |

Współczynnik przy odchyleniu od średniej w modelu C: −0,16 po 5 krokach, −0,40 po 10, −0,59 po 20, −0,74 po 50 i
około −0,79 od 150 kroków. Połowa odchylenia znika w około 14 kroków, a w długim czasie znika cztery piąte.

## Co z tego wynika

1. **Latent zachowuje się jak cząstka z bezwładnością w dołku, który sam powoli płynie.** Pchnięcie rozpędza ją
   przez pęd (trzy kroki dokładnie jak w AdamW), po czym siła zwrotna ściąga ją z powrotem. Okres tego ruchu to
   około 8-10 kroków (pchnięcia zmieniają znak po 3-5 krokach), a odchylenie od średniej zanika w połowie po około 14
   krokach. Równanie jednego kroku z docs/23 nie ma członu zwrotnego, dlatego jest dokładne na 1-4 kroki i zawyża
   skutek dwukrotnie po 10 krokach, 4,6 raza po 20 i 9 razy po 50.
2. **Trwały ślad jednej paczki to około 0,4 jej pierwszego kroku**, a nie 3,2 ze szczytu ani 5,8 z ekstrapolacji.
   Dla części właściwej jednej sondzie około 0,24, dla części wspólnej sond około 0,7.
3. **Ocena paczki po jednym kroku mierzy głównie fazę drgań.** Pchnięcie, które wygląda na duże po jednym kroku, w
   87 procentach szczytu zostanie cofnięte. Ranking kandydujących paczek trzeba sprawdzać po 20-50 krokach, a miarą
   powinna być część, która zostaje.
4. **Uczciwa prognoza całego ruchu ma dwa garby.** 0,64-0,67 na 1 krok (pęd), minimum 0,21 na 7 kroków, potem 0,29
   na 40-75 kroków (powrót do średniej) i około 0,2 na 500 kroków (0,19-0,23 w trzech zbiorach). Znajomość bieżącej
   paczki przestaje pomagać po 20 krokach.
5. **Pchnięcie paczki jest w chwili `t` prawie w całości nową informacją.** Stan sieci widziany przez pęd i
   odchylenie wyznacza 5,5 procent pchnięcia, a razem z wcześniejszymi pchnięciami 11-15 procent. To jest
   bezpośrednia przyczyna niskiego sufitu prognoz na wiele kroków z samej historii.
6. **Do sterowania nadaje się postać z członem zwrotnym**: stan to położenie, prędkość i odchylenie od wolnej
   średniej; wejściem jest pchnięcie paczki. Stała macierz odpowiedzi na pchnięcia nie wystarcza (K3), bo siła
   zwrotna zależy od stanu, w którym pchnięcie pada.

## Zastrzeżenia

- To regresja na zwykłym uczeniu, nie interwencja. Pchnięcie zawiera część wyznaczoną przez stan sieci; pęd i
  odchylenie kontrolują ją tylko częściowo.
- Dopasowanie na jednym ziarnie, sprawdzenie na późniejszych krokach tego ziarna i na drugim ziarnie. Kroki
  1000-3000, baza 1500 klatek bez dokładania, paczka 32, stały współczynnik uczenia 5e-4, β1 = 0,9.
- Współczynniki są wspólne dla 768 współrzędnych i 180 sond. Odpowiedź w wybranych kierunkach (na przykład kanały
  J128 albo kierunek tożsamości) może być inna i nie była liczona.
- Horyzonty powyżej 50 kroków: mało niezależnych okien, przedziały zaniżone.

## Czego to NIE dowodzi

- Nie dowodzi, że różnica między dwiema paczkami podanymi w tym samym stanie zachowa się tak samo jak średnia
  odpowiedź na pchnięcie. To zmierzą dopiero pary przebiegów różniące się jedną paczką.
- Nie mówi nic o stracie detekcji ani o mAP. Trwały ślad w latencie sond nie musi być użyteczny.
- Nie rozstrzyga, czy trwałe 0,4 to ruch w kierunkach, w których sieć się uczy, czy powolne dochodzenie do zera.
- Nie dotyczy sytuacji, w której zmienia się skład zbioru (dokładanie klatek co 500 kroków): to trwała zmiana
  średniego pchnięcia, a nie jedno pchnięcie.
