# Wynik: wagi błądzą losowo ze słabym trwałym dryfem, latent sond drga w miejscu

Data: 2026-10-05. Plan: [PLAN.md](PLAN.md). Poprawka po wyniku: [AMENDMENT_BLADZENIE_LOSOWE.md](AMENDMENT_BLADZENIE_LOSOWE.md).
Bez treningu, bez GPU, bez Edenu. Dane: 101 zapisów wag co 10 kroków i latent 180 sond z biegu gęstej historii
(`../maly_vit_ruch_czastek_gesta_historia/wyniki/pelny`, Eden 1818732, ziarno 47, kroki 0-1000). Liczby:
`wyniki/KREGOSLUP.json` (plan), `wyniki/POROWNANIE_Z_BLADZENIEM.json` i `wyniki/DRYF_OKNA.json` (poprawka), macierze
iloczynów przesunięć `wyniki/GRAMY_WAG.npz`.

Słowa: **kręgosłup** to jeden trwały kierunek, wzdłuż którego wagi powoli się przesuwają, a wszystko inne to drgania w
poprzek (Xu i in. 2026). **Błądzenie losowe**: każdy przyrost w nowym, przypadkowym kierunku. **Dryf**: część ruchu,
która w każdym oknie czasu idzie w tę samą stronę.

## Werdykt

1. **Kryterium K1 jest spełnione, ale nic nie rozstrzyga.** Pierwszy kierunek główny niesie 0,886 kwadratu
   przesunięcia wag od kroku 300 do 1000. Zwykłe błądzenie losowe daje 0,81, a z pamięcią pędu 0,815 (poprawka).
2. **Kryterium K2 nie jest spełnione.** Mediana cosinusa przesunięć w sąsiednich oknach 100 kroków wynosi 0,051
   (kwartyle 0,004 i 0,123, największa wartość 0,19), przy progu 0,5.
3. **Trwały dryf w wagach istnieje, ale jest słaby.** Przesunięcia w oknach odległych od siebie mają cosinus 0,05
   (okna 50 kroków), 0,10 (100 kroków) i 0,16 (200 kroków); dwa sąsiednie okna po 350 kroków 0,26. W oknie 100 kroków
   dryf niesie więc około jednej dziesiątej energii przesunięcia, reszta to dyfuzja. Najsilniejszy jest w końcowej
   normalizacji (0,30 w oknach 100 kroków) i w głowie (0,17), najsłabszy w łodydze i pierwszych blokach (0,09).
4. **Latent sond nie dyfunduje, tylko drga wokół wolno płynącego środka.** Przesunięcia w sąsiednich oknach są
   przeciwne (cosinus −0,43 dla 50 kroków, −0,36 dla 100, −0,22 dla 200), netto do drogi wynosi 0,039 (błądzenie
   losowe: 0,12). Trwały dryf w latencie to 2 procent energii przesunięcia w oknie 100 kroków i 5 procent w oknie 200
   kroków; po wygładzeniu latentu średnią z 50 kroków 4,5 i 9 procent.
5. Wniosek z planu na przypadek niespełnionego K2: w tym biegu nie ma wolnego kierunku, który dałoby się przedłużać
   na 100 kroków, ani w wagach, ani w latencie. Jest słaby dryf widoczny dopiero w oknach kilkuset kroków.

## Tabela 1. Wagi od kroku 300 do 1000 (70 przyrostów co 10 kroków)

| grupa wag | parametrów | udział 1. kierunku | netto do drogi | autokorelacja przyrostów: 10, 20, 30, 100 kroków | cosinus okien 100 kroków: sąsiednie / odległe | okna 200 kroków: sąsiednie / odległe | dwa okna po 350 kroków |
|---|---:|---:|---:|---|---|---|---:|
| cała sieć | 2 926 493 | 0,886 | 0,176 | 0,29; −0,01; −0,05; 0,01 | 0,06 / 0,10 | 0,19 / 0,16 | 0,26 |
| łodyga splotowa | 218 904 | 0,878 | 0,171 | 0,33; 0,00; −0,05; 0,00 | 0,02 / 0,09 | 0,15 / 0,16 | 0,25 |
| blok 0 | 444 864 | 0,884 | 0,173 | 0,31; −0,01; −0,06; 0,01 | 0,04 / 0,09 | 0,17 / 0,15 | 0,25 |
| blok 4 | 444 864 | 0,901 | 0,193 | 0,25; −0,02; −0,03; 0,02 | 0,15 / 0,14 | 0,29 / 0,20 | 0,33 |
| blok 5 (ostatni) | 444 864 | 0,887 | 0,175 | 0,22; −0,04; −0,04; 0,01 | 0,10 / 0,11 | 0,22 / 0,16 | 0,26 |
| końcowa normalizacja | 384 | 0,935 | 0,188 | −0,01; −0,10; −0,03; −0,01 | 0,29 / 0,30 | 0,55 / 0,42 | 0,53 |
| głowa | 38 021 | 0,906 | 0,180 | 0,15; −0,08; −0,04; 0,01 | 0,15 / 0,17 | 0,33 / 0,25 | 0,40 |
| błądzenie losowe (odniesienie) | | 0,811 | 0,120 | 0; 0; 0; 0 | 0 / 0 | 0 / 0 | 0 |
| błądzenie z korelacją sąsiednich przyrostów 0,3 | | 0,815 | 0,151 | 0,30; 0; 0; 0 | 0,02 / 0 | 0,01 / 0 | 0 |

W tabeli „sąsiednie” to średnia po wszystkich położeniach okna (mediana z planu dla całej sieci: 0,051).
Norma wag: 79,7 w kroku 0 i 81,2 w kroku 1000; przesunięcie od kroku 300 do 1000 ma normę 15,7. Przesunięcie z
rozgrzewki (kroki 0-300) ma z późniejszym (300-1000) cosinus 0,22.

Dopasowanie „stały dryf plus błądzenie z pamięcią jednego zapisu” do macierzy iloczynów (R² 0,994) daje dla całej
sieci stosunek energii dryfu do dyfuzji 0,006 na przyrost 10 kroków. Ten model nie ma ujemnej korelacji przy
opóźnieniach 20-50 kroków, którą widać w danych, więc zaniża dryf; miary z okien odległych są pewniejsze.

## Tabela 2. Latent 180 sond (średnia po tokenach narzędzia) od kroku 300 do 1000

| wielkość | latent narzędzia | latent tła | po wygładzeniu 50 kroków: narzędzie / tło |
|---|---:|---:|---|
| udział 1. kierunku głównego | 0,762 | 0,761 | |
| liczba kierunków do 90 procent | 12 | 10 | |
| netto do drogi | 0,039 | 0,042 | |
| autokorelacja przyrostów: 10, 20, 30, 100 kroków | −0,30; −0,12; −0,03; 0,03 | −0,29; −0,10; −0,04; 0,03 | |
| cosinus okien 50 kroków: sąsiednie / odległe | −0,43 / 0,01 | −0,42 / 0,01 | −0,26 / 0,02 i −0,23 / 0,02 |
| cosinus okien 100 kroków: sąsiednie / odległe | −0,36 / 0,02 | −0,32 / 0,03 | −0,21 / 0,05 i −0,16 / 0,06 |
| cosinus okien 200 kroków: sąsiednie / odległe | −0,22 / 0,05 | −0,16 / 0,08 | −0,03 / 0,09 i 0,03 / 0,16 |
| dwa okna po 350 kroków | −0,02 | 0,04 | 0,12 i 0,18 |
| przesunięcie z kroków 0-300 wobec 300-1000 | −0,33 | −0,29 | −0,31 i −0,29 |

Dla wielkości, która tylko drga wokół stałego środka, cosinus sąsiednich okien dąży do −0,5. Latent jest blisko tej
granicy: w skali 50-100 kroków jego ruch to prawie w całości drgania.

## Co z tego wynika

1. **Dwie przestrzenie, dwa rodzaje ruchu.** Wagi oddalają się od punktu startu jak w dyfuzji (przesunięcie rośnie z
   czasem, nic nie wraca), a latent sond pozostaje w ograniczonym obszarze i wraca. Większość dyfuzji wag odbywa się
   więc w kierunkach, których latent sond nie odczuwa. To zgadza się z obrazem z literatury: reprezentacja dryfuje
   w podprzestrzeni, która nie zmienia odczytu (Pashakhanloo i Koulakov 2023), a uczenie zachodzi w kierunkach
   łagodnych, podczas gdy w stromych sieć tylko oscyluje (Song i in. 2024).
2. **Dlaczego prognozy latentu na 100-500 kroków nie biją „bez zmian” o więcej niż 0,2-0,35.** Jeśli wielkość drga
   wokół środka, najlepsza uczciwa prognoza to sam środek, a ta daje co najwyżej 0,5 (błąd równy jednej wariancji
   drgań wobec dwóch dla „bez zmian”). Trwały dryf, który mógłby dodać resztę, to w latencie 2-10 procent energii.
   Cel 85 procent na całym ruchu latentu jest więc poza zasięgiem z powodu budowy tego ruchu, a nie z braku modelu.
3. **Gdzie jest coś wolnego.** W wagach głowy i końcowej normalizacji dryf jest 2-3 razy wyraźniejszy niż w
   łodydze. Jeśli szukać wielkości, którą da się przedłużać, to bliżej wyjścia sieci i w oknach kilkuset kroków.
4. **Porównanie z Xu i in. wymaga miary odpornej na dyfuzję.** Ich dwie miary liczone na jednej trajektorii (udział
   pierwszego kierunku oraz zgodność skumulowanego kroku z tym kierunkiem, u nich 0,15-0,34 dla 200 kroków) są
   niezerowe także dla błądzenia losowego. U nas zgodność przesunięcia w oknie 200 kroków z pierwszym kierunkiem
   wynosi 0,61 (okno 100 kroków: 0,45), a błądzenie losowe o tej samej liczbie przyrostów daje 0,50 (0,35). Nadwyżka
   nad błądzeniem jest mała i zgodna ze słabym dryfem z okien odległych. Bez długości trajektorii z ich pracy tych
   liczb nie da się zestawić wprost; pewne jest tylko, że u nas dryf w skali 1000 kroków nie dominuje nad dyfuzją.

## Zastrzeżenia

- Jedno ziarno, pierwsze 1000 kroków, baza 1500 klatek bez dokładania. Dryf słabnie w czasie (sąsiednie okna 200
  kroków: 0,25; 0,21; 0,16; 0,13 dla kolejnych położeń), więc liczby dotyczą tej fazy uczenia.
- Zapis co 10 kroków uśrednia drgania o okresie 8-10 kroków, które widać w nagraniach co krok
  (`../maly_vit_odpowiedz_latentu_na_paczke/WYNIKI.md`).
- Miary w wagach są euklidesowe: wszystkie 2,9 mln parametrów liczą się jednakowo, także te, od których wyjście
  sieci prawie nie zależy.
- Latent to średnia po tokenach narzędzia 180 sond; pojedyncze tokeny mogą zachowywać się inaczej.

## Czego to NIE dowodzi

- Nie dowodzi, że kręgosłupa nie ma w dłuższym uczeniu albo przy malejącym współczynniku uczenia.
- Nie dowodzi, że dryfu nie da się wydobyć lepszą miarą (na przykład ważoną wpływem parametru na wyjście sieci).
- Nie mówi, czy kierunek dryfu jest ten sam w innych ziarnach.
- Nie wiąże dryfu z jakością detekcji; mAP rośnie w tym biegu z 0 do 0,31, a tu mierzona jest tylko geometria ruchu.
