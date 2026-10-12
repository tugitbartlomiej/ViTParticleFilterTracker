# Wynik: mAP drga razem z wagami, a uśrednienie wag z ostatnich 100 kroków podnosi je o 0,05-0,11

Data: 2026-10-05. Plan: [PLAN.md](PLAN.md). Bez treningu i bez Edenu: sama ocena zapisanych wag na lokalnej karcie
(RTX 3070), 3000 klatek walidacji, ten sam kod oceny co w biegach. Liczby: `wyniki/SREDNIA_WAG.json` (plan, ziarno 47,
kroki 500-1000, zapisy co 10 kroków) i `wyniki/POZNA_FAZA_S48.json` (uzupełnienie po wyniku, ziarno 48, kroki
1000-3000, zapisy co 100 kroków).

Słowa: **wagi surowe** to stan sieci w jednym kroku. **Wagi uśrednione** to średnia arytmetyczna parametrów z kilku
ostatnich zapisów. Uczenie się nie zmienia: uśredniona kopia służy tylko do oceny.

## Werdykt

1. **mAP surowych wag drga.** Między zapisami odległymi o 10 kroków zmienia się średniokwadratowo o 0,041, przy
   trendzie +0,026 na 100 kroków; odchylenie wokół trendu wynosi 0,034, a rozstęp w oknie 100 kroków 0,09-0,15.
   Przykład: 0,294 w kroku 660, 0,160 w kroku 680, 0,314 w kroku 700.
2. **Uśrednianie wag usuwa drgania i podnosi mAP (K2 spełnione).** Wagi uśrednione z ostatnich 100 kroków dają mAP
   wyższe od średniego mAP surowych wag z tego okna o 0,062-0,093 (5 z 5 chwil) i wyższe od wag z końca okna w 4 z 5
   chwil (o 0,051-0,109; raz niższe o 0,003). W kroku 1000: 0,310 surowe, 0,419 uśrednione.
3. **Wynik powtarza się na drugim ziarnie i w późniejszej fazie.** Ziarno 48, kroki 1400-3000, zapisy co 100 kroków:
   średnia z 5 zapisów jest lepsza od wag z końca okna we wszystkich 9 chwilach, średnio o 0,054 (od 0,011 do 0,105),
   i lepsza od najlepszego surowego zapisu w oknie we wszystkich 9.
4. **Statystyki BatchNorm nie mają znaczenia** przy tych oknach: uśrednione jak wagi albo przeliczone na 512 klatkach
   bazy dają to samo z dokładnością do 0,01. Samo przeliczenie statystyk na wagach surowych nic nie zmienia.

Dla skali: zysk z doboru klatek w kampanii 4.3 wobec losowych to 0,016 pola pod krzywą i 0,022 na końcu uczenia.

## Kryteria z planu

| kryterium | wynik | spełnione |
|---|---|---|
| K0: mAP surowych wag w krokach 500-1000 co 100 zgodne z `OCENY.json` biegu do 0,005 | różnice od −0,0017 do +0,0018 | tak |
| K1 (opisowo): drgania mAP | zmiana co 10 kroków 0,041; odchylenie wokół trendu 0,034; trend +0,026 na 100 kroków; rozstęp w oknach 100 kroków 0,088; 0,155; 0,153; 0,105; 0,100 | opis |
| K2: okno 100 kroków lepsze od średniego mAP okna o co najmniej 0,01 w 4 z 5 chwil ORAZ lepsze od końca okna w 4 z 5 | 5 z 5 (+0,070; +0,062; +0,074; +0,093; +0,082) oraz 4 z 5 (+0,051; −0,003; +0,062; +0,059; +0,109) | tak |
| K3 (opisowo): okno i BatchNorm | tabela 1 | opis |

## Tabela 1. Ziarno 47, zapisy co 10 kroków (mAP@0,5, 3000 klatek walidacji)

| krok | surowe | średnie mAP surowych w oknie 100 kroków | najlepsze surowe w oknie 100 kroków | uśrednione, okno 50 kroków | uśrednione, okno 100 kroków | uśrednione, okno 200 kroków | średnia wykładnicza (waga nowego zapisu 0,2) | średnia wykładnicza (0,1) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 600 | 0,256 | 0,237 | 0,268 | 0,299 | 0,307 | 0,290 | 0,309 | 0,287 |
| 700 | 0,314 | 0,250 | 0,314 | 0,299 | 0,312 | 0,319 | 0,316 | 0,319 |
| 800 | 0,320 | 0,308 | 0,362 | 0,382 | 0,382 | 0,357 | 0,380 | 0,367 |
| 900 | 0,348 | 0,314 | 0,378 | 0,421 | 0,407 | 0,405 | 0,423 | 0,411 |
| 1000 | 0,310 | 0,337 | 0,377 | 0,405 | 0,419 | 0,426 | 0,420 | 0,428 |

Wagi uśrednione: statystyki BatchNorm przeliczone na 512 klatkach bazy. Z BatchNorm uśrednionym jak wagi: różnice
do 0,010 (na przykład okno 100 kroków: 0,305; 0,320; 0,380; 0,408; 0,416). Surowe wagi z przeliczonym BatchNorm:
0,258; 0,313; 0,321; 0,349; 0,307, czyli bez zmiany.

Najwyższe mAP pojedynczego surowego zapisu w całym odcinku 500-1000 to 0,378 (krok 880). Wagi uśrednione
przekraczają tę wartość od kroku 800.

## Tabela 2. Ziarno 48, zapisy co 100 kroków (uzupełnienie po wyniku)

| krok | surowe | najlepsze surowe z 5 zapisów | uśrednione z 3 zapisów (200 kroków) | uśrednione z 5 zapisów (400 kroków) |
|---:|---:|---:|---:|---:|
| 1400 | 0,387 | 0,387 | 0,402 | 0,400 |
| 1600 | 0,413 | 0,413 | 0,425 | 0,424 |
| 1800 | 0,407 | 0,413 | 0,441 | 0,442 |
| 2000 | 0,443 | 0,443 | 0,454 | 0,465 |
| 2200 | 0,393 | 0,443 | 0,459 | 0,468 |
| 2400 | 0,375 | 0,443 | 0,481 | 0,480 |
| 2600 | 0,415 | 0,430 | 0,461 | 0,485 |
| 2800 | 0,450 | 0,453 | 0,501 | 0,499 |
| 3000 | 0,385 | 0,453 | 0,470 | 0,490 |

Surowe mAP tego ziarna w krokach 1000-3000: odchylenie wokół trendu 0,033, trend +0,005 na 100 kroków, zmiana między
sąsiednimi ocenami co 100 kroków 0,047.

Dla skali, z zapisów kampanii v2 (10 ziaren, wagi surowe): w kroku 4500 ramię bez dokładania ma średnio 0,42, a
ramiona z dołożonymi 938 klatkami 0,49-0,50. Wagi uśrednione ziarna 48, bez żadnego dokładania, mają 0,49-0,50 już
w kroku 2800-3000.

## Co z tego wynika

1. **Sieć krąży wokół lepszego modelu, niż sama w danej chwili jest.** To funkcyjny odpowiednik obrazu z pomiarów
   latentu: stan chwilowy drga wokół wolno płynącego środka, a środek (wagi uśrednione) jest o 0,05-0,11 mAP lepszy.
2. **Pojedyncza ocena mAP surowych wag jest złą miarą.** Jej błąd (0,034) jest większy niż postęp na 100 kroków i
   niż zysk, o który toczą się porównania ramion. Ocena co 100 kroków próbkuje fazę drgań.
3. **Najtańsze przyspieszenie uczenia nie wymaga doboru danych.** Wystarczy trzymać uśrednioną kopię wag i ją
   oceniać. Każde twierdzenie o przyspieszeniu przez dobór klatek musi być odtąd pokazane na wagach uśrednionych,
   inaczej recenzent słusznie zapyta, czy zysk nie znika po uśrednieniu albo po obniżeniu współczynnika uczenia.
4. **Do prognozy i sterowania właściwym obiektem jest środek.** Latent, tożsamość i skutek paczki warto liczyć na
   wagach uśrednionych: tam nie ma drgań, które zajmują 90-98 procent ruchu latentu surowych wag.

## Zastrzeżenia

- Dwa ziarna (47: kroki 500-1000; 48: kroki 1000-3000), baza 1500 klatek bez dokładania. Walidacja, nie zbiór testowy.
- Uśrednianie po fakcie z zapisów co 10 albo co 100 kroków. Średnia prowadzona w uczeniu co krok może dać nieco
  inne liczby; nie sprawdzano jej.
- mAP@0,5 na 3000 klatek; dla AP75 i AP50:95 nie liczono.
- Okna zachodzą na siebie, pięć i dziewięć chwil pomiaru to mała próba; kryterium K2 było ustalone przed wynikiem
  tylko dla ziarna 47.

## Czego to NIE dowodzi

- Nie dowodzi, że dobór klatek przestaje pomagać po uśrednieniu wag. Tego nie mierzono: zapisów wag ramion z
  dokładaniem w odstępach krótszych niż 500 kroków lokalnie nie ma.
- Nie dowodzi, że zysk utrzyma się do końca długiego uczenia ani przy malejącym współczynniku uczenia (wtedy drgania
  maleją same).
- Nie dowodzi, że wagi uśrednione są lepszym punktem startu dalszego uczenia; literatura mówi, że nie są.
- Nie rozstrzyga, jaką część zysku dałoby zwykłe obniżenie współczynnika uczenia pod koniec.
