# Wynik: prognoza od checkpointu 90 do 100, krok po kroku

Data: 2026-10-03. Kod: `kod/prognoza_900.py`, wykres `kod/wykres_900.py`. Liczby i rysunek:
`wyniki/prognoza_900/PROGNOZA_900.json`, `wyniki/prognoza_900/PROGNOZA_900.png`. Czas: 791 s, bez treningu.

## Pytanie

Jak dobrze równanie Langevina ze ściąganiem do średniej przewiduje latent na każdym z kroków 901..1000, gdy
startuje z kroku 900 (checkpoint 90)? Jak to wypada wobec NNDMD na tym samym odcinku?

## Aparat

Równanie (proces Ornsteina-Uhlenbecka przy stałym poziomie):

```
x̂(t+h) = μ_t + (x_t − μ_t) · exp(−κ h),   μ_t = średnia z ostatnich k stanów
```

Dwa parametry na wielkość: k z {21, 51, 101, 201} i κ z siatki. Dopasowanie wyłącznie na parach (start, cel)
z celem do kroku startu włącznie, starty od kroku 200. Kroki 901..1000 nie uczestniczą w dopasowaniu.
Porównania: prognoza bez zmian (zero), sama średnia z ostatnich 101 stanów, przedłużenie prędkości z ostatnich
10 kroków, sufit wyroczni (średnia ruchoma 21 wokół celu, skracana przy kroku 1000) oraz NNDMD z
`../maly_vit_nndmd_lokalne_przyrosty_hopper/run1819918/METRICS_TEST.json`. NNDMD liczono na tym samym biegu
uczenia (wagi k0000..k1000 z gęstej historii), ale na 1500 innych obrazach i wszystkich tokenach.

Wielkości: 30 losowych tokenów na obraz (360 jednostek, odpowiednik NNDMD), wszystkie tokeny narzędzia (218),
siła tożsamości (rzut tokenów narzędzia na kierunek LDA, 12 obrazów). Miara: 1 − suma błędów² / suma błędów²
prognozy bez zmian, po wszystkich jednostkach i współrzędnych. Dodatkowo średnia po startach 700, 720, ..., 900,
każdy z własnym dopasowaniem tylko na przeszłości.

## Wynik

| wielkość | k, κ (start 900) | h=1 | h=10 | h=50 | h=100 | średnio h=1..100 | średnio, starty 700–900 | sufit h=20..90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tokeny losowe | 51; 0,030 | 0,03 | 0,13 | 0,20 | 0,21 | 0,22 | 0,23 | 0,90 |
| tokeny narzędzia | 51; 0,044 | 0,02 | 0,19 | 0,20 | 0,22 | 0,25 | 0,26 | 0,87 |
| siła tożsamości | 101; 0,055 | -0,11 | 0,06 | 0,47 | -0,20 | 0,29 | 0,33 | 0,82 |

NNDMD (tokeny, checkpointy 91..100): -0,09, -0,22, -0,21, -0,27, -0,19, -0,21, -0,42, -0,52, -0,51, -0,38.
Przedłużenie ruchu: ujemne od pierwszego kroku, -31 dla tokenów losowych na h=100.

## Co z tego wynika

1. Równanie jest lepsze od prognozy bez zmian na prawie każdym kroku, o 20–30 procent dla tokenów. Zysk rośnie
   przez pierwsze 10–20 kroków i potem stoi. Nigdzie nie zbliża się do 85 procent.
2. Dopasowane κ = 0,03–0,055 na krok: odchylenie od poziomu maleje o połowę w 13–23 kroki. Po około 50 krokach
   prognoza to praktycznie sama średnia z ostatnich k stanów, dlatego linie niebieska i pomarańczowa się schodzą.
3. Parametry są stabilne: starty 700, 800 i 900 dają to samo k = 51 i prawie to samo κ dla tokenów.
4. NNDMD jest gorszy od prognozy bez zmian na wszystkich 10 checkpointach. Przedłużanie ruchu daje katastrofę.
   Ruch latentu na tym odcinku zawraca, więc wygrywa model, który ściąga z powrotem, a przegrywa każdy, który
   pcha do przodu.
5. Między równaniem (około 0,25) a sufitem (około 0,87) zostaje szeroka luka. To powolne krążenie poziomu,
   którego nie da się odczytać z przeszłości latentu.
6. Siła tożsamości jednego startu jest hałaśliwa (12 obrazów): na h=100 spada do -0,20, bo kilka obrazów odbiło
   w ostatnich krokach (panel A). Średnio po startach daje 0,33 i jest najwyższa z trzech wielkości.

## Czego to NIE dowodzi

- Nie dowodzi, że model z dodatkową informacją (stan optymalizatora, wagi, kolejność paczek) nie zrobi więcej.
- Porównanie z NNDMD jest na tym samym biegu, ale innych obrazach i innej liczbie tokenów.
- Jedno ziarno, jeden przebieg uczenia, okno kroków 700–1000.
