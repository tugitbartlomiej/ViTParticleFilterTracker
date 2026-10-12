# Wyniki: serwowizja na pikselach i zawias w nacięciu (2026-09-30)

Skrypt `zawias.py`, wynik `wyniki/ZAWIAS.json` (podsumowania i wszystkie okna), podgląd lokalny
`wyniki/ZAWIAS_PODGLAD.png` (6 wylosowanych okien; nie w repo, bo *.png jest ignorowane). 170 filmów, 143 s na CPU.
Pliki `wyniki/PROBA_3.json` i `PROBA_6.json` to próby na 3 i 6 filmach wcześniejszą wersją kodu.

## Odstępstwa od PLAN.md (dodane po próbie na 3 filmach, przed pełnym biegiem)

1. Stała prędkość okazała się gorsza od braku zmiany, bo końcówka często zawraca. Dodano więc prędkość tłumioną
   (`piksele_vb`, `zawias_vb`): współczynnik β dopasowany na jednej połowie filmów i oceniany na drugiej.
2. Dodano test kątów, czyli bezpośredni test równania serwowizji w rzucie płaskim: ruch końcówki prostopadły do osi
   = s · obrót osi (s = odległość końcówki od zawiasu). Przewidujemy obrót osi między t a t+k z ruchu końcówki
   i zawiasu z przeszłości. Podłoga: zawias z innej chwili tego samego filmu.
3. „Strona trzonu” liczona względem końcówki (zawias za końcówką, po stronie trzonu). Wersja z planu, względem
   środka obrysu, została w JSON jako `strona_trzonu_za_srodkiem`.

## Test 1: czy osie przecinają się w jednym punkcie (4215 okien po 30 klatek, 165 filmów, współrzędne oka)

- Osie przechodzą średnio 2,2 px od wspólnego punktu; po przetasowaniu kierunków 4,9 px.
  Iloraz w medianie 0,53; w 86 % okien poniżej 1, w 47 % poniżej 0,5.
- Punkt leży po stronie trzonu w 92 % okien, w medianie 144 px od końcówki wzdłuż osi (kwartyle 76–219 px).
- Odległość punktu od środka rogówki: mediana 1,26 promienia (kwartyle 0,85–1,80); w pasie 0,7–1,5 promienia
  45 % okien. Punkt leży więc zwykle w okolicy brzegu rogówki, ale z dużym rozrzutem.
- Słabość: w oknie 30 klatek oś obraca się tylko o około 3 stopnie (mediana), więc punkt przecięcia jest słabo
  wyznaczony. Na podglądzie część punktów leży przy brzegu rogówki (nacięcie), a część tuż za końcówką.

## Test 2: prognoza końcówki (współrzędne oka, β z drugiej połowy filmów)

| k | brak zmiany | piksele, stała prędkość | zawias, stała prędkość | piksele, tłumiona | zawias, tłumiona |
|---|---|---|---|---|---|
| 1 klatka, mediana błędu | 3,16 px | 4,29 | 4,31 | 3,20 | 3,21 |
| 1 klatka, umiejętność | 0 | −62 | −89 | +2,6 | +2,3 |
| 5 klatek, mediana błędu | 8,94 px | 11,00 | 11,07 | 8,92 | 9,01 |
| 5 klatek, umiejętność | 0 | −104 | −114 | −0,6 | −0,4 |

- Zawias tłumiony lepszy od pikseli tłumionych w 58 ze 164 filmów (k = 1) i 43 ze 163 (k = 5): model zawiasu
  przegrywa. Stała prędkość bez tłumienia: 83 ze 164, czyli rzut monetą.
- Końcówka jest prawie nieprzewidywalna z własnej przeszłości w skali 1–5 klatek: najlepszy model usuwa
  3 % błędu braku zmiany. Współczynniki β bliskie zeru (od −0,09 do 0,25) mówią, że prędkość prawie się nie utrzymuje.
- We współrzędnych obrazu obraz jest ten sam (piksele tłumione 5,8 % przy k = 1, zawias 4,0 %).

## Test 3: równanie serwowizji dla obrotu osi (pary klatek z |s| ≥ 20 px, obrót < 30°)

| układ, k | par | nachylenie (powinno być 1) | korelacja | korelacja z zawiasem z innej chwili | filmy: zawias lepszy od podłogi |
|---|---|---|---|---|---|
| oko, 1 | 53 260 | 0,08 | 0,14 | 0,12 | 96/163 (p = 0,014) |
| oko, 5 | 51 630 | 0,19 | 0,35 | 0,30 | 105/163 (p = 0,0001) |
| obraz, 1 | 53 475 | 0,07 | 0,12 | 0,12 | 93/165 (p = 0,06) |
| obraz, 5 | 51 942 | 0,13 | 0,28 | 0,24 | 112/164 (p = 0,000002) |

- Ruch końcówki prostopadły do osi jest 5–14 razy większy, niż wynikałoby z obrotu osi wokół zawiasu.
  Obrót osi tłumaczy więc tylko niewielką część ruchu końcówki; umiejętność przewidywania kąta jest ujemna.
- Zawias z tej samej chwili jest trochę lepszy od zawiasu z innej chwili (w większości filmów, p ≤ 0,014 dla oka),
  więc punkt niesie realną informację, ale mało.

## Werdykt według kryterium z PLAN.md

- (1) Osie zbiegają się wyraźniej niż po przetasowaniu: **tak** (iloraz 0,53, 86 % okien).
- (2) Punkt po stronie trzonu i przy brzegu rogówki: **częściowo** (strona trzonu 92 %; odległość 1,26 promienia
  w medianie, ale tylko 45 % okien w pasie 0,7–1,5).
- (3) Prognoza z zawiasem lepsza niż z samych pikseli w większości filmów: **nie** (58/164 i 43/163).

Geometria zawiasu jest widoczna: osie cystotomu zbiegają się w punkcie za końcówką, zwykle w okolicy brzegu rogówki.
Liniowe równanie serwowizji z nieruchomym zawiasem nie opisuje jednak ruchu końcówki klatka po klatce: większość tego
ruchu nie jest obrotem wokół nacięcia. Model zawiasu w tej postaci odpada jako model prognozy.

## Możliwe przyczyny (niesprawdzone)

- Obrót igły wokół własnej osi przesuwa zagięty koniec cystotomu w bok bez zmiany osi trzonu (czwarty stopień
  swobody, niewidoczny w osi).
- Oko obraca się i przesuwa, więc nacięcie wędruje; odejmujemy tylko przesunięcie środka rogówki.
- Szum adnotacji końcówki (liczby całkowite pikseli, półautomatyczne śledzenie) przy ruchu rzędu 3 px na klatkę.
- Oś z całego obrysu miesza trzon z zagiętym końcem.

## Czego to NIE dowodzi

- Nie obala serwowizji jako teorii: sprawdzono tylko rzut płaski z nieruchomym zawiasem, bez kalibracji i głębokości.
- Tylko cystotom (inne narzędzia nie mają kolumn w `annotation.csv`), tylko faza kapsuloreksji, jeden szpital (S1).
- Skale 1 i 5 klatek; wolniejszy ruch (sekundy) nie był badany.
