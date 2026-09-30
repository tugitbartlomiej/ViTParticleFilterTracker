# Serwowizja na pikselach: czy cystotom obraca się wokół nacięcia (2026-09-30)

Plan zatwierdzony przez użytkownika („zatwierdzam i licz”, 2026-09-30). Pomiar lokalny na adnotacjach, bez
uczenia sieci.

## Pytanie

Czy ruch narzędzia na obrazie da się opisać modelem z serwowizji (sterowania robotem na podstawie obrazu), w którym
narzędzie obraca się wokół nacięcia w rogówce jak wokół zawiasu? Wtedy ruch całego narzędzia na obrazie zależy od
kilku liczb (kąt osi, odległość końcówki od zawiasu), a prędkość pikseli jest liniowa względem ich prędkości.

## Dane

Cataract-LMM `3_Object_Tracking/annotations/<film>.zip`, plik `annotation.csv`: 170 filmów fazy kapsuloreksji,
każda klatka 720×480 z obrysem cystotomu (`cystotome_seg`), jego końcówką (`cystotome_tip`) i środkiem rogówki
(`cornea_center`, `cornea_bbox`).

## Test 1: czy osie przecinają się w jednym punkcie

- Oś trzonu w klatce: oś główna obrysu z momentów wielokąta (środek ciężkości + kierunek największego rozrzutu),
  skierowana od końcówki w stronę trzonu. Klatki z wydłużeniem obrysu poniżej 4 odrzucone.
- Okno 30 kolejnych klatek (co 15). Punkt najbliższy wszystkim osiom okna (najmniejsze kwadraty).
  Miara: mediana odległości osi od tego punktu.
- Podłoga: te same położenia i kierunki, ale kierunki przetasowane między klatkami okna (20 losowań).
- Okno liczy się, gdy ma co najmniej 20 dobrych klatek i rozrzut kątów osi co najmniej 2 stopnie
  (przy mniejszym punkt przecięcia jest nieokreślony).
- Przewidywania modelu zawiasu: odległość osi od punktu wyraźnie mniejsza niż w podłodze; punkt leży po stronie
  trzonu (nie końcówki); punkt leży blisko brzegu rogówki (odległość od środka około jednego promienia).
- Współrzędne względem środka rogówki (ruch oka odjęty) i surowe.

## Test 2: prognoza końcówki w klatce t+k (k = 1 i 5)

- `brak_zmiany`: p_t.
- `piksele_v`: stała prędkość na obrazie, p_t + (p_t − p_{t−k}).
- `zawias_v`: stała prędkość kąta i odległości końcówki od zawiasu; zawias z okna 30 klatek kończącego się na t
  (tylko przeszłość).
- Miary: mediana błędu w pikselach, umiejętność względem braku zmiany, filmy, w których `zawias_v` bije
  `piksele_v` (test znaków). Osobno klatki z ruchem końcówki co najmniej 2 px na krok.

## Kryterium

Model zawiasu przechodzi, gdy (1) w większości okien odległość osi od punktu jest wyraźnie mniejsza niż w podłodze,
(2) punkt leży po stronie trzonu i blisko brzegu rogówki, (3) `zawias_v` bije `piksele_v` w większości filmów.
Gdy (1) nie zachodzi, model zawiasu odpada.

## Ryzyka nazwane przed pomiarem

- Oko się porusza i obraca; odejmujemy tylko przesunięcie środka rogówki, obrotu oka nie.
- Z jednej kamery nie ma głębokości; odległość końcówki od zawiasu na obrazie miesza wsuwanie i pochylenie.
- Cystotom jest zagiętą igłą: oś z całego obrysu to kompromis między trzonem a zagiętym końcem.
- To rzut płaski modelu serwowizji, bez kalibracji kamery; pełna macierz interakcji wymaga głębokości.
