# Próba sterowania wejściem FastGAN, 2026-09-23

Status: ukończona lokalnie; szczegóły i ograniczenia w `WYNIKI_PROBY_TRUDNYCH.md`. Wagi sieci pozostały niezmienione.

Użytkownik zatwierdził próbę słowami „zrób tak”, po przedstawieniu porównania losowania, błędu detektora i dodatkowego sygnału Koopmana. To mały eksperyment diagnostyczny. Nie trenujemy wag generatora ani detektora i nie rozpoczynamy kampanii RF-DETR.

## Pytanie i zakres

Czy zmiana wejścia zamrożonego FastGAN pozwala znaleźć trudniejsze otoczenie NIEZMIENIONEGO prawdziwego cystotomu? Czy dodatkowy sygnał Koopmana pomaga bardziej niż sam błąd detekcji? To próba na wycinkach 256 × 256, nie ocena jakości detektora na całych filmach.

Używamy ukończonego generatora EMA po 50 000 kroków, detektora uniform_time_b500_s42 oraz zamrożonej bazy PCA odpowiadającej dokładnie tym wagom. Punktem wyjścia są prawdziwe wycinki treningowe z ręcznymi maskami cystotomu. Podział między dopasowaniem operatora, kontrolą operatora i próbą optymalizacji odbywa się po operacjach. Generator mógł widzieć te operacje w swoim treningu; nie jest to niezależna ocena uogólniania generatora.

## Zachowanie narzędzia i ograniczenia

Generator daje obraz RGB i maskę, ale jego maska nie jest prawdą odniesienia. Zachowujemy prawdziwe piksele narzędzia oraz jego najbliższego otoczenia. W pozostałej części mieszamy prawdziwy obraz z tłem proponowanym przez FastGAN, ze stałym udziałem 0,35. Przed mieszaniem zmniejszamy i powiększamy obraz generatora, aby ograniczyć przenoszenie cienkich syntetycznych narzędzi. Nie gwarantuje to poprawności medycznej ani braku dodatkowych obiektów: galeria wymaga kontroli wzrokowej. Maska prawdziwego cystotomu pozostaje stała, a zachowanie jego pikseli sprawdzamy liczbowo.

Kontrole: oryginał, wklejenie na własne tło (powinno odtworzyć oryginał), losowy FastGAN, najlepszy z budżetowo dopasowanego losowania oraz optymalizacja wejścia generatora. Te same punkty startowe i zakres zmian we wszystkich porównaniach. Nie przypisujemy efektu Koopmanowi bez przewagi nad optymalizacją samej detekcji.

## Cząstki i Koopman

Cząstka jest tokenem siatki backbone. Profil: `(z-beta)/abs(gamma)`, następnie rzut na istniejącą bazę PCA. Używamy udziału rzeczywistej maski w powierzchni tokena zamiast całego prostokąta. Kierunek RDZEN jest dodatkowym odczytem, nie certyfikatem prawdziwej tożsamości.

Operator afiniczny EDMD dopasowujemy na odpowiadających sobie tokenach narzędzia przed i po rozjaśnieniu otoczenia o 15 procent. Instrument i jego pozycja są niezmienione. Wybieramy do sześciu niestałych funkcji własnych z wartościami własnymi najbliższymi +1; odrzucamy trywialną funkcję stałą i sprawdzamy odległość od +1. Podajemy wyniki na innych operacjach, w tym błąd operatora wobec przewidywania „nic się nie zmienia” i dopasowania z potasowanymi parami. Jeśli operator nie przewiduje lepiej, nie ogłaszamy jego użyteczności na podstawie samego widma.

Sygnał optymalizacji jest błędem tego operatora w wybranych modach, przeskalowanym rozrzutem reszt na zbiorze dopasowania. To nie jest pełne równanie Langevina ani pomiar pamięci czasu. Brak sekwencji wideo w tym pilocie wyklucza wniosek o jądrze pamięci.

## Ramiona i budżet

- Oryginał i kontrola wklejenia.
- Losowy obraz FastGAN z ustalonego punktu startowego.
- Najlepszy z 20 losowych wejść w tym samym otoczeniu punktu startowego; kontrola budżetu wyszukiwania.
- 20 kroków gradientowych wejścia generatora według błędu detekcji.
- Taki sam przebieg z dodanym błędem Koopmana, współczynnik 0,2 przy `log(1+reszta)`.

Docelowa mała próba: 32 operacje do operatora, 16 do jego niezależnej kontroli, 12 do porównania ramion. Przed nią krótki test techniczny w osobnym katalogu. W ramach każdego obrazu maksymalna odległość wejścia od startu wynosi 0,35 RMS. Ziarna, stan wejścia generatora i stały szum jego warstw są zapisywane. Wybieramy końcowy krok, nie najlepszy krok po obejrzeniu wyniku; losowanie wybiera najlepszą spośród kandydatur według jawnego kryterium detekcji.

## Wyniki i warunek przejścia dalej

Zapisujemy trafienie (dowolna ramka cystotomu z pewnością i IoU co najmniej 0,5), siłę najlepszego dopasowania, kontrast kierunku RDZEN, resztę Koopmana, mapy tokenów, błąd zachowania pikseli i pełną galerię. Raport zawiera wszystkie operacje, także takie, na których oryginalny wycinek był już chybiony. Osobno liczymy nowe chybienia wśród oryginalnych trafień.

Nie uznajemy błędu na obrazie z artefaktami za użyteczną próbkę uczącą. Dalsze wyszukiwanie prawdziwych klatek jest uzasadnione dopiero po ocenie galerii. Dowód poprawy uczenia wymaga osobnego porównania treningów na niezależnych filmach i nie jest wynikiem tej próby.
