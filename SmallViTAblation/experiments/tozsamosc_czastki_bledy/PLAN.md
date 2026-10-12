# Cząstki narzędzia, dynamika i błędy detektora

Status: przygotowanie kodu, bez uruchomienia. Polecenie użytkownika z 2026-09-23: „na razie niczego nie licz”.

## Pytanie i zakres

Czy przestrzenny rozkład sygnału narzędzia oraz jego dynamika wskazują błędy zamrożonego detektora na innych operacjach lepiej niż pewność detektora i statyczny opis tej samej klatki?

To etap diagnostyczny. Nie mierzy przyrostu jakości po treningu. Wysoka skuteczność wskazywania błędów nie oznacza, że są one usuwalne przez dodanie tych klatek. Następny, oddzielny eksperyment musiałby porównać trening na równych budżetach i osobnych filmach.

## Dane i znaczenie cząstek

- Istniejące zapisy `CZASTKI_*`: `profil_red`, `frame_ids`, `pred_box`, `score`, `baza.npz`, `meta.json`.
- Cząstka jest tokenem w stałej komórce obrazu. Kod nie zakłada śledzenia tego samego fragmentu metalu.
- Maska pochodzi z `cystotome_seg` oryginalnych adnotacji. Nie zastępujemy jej prostokątem detektora ani odcinkiem trzonka.
- Kod przygotowania danych przyjmuje jawny indeks wiążący `frame_id` z segmentacją i rozmiarem obrazu. Zapobiega zgadywaniu numeracji klatek w archiwach. Dołączono eksporter z archiwów CSV, wymagający jawnego wzorca nazwy obrazu.
- Zapisane profile korzystają z wcześniej dopasowanej bazy PCA. Jej możliwa znajomość filmów oceny oraz znany problem powtarzalności rejestratora pozostają ograniczeniami analizy eksploracyjnej. Nie nazywamy jej ostateczną niezależną walidacją.
- Test końcowy (`split=test`) jest wykluczony. Dla pierwszego etapu wybieramy jednoznacznie split `val` lub `train`; nie mieszamy ich po cichu.

## Kierunek narzędzia i przestrzeń pozostałych cech

Domyślnie kierunek odróżniający narzędzie od tła jest dopasowywany wyłącznie na operacjach części A. Pozwala to uniknąć przecieku ze starego kierunku, którego zakres danych wymaga audytu. Porównanie ze znanym wektorem zapisujemy jako cosinus; sam ten odczyt nie zmienia kierunku użytego w ocenie.

Dla znormalizowanego kierunku w: a_p = z_p^T w; r_p = z_p - a_p w. To rozkład geometryczny, nie dowód pełnego oddzielenia semantyki. Dla maski narzędzia liczymy średnią sygnału, rozrzut, kontrast z tłem, średnie w pobliżu końcówki i trzonka oraz momenty przestrzenne. Mierzymy też średnie pozostałych cech wewnątrz maski. Pojedyncze maksimum nie jest kryterium rozpoznania.

## Uczciwy podział

Zewnętrzny podział grupowy po operacjach. W każdym przebiegu:

1. A: kierunek narzędzia, skalowanie, Koopman, model z pamięcią i jego nominalna kowariancja.
2. B: predyktor błędu detektora. Dostaje prognozy dynamiki z modelu A, nigdy dopasowanego na B.
3. C: ocena wszystkich wariantów. Całe operacje C są wyłączone z A i B.

Powtarzamy podział dla wcześniej ustalonych ziaren. Wyniki nie są niezależnymi replikacjami klinicznymi. Raportujemy rozrzut po operacjach i różnice sparowane, bootstrap po operacjach. Bez wybierania najlepszego ziarna.

## Drabina porównań

- U: ujemna pewność detektora.
- G: pewność i geometria z etykiet, identycznie dostępne wszystkim kolejnym wariantom.
- I: G oraz przestrzenny sygnał narzędzia.
- R: I oraz pozostałe cechy tokenów, prosta różnica czasowa i ta sama liczba opóźnień. To kontrola dostępu do historii.
- K: R oraz błąd prognozy EDMD z ustalonym słownikiem [s, s²] i rzeczywiste/urojone współrzędne modów.
- L: K oraz błąd prognozy z pamięcią, wielkość poprawki pamięciowej i zmiana prognozy po dołączeniu historii pozostałych cech tokenów.
- F: L oraz innowacja nominalnego filtru Kalmana, jej wariancja i wynik normalizacji.
- P: identyczne cechy F, ale dynamiczny dodatek przesunięty cyklicznie wewnątrz filmu. Zachowuje wartości, przebieg i kontekst filmu; przerywa zgodność czasową z błędem. Nie jest uniwersalną podłogą dla dowodu metastabilności.

Predyktory błędu to ten sam model logistyczny o ustalonej regularyzacji. Większa liczba kolumn może zmieniać pojemność; ocena poza operacjami i kontrola P ograniczają tę pułapkę, ale nie zastępują przyszłego badania wymiaru.

## Koopman, pamięć i Kalman

EDMD jest liniowym operatorem w ustalonym nieliniowym słowniku. Nie jest deep Koopman autoencoderem. Widmo raportujemy, nie utożsamiamy wartości własnej bliskiej jedynce z tożsamością.

Model z pamięcią to regresja z opóźnieniami stanu, skończone przybliżenie członu pamięciowego. Nie jest identyfikacją pełnego równania Mori–Zwanziga. Na wejściu nie ma pomiaru ruchu kamery ani sterowania ręką; reszty nie wolno przypisać wyłącznie brakowi wiedzy sieci. Pełne wymuszenia i człon zdarzeń są późniejszym rozszerzeniem, po udostępnieniu takich pomiarów.

Kalman: obserwowanym stanem jest przestrzenny sygnał narzędzia. Dynamika pochodzi z liniowego modelu jednego kroku, Q z jego reszt na A, R jest jawnym założeniem konfiguracyjnym. Nie identyfikujemy osobno Q i R z tych samych reszt. Q jest efektywną kowariancją modelu; P i S pozostają nominalne. Raportujemy ich kalibrację na C. Nie nazywamy ich niepewnością parametrów detektora.

Sekwencje powstają tylko przy dokładnie zadanym odstępie klatek, wewnątrz tego samego filmu i operacji. Przerwy zerują historię. Nie sklejamy rzadkiej puli w pozornie gęsty film. Wszystkie warianty oceniamy na tych samych klatkach z dostępną historią. Brak dostatecznej liczby takich klatek przerywa bieg.

## Kryteria przed uruchomieniem

Cel: błąd = IoU < 0,5 lub pewność < 0,5; osobno raportujemy sam błąd lokalizacji. Uwaga: pierwsza definicja częściowo zależy od pewności, dlatego lokalizacja jest obowiązkową kontrolą.

Miary: AUC, average precision, Brier dla modeli probabilistycznych, odsetek błędów w najwyżej ocenionej części każdego filmu. Różnice F-R i F-P z przedziałem bootstrapu po operacjach. Analiza wszystkich przypadków oraz opis odrębności przypadków z silnym sygnałem narzędzia. Bez dopasowania progu na C.

Sygnał wart kolejnego etapu: poprawa F-R i F-P na nowych operacjach, zgodny kierunek po ziarnach oraz także dla lokalizacji, nie tylko przy celu zawierającym confidence. Przedział obejmujący zero oznacza brak rozstrzygnięcia. Pogorszenie raportujemy wprost.

## Granica wnioskowania i następny etap

Eksport rankingów jest diagnostyką. Ponieważ etykiety są dostępne, ranking po prawdziwym błędzie jest prostym konkurentem dla późniejszego doboru danych. Obecny etap nie może go uczciwie „pokonać” we wskazywaniu tego samego błędu.

Dopiero przyszły trening sprawdzi, czy grupy o podobnej sygnaturze cząstek i dynamiki dają większy spadek błędu niż wybór według samego błędu i losowanie. Wyszukiwanie podobnych klatek z dopasowaniem pozy i interwencje w latent wymagają osobnego etapu; nie są ukryte w tym pilocie.
