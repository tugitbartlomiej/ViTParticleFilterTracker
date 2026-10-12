# Szukanie podprzestrzeni tożsamości narzędzia w latencie

Data: 2026-10-01, 16:46:21 (Europe/Warsaw).
Status: metoda zaakceptowana przez użytkownika, plan do realizacji. Dane źródłowe 400 klatek są przygotowywane osobno; nie uzyskano jeszcze wyników identyfikacji.

## Powiązanie z wcześniejszymi planami

Ten plan rozwija wcześniejsze zapisy o wielowymiarowej tożsamości w planie poprawy równań i planie ruchu cząsteczek. Stanowi osobny plan identyfikacji, nie dubluje badania prognostycznego. Powiązane pliki: 2026-10-01_14-06-21_poprawa_rownan_langevina_i_koopmana.md i 2026-10-01_16-28-13_ruch_czasteczek_a_uczenie_detekcji_na_stalych_klatkach.md.

## Hipoteza i definicje

Poszukiwać kilku wspólnych kierunków, czyli podprzestrzeni związanej z narzędziem, a nie narzucać jednej osi LDA ani jednej osi bazowej. Cząsteczka to pełny wektor tokenu z_i w R^768. Makrowektory e_j definiują wspólną bazę; mikrowektory to z_ij e_j. Kierunek konceptu może być kombinacją wielu makrowektorów. 600 cząsteczek dotyczy używanej architektury i obrazu480×320.

Operacyjnie najpierw badamy obecność konkretnego narzędzia. Tożsamość odróżniająca je od innych instrumentów wymaga dodatkowego porównania z innymi narzędziami. Reakcja na metal, krawędzie lub obecność dowolnego narzędzia nie wystarcza.

## Dane: 400 par i kontrola edycji

400 unikalnych naturalnych klatek train z prawdziwą maską narzędzia, zróżnicowanymi kątami i filmami. Każdej odpowiada jedna syntetyczna wersja z usuniętym narzędziem: docelowo800obrazów. Inpainting lokalnym modelem jest osobnym etapem; proponowany LaMa, wybór/pobranie/uruchomienie dokumentować jawnie. Nie twierdzić, że wersja bez narzędzia ujawnia rzeczywiste tło spod niego.

Kąt wyznaczyć z maski, z jawną konwencją osi modulo180 i niepewnością. To naturalne orientacje, nie kontrolowana rotacja na identycznym tle. Zachować wszystkie pixele poza zadeklarowaną maską edycji; ewentualną dylatację i blending zapisać osobno od prawdziwej maski.

Dla każdej cząsteczki zapisać pokrycie patcha maską: liczba pikseli narzędzia / liczba pikseli patcha. W obu obrazach pary analizować te same indeksy tokenów wskazane ORYGINALNĄ maską. Zapis pełnych wektorów; brak wymuszonego poolingu i redukcji do jednego skalaru.

## Pomiar i identyfikacja

Zamrożony ViT, tryb eval, stały checkpoint i preprocessing. Delta_i = z_i(obraz_z_narzedziem) - z_i(obraz_bez_narzedzia). Ten pomiar jest odpowiedzią na zmianę wejścia, nie ruchem podczas uczenia. Porównanie tego samego obrazu przez kolejne checkpointy stanowi drugi, oddzielny wymiar.

Zachować wszystkie delta_i i ich pokrycia. Dopasować niewielką podprzestrzeń ze spójnego sygnału między parami, z regularyzacją i wagami pokrycia; w analizie kontrolować liczbę tokenów oraz udział poszczególnych obrazów/filmów. Nie wybierać kierunków wyłącznie po największej normie ruchu. PCA/SVD różnic to baza kandydatów, nie dowód semantyki. Porównać także metodę dyskryminacyjną z kontrolami tła, artefaktów i innych instrumentów. Liczbę kierunków i hiperparametry wybrać na walidacji.

Oceniać odróżnianie obecności narzędzia oraz stabilność podprzestrzeni na odłożonych filmach i kątach. Podział par po filmach; obie wersje jednej pary pozostają w tej samej części danych. Czterysta klatek train nie jest niezależnym testem generalizacji detektora. Niezależne filmy do ostatecznej oceny dobrać osobno; bieżące dane służą eksploracji/dopasowaniu.

## Kontrole i test wpływu

- Tło, okolica narzędzia, różne progi pokrycia tokenu, jasność/kontrast i orientacja.
- Sham edits: porównywalne maski i uzupełnienie tła bez usuwania właściwego narzędzia, aby oszacować sygnał artefaktów. W razie potrzeby drugi model inpaintingu i transfer między metodami edycji.
- Inne narzędzia dla rozróżnienia tożsamości od samej obecności instrumentu; brak takich danych oznacza ograniczenie interpretacji.
- Osłabić/usunąć znalezioną podprzestrzeń w mierzonej warstwie, następnie rzeczywiście dokończyć forward do detektora. Sprawdzić stratę, ramki/IoU, pewność oraz AP/mAP na odpowiednich zbiorach.
- Porównać z losową podprzestrzenią o tej samej liczbie kierunków i dopasowanej wielkości perturbacji, a także z kontrolą zachowania reszty aktywacji. Kontrolować rozkład i nienaturalność ingerencji.
- Pogorszenie po ablacji dowodzi wrażliwości na interwencję; samo nie rozstrzyga, że kierunki są wyłącznie semantyczną tożsamością. Ocena zależy od pozostałych kontroli.

## Metryki i warunek powodzenia

Rozdzielenie obecności narzędzia poza dopasowaniem, odporność na kąt/tło, powtarzalność kierunków/subprzestrzeni oraz dodatkowy spadek jakości detekcji względem właściwych kontroli perturbacji. Raportować błędy, rozrzut i negatywy. Filmy/obrazy są jednostkami oceny, nie wszystkie tokeny traktowane jako niezależne próbki. Brak z góry deklarowanej skuteczności ani oryginalności.

## Artefakty i etapy

1. Manifest400źródeł, maski, kąty, pokrycia tokenów i sumsSHA256.
2. Pilot jakości inpaintingu przed stworzeniem wszystkich400syntetyków; oryginały i edycje osobno, status każdej pary jawny.
3. Pełne tokeny par i mały zbiór kontroli, konfiguracja checkpointu oraz warstwy.
4. Kandydaci podprzestrzeni, ocena na odłożonych danych i test ingerencji.
5. Raport: co wykazano, czego nie wykazano i porównanie z literaturą. Oddzielnie kolejny etap prognozowania ruchu podczas uczenia.

Research zlecono Astrze przez Undermind/DeepSearch: biologiczne conceptcells/populationcoding oraz AI CAV/TCAV,NetworkDissection,pairedcounterfactualdiscovery i interwencje. Zbieżność z nimi wymaga porównania konkretnych procedur, nazwa cząsteczka nie stanowi dowodu nowości.
