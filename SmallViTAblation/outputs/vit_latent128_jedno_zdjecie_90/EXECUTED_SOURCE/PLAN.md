# Jedno zdjęcie, pełny latent i wyróżnione J128

Zlecenie 2026-10-05: wybrać jeden obraz z puli400, sztucznie zmienić latent, uzyskać wykrycie powyżej90%, zapisać obraz i pełne wartości latentu z wyróżnieniem128 wybranych kanałów.

## Kryterium i podłoga

W jednym obrazie mierzymy osobno confidence najpewniejszej ramki i jej IoU względem etykiety. Warunek sukcesu: oba >0,90. Docelowe zatrzymanie: confidence>=0,93 i IoU>=0,95. Nie jest to 90% AP na nowych filmach ani kalibracja pewności. Podłoga to ten sam obraz, checkpoint k1000 i dekoder bez interwencji. Brak spełnienia warunku obala sukces tej próby.

## Dobór i dostęp do odpowiedzi

Wybór z original train historycznej puli400, spośród obrazów z wysokością ramki >=24px i szerokością >=32px, na podstawie wcześniejszego treningowego oracle. Preferujemy najwyższe minimum confidence i IoU. Jest to jawny wybór korzystnego pojedynczego przykładu i optymalizacja z jego GT, a nie test generalizacji.

Pierwszy wybór według największego przyrostu IoU nie spełnił warunku confidence: po6000 kroków confidence0,769693, IoU0,999754. Zapisany w NIEUDANE_PODEJSCIE wraz z pierwotnym kodem i latentem. Po obejrzeniu tego wyniku zmieniono kryterium wyboru na minimum dwóch wymaganych miar i obniżono opcjonalne zatrzymanie z confidence0,95 do0,93; wymagany sukces0,90 pozostał ten sam. Jest to adaptacyjne wyszukiwanie przykładu.

## Interwencja i zapis

Wagi/bufory ViT zamrożone. Zmieniamy surowe aktywacje600×128 po GELU ostatniej MLP, projekcja na minimum dokładnego GELU -0,16997120. Pozostałe600×640 wartości pozostają identyczne. Start od wcześniejszego teacher train. Cel: dodatni logit prawidłowego środka, tłumienie innych lokalizacji, dopasowanie log-odległości ramki i GIoU. Adam lr0,05, maksymalnie2000 kroków, pomiar co25; w razie braku sukcesu dwa kolejne ustalone restarty lr0,15 i0,3. Bez dobierania z nowych filmów.

Zapis pełnych macierzy600×768 przed/po/różnicy w NPZ i CSV, macierzy600×128, indeksów osi, statystyk128 kanałów, wartości każdego tokenu w128 kanałach, siatki20×30, zdjęcia źródłowego i faktycznego wejścia480×320, wizualizacji ramek przed/po, checkpointu i wyniku. CSV pełnego latentu wyróżnia wybrane kolumny prefiksem J128. Obraz zostaje ten sam; wynik po interwencji wymaga wstawienia zapisanego latentu do sieci.

## Kontrole

Tail zgadza się z pełnym forwardem; zapisany latent odczytany z dysku daje ten sam wynik w pełnym forwardzie. Hash wag/buforów bez zmiany. Pozostałe640 kanałów bitowo identyczne. Wybrane aktywacje powyżej minimum GELU. To warunek zakresu, nie dowód osiągalności naturalnego zdjęcia. Nie zmieniamy pikseli ani etykiety zdjęcia.
