# Wartości przypisane 768 osiom latentu

Zapisano podpisane wykresy PRZED, PO i RÓŻNICA oraz wspólne zestawienie. Źródłem jest oryginalny latent jednego zdjęcia, przed i po sztucznej korekcie J128 przy zamrożonych wagach. Nie jest to latent po treningu Lorentza ani latent obrazu z SD.

Makrowektory e_j są stałymi osiami bazowymi: ich kierunki i jednostkowe normy nie zmieniają się. Zmieniają się współrzędne cząsteczek z_j. Aby pokazać jedną wartość na oś, wykres agreguje wszystkie 600 cząsteczek: średnia zachowuje znak, RMS opisuje wielkość aktywacji i ogranicza efekt znoszenia dodatnich i ujemnych wartości. Są to statystyki współrzędnych, nie siły paczek/gradientu. Każda cząsteczka nadal ma własną wartość.

Pomarańczowy wyróżnia wybrane 128 kanałów. Pozostałe 640 współrzędnych nie zmieniło się dla żadnej cząsteczki. Numery na obrazach są od1 do768; CSV podaje również natywne indeksy od0 do767. Pełne liczby w CSV, kafelki zaokrąglają do2 miejsc.

Atlas32x24 jest indeksem kanałów, nie mapą przestrzenną. Stany PRZED i PO mają tę samą skalę kolorów. RÓŻNICA ma osobną skalę symetryczną wokół0. Nie zastosowano PCA ani odejmowania środka chmury. Różnica oznacza po-przed i nie jest dowodem poprawy na innych obrazach.

Istniejący PDF prezentacji pozostawiono bez zmian; nowe materiały są w tym folderze.
