# Audyt hipotezy szkła w pełnych 768 wymiarach

2026-10-10. Analiza na jawne polecenie użytkownika. Historyczne dane tylko do odczytu.

1. Gęsty bieg: wszystkie 8 zapisanych obrazów, 600 tokenów, 768 współrzędnych; kroki 0–1094 co 1, później co 50 do 22950. Rozgrzewka kończy się w 300. Luka 1094→1100 nie jest traktowana jako równy odstęp.
2. Osobny bieg 60 epok: ponowna inferencja tych samych ośmiu obrazów po istniejących checkpointach, bez uczenia. Różne biegi nie są sklejane.
3. Zawracanie: autokorelacja iloczynu skalarnego przyrostów, jawny odstęp oraz lag; porównanie z cosinusem kierunków. MSD: dwuczasowe odległości i średnie po równych oknach. Starzenie: wiek startu vs przyrost dla stałego opóźnienia, zmiana amplitudy, znormalizowany kształt i czas przekroczenia wspólnego progu. Brak przekroczenia jest cenzurowaniem, nie nieskończonym czasem.
4. Plateau: lokalne nachylenie log MSD względem log opóźnienia, czułość na okno. Operacyjna mocna przesłanka: nachylenie <=0.2 przez przynajmniej dekadę opóźnień, potem ponowny wzrost. Próg jest jawną konwencją audytu, nie uniwersalnym prawem szkła. Samo nasycenie OU nie rozstrzyga.
5. Układy odniesienia: surowy, po odjęciu centroidu obrazu oraz względem 6 sąsiadów w pełnym 768D ustalonych w kroku 300. Sąsiedzi są podobieństwem reprezentacji, nie potwierdzonym fizycznym oddziaływaniem.
6. Kontrole: poprawność siatki czasu, zgodność macierzy Grama z bezpośrednimi odległościami float64, precyzja inferencji, stałe statystyki BatchNorm dla wybranych punktów, kalibracja prostym OU. Wyniki per obraz; 8 obrazów pochodzi z 7 filmów i jednego ziarna treningu. Przedziały bootstrapowe nie zastępują powtórzeń treningu.
7. Raport i wykresy w outputs/audyt_szkla_768d_20261010. Ostateczny werdykt rozróżnia zgodność kinematyczną, dowód mechanizmu i brak rozstrzygnięcia. Długość treningu i mAP nie są same w sobie dowodem szkła.

Skille: statistical-analysis, data-validation, data-visualization. Literatura: Undermind, pełne teksty Winter/Janssen, Baity-Jesi et al., Herrera Segura et al.
