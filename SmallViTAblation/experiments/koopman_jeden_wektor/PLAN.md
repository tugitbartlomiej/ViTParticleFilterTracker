# Czy jeden pełny wektor tokenu jest przewidywalny?

Data: 2026-09-25. Osobna analiza diagnostyczna na istniejących danych, bez treningu RF-DETR/FastGAN i bez PCA.

## Pytanie i obalenie

Czy jawny, liniowy model jednego pełnego tokenu (384 współrzędne) potrafi prognozować jego zmianę lepiej niż brak zmian oraz powtórzenie ostatniego przyrostu? Negatywny wynik na późniejszych checkpointach i innych grupach obala skuteczność wybranego wariantu w tej próbie. Nie gwarantujemy znalezienia zwycięzcy.

## Dane i wybór przed oceną

- Źródło: sąsiedni koopman_gesta_historia/wyniki/macierze_wiele_obrazow, 73 stany 72 obrazów od kroku 5 do 365 co 5 aktualizacji. Parujemy tylko ten sam obraz.
- Osiem kandydatów to pozycje tokenów o największym średnim pokryciu maską na 48 obrazach fit. Maska służy do wyboru miejsc, nie uśrednia reprezentacji. Każdy kandydat zachowuje wszystkie 384 liczby. To test tokenu, nie zamiennik wcześniejszego pełnego operatora obrazu.
- Wcześniejsze grupy fit dzielimy deterministycznie na 6 grup dopasowania i 2 walidacji. Wybór tokenu, modelu i ridge używa tylko początków 105,145,185 i ich następnych checkpointów; maksymalny cel to krok 190.
- Okno dopasowania: ostatnich 8 znanych odcinków, 288 par wewnętrznych; po wyborze 384 par ze wszystkich grup fit. Nie korzystamy z przyszłych celów.
- Końcowa ocena: wszystkie początki 210..360 co 5 aktualizacji, 24 obrazy z czterech innych grup. Wybór zamrażamy przed odczytem ich błędów. Historyczne metryki tych obrazów były już oglądane, więc to nie jest nowy niezależny test końcowy.

## Modele i podłogi

1. Afiniczny operator przyrostowy na 384 liczbach.
2. Afiniczny operator z pełnym poprzednim przyrostem: 768 liczb, bez kompresji.
3. Afiniczny operator prognozujący następny przyrost z poprzedniego, potem dodający go do stanu.
4. Strukturalny model v_next=alpha*v (jedno alpha na cały token).
5. Strukturalny model v_next=diag(alpha)*v (384 współczynniki, bez mieszania kanałów; to założenie o dynamice, nie PCA).

Dla modeli 1–3 ridge względny {0.001,0.1}; dla modelu 5 kara względem alpha=1 wynosi 0.001 średniej energii wejścia. Model 4 jest najmniejszym kwadratem bez wyrazu wolnego. Nie obcinamy współczynników po ocenie.

Wybór: najmniejszy łączny walidacyjny MSE podzielony przez MSE ostatniego przyrostu dla tego samego tokenu; remisy deterministyczne. Podłogi: z_next=z oraz z_next=z+v. Wszystkie warianty mają dostęp wyłącznie do przeszłości; prostsze modele mogą używać jej podzbioru. Pełna macierz jest zapisywana w konwencji wektorów wierszowych: s_next=s*A+b.

## Kryteria ustalone przed wynikiem

Główna ocena dotyczy jednego następnego checkpointu, czyli 5 aktualizacji. Sukces pilota wymaga jednocześnie: RMSE <=0.95 RMSE ostatniego przyrostu, RMSE <=0.8 RMSE braku zmian, średni cosinus prognozowanego i rzeczywistego przyrostu >=0.5 oraz przewaga nad ostatnim przyrostem w co najmniej 3 z 4 grup. Osobno raportujemy błąd względem wielkości rzeczywistej zmiany, błąd na obrazach z maską obejmującą token oraz wszystkie grupy. Wybór nie opiera się na końcowych błędach.

Krótkie prognozy nie dowodzą kompletności tożsamości ani liniowości globalnej. Token na narzędziu może zawierać także kontekst. Lokalne aktualizowanie macierzy nie jest jednym stałym operatorem całego treningu. Dodatkowo dla początków 210,230,250 zamrażamy wybrany lokalny operator i wykonujemy rollout 20 checkpointów bez podstawiania prawdziwego przyszłego stanu; to diagnostyka długiej prognozy, nie kryterium wyboru.

## Zasoby i odtwarzalność

CPU, cztery wątki BLAS; odczyt ok.4.6 GB zapisanych tensorów i cache ośmiu tokenów ok.130 MB. Macierze maksymalnie 768×768, bez nowego inference. Zapisać protokół, SHA256 źródeł, wybrany wariant przed oceną, błędy per obraz i checkpoint, macierze, wykres i wynik negatywny, jeśli wystąpi. Małe testy muszą potwierdzić zgodność zapisanej macierzy z prognozą oraz brak przyszłości w oknie.

Podstawy: [Tu i in.](https://doi.org/10.3934/jcd.2014.1.391) oraz [poprzedni audyt sześciu prac](../koopman_gesta_historia/AUDYT_LITERATURY_6_KOOPMAN.md). Nie zakładamy, że niedostateczny rząd oznacza błąd solvera lub że mały residual potwierdza poprawne rozpoznawanie.
