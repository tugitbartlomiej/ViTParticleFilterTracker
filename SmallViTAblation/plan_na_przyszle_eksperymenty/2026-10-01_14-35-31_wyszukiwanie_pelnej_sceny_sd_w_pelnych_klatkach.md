# Plan poprawki: pełna scena SD względem pełnych klatek puli

Data utworzenia: 2026-10-01 14:35:31 (Europe/Warsaw).
Ostatnia modyfikacja: 2026-10-01 14:43:30 (Europe/Warsaw).
Status: zapis wymagania i planu porównania. Nie implementowano wyszukiwarki ani nie uruchamiano nowych testów.

## Obowiązujące wymaganie

Porównywać cały wygenerowany i zdekodowany obraz Stable Diffusion z całymi prawdziwymi obrazami dostępnej puli. Wybrać podobne pełne sceny jako nowe klatki treningowe. Wyszukiwanie po samym wycinku narzędzia nie spełnia tego wymagania w nowym ramieniu.

Podstawą wyboru mają być cechy całego obrazu: obraz do obrazu, całość do całości. Reprezentacja i reguła podobieństwa powinny zachowywać układ pełnej sceny oraz uwzględniać narzędzie i tło. Nie redukować decyzji do cech pojedynczego cropu narzędzia. Sama globalna średnia cech może zgubić układ; sprawdzić to w późniejszym porównaniu.

Konkretny model lub miara podobieństwa nie są narzucone. Nie musi to być DINOv2 ani odległość między latentami DINO. Wybór wymaga porównania jakości pełnoobrazowego wyszukiwania, efektu uczenia i kosztu. Obecne wyszukiwanie wycinkowe jest kontrolą eksperymentalną.

## Co już robi istniejący kod

- SD generuje i dekoduje pełną klatkę 576×384. Pełny obraz już istnieje przed wyszukiwaniem; nie trzeba go dopiero odtwarzać z cropu.
- Prowadzenie małym ViT i pomiar siły tożsamości korzystają z wycinka 176×176 wokół narzędzia (`prowadzenie.py`, `uklady.wycinek_176`).
- Zapytanie retrieval powstaje z wycinka narzędzia 256×256 (`uklady.wycinek_zapytania`), a `wybor.py` koduje przygotowany wycinek przez DINOv2. `petla4.py` zapisuje ten wycinek jako zapytanie.

Nowa poprawka dotyczy kroku retrieval: pełny obraz zapytania kontra pełny obraz kandydata. Nie stanowi automatycznie decyzji o prowadzeniu SD całym latentem sceny ani o prognozowaniu całego latentu. W pierwszym porównaniu sposób generowania i prowadzenia obrazu SD pozostaje wspólny dla metod wyszukiwania. Zmiana prowadzenia wymaga osobnego pytania i ablacji.

## Etap 1. Kontrakt obrazu i audyt indeksu

- [ ] Ustalić, gdzie zachowywany jest końcowy pełny obraz RGB po dekodowaniu SD, oraz połączyć go z identyfikatorem kotwicy, seedem, obrotem, modelem i ustawieniami generacji.
- [ ] Dla kandydatów korzystać z całych źródłowych klatek puli, z jednoznacznym frame_id i rozmiarem natywnym. Dotychczasowy indeks wycinków nie może zostać nazwany indeksem pełnych obrazów.
- [ ] Zapisać wspólny kontrakt kolorów RGB, zakresu wartości, proporcji i rozdzielczości. Wyrównanie rozmiaru musi zachować całe pole widzenia; ewentualny letterbox ma jawne maskowanie dopełnienia w metryce. Nie używać ukrytego center-crop w preprocessing modelu percepcyjnego lub encodera.
- [ ] Sprawdzić na zapisanym zapytaniu i kandydacie, że w każdym nowym ramieniu dostępna jest cała scena, włącznie z tłem i położeniem narzędzia. Pełny obraz może być przeskalowany, ale nie zastąpiony wycinkiem.
- [ ] Ustalić wspólną dostępną pulę, istniejące wykluczenia, limit nowych klatek i sposób postępowania przy wyczerpaniu listy. Zapisane przyszłe klatki z odłożonego testu nie stają się pulą kandydatów.

## Etap 2. Porównywane rodziny podobieństwa

1. Bezpośrednie metryki pełnych obrazów po wspólnym wyrównaniu rozmiaru i kolorów: MAE, MSE i SSIM. To orientacyjne kontrole, nie gwarantowane miary semantycznej zgodności sceny. Małe przesunięcie narzędzia, refleksy, dominujące tło i różnica ekspozycji mogą zmienić ranking bardziej niż właściwa treść. Korekcja koloru lub rejestracja są osobnymi, jawnie opisanymi wariantami, z kosztem obliczeń.
2. Pełnoobrazowe metryki percepcyjne, na przykład odległość z ustalonego, zamrożonego modelu. Zapisać wersję, sposób normalizacji, zakres obrazu i poziom agregacji. Nie ograniczać porównania do maski narzędzia; jeśli model używa kafli, muszą łącznie obejmować całą scenę według stałej reguły.
3. Reprezentacje pełnych klatek, z podobieństwem globalnym lub ustalonym porównaniem tokenów. DINOv2 może być jednym ramieniem, ale nie obowiązkowym standardem. Nowy pełnoobrazowy indeks porównywać z innymi reprezentacjami dostępnymi w uzasadnionym budżecie. Globalne uśrednienie może zgubić layout, a dopasowanie tokenów zwiększa koszt: obie własności raportować.
4. Opcjonalna geometria/layout całej sceny: pozycja, skala i orientacja narzędzia oraz układ istotnych obszarów. Jawnie opisać źródło geometrii, błędy jej estymacji i koszt. Geometria jako pomocniczy składnik nie zastępuje pełnoobrazowego porównania samym cropem. Nie używać anotacji odłożonego testu do wyboru kandydatów.
5. Dotychczasowy retrieval DINOv2 po cropie256 jako kontrola, z tym samym obrazem SD przed wycięciem. Dodatkowo losowy dobór z tej samej dozwolonej puli jako kontrola wpływu nowych danych na uczenie.

Ewentualne wagi mieszanki metryk oraz liczba kandydatów do dokładnego rerankingu są parametrami ustalanymi na części rozwojowej. Ranking nie może po cichu stosować cropu w pierwszym filtrze, który wyklucza kandydatów dla nowego ramienia full-image. Jeżeli użyje się szybkiego filtra, także on musi opierać się na pełnych klatkach; trzeba zmierzyć utratę trafień względem porównania dokładnego.

### Kandydat wskazany przez użytkownika: DETR-E

Według relacji użytkownika jego wcześniejsza metoda DETR-E korzystała z DINO, SAM i klasteryzacji do wyszukiwania różnic. Jest kandydatem do adaptacji na wyszukiwanie podobieństw między całymi obrazami. To zapis relacji użytkownika, a nie zweryfikowany opis implementacji. Nie mamy jeszcze ścieżki do definicji, kodu ani tekstu metody; nie przyjmować nieznanej architektury lub funkcji celu jako faktu.

W późniejszym etapie ustalić źródło z użytkownikiem i przeczytać definicję, kod lub tekst DETR-E. Sprawdzić, jakie cechy i relacje tworzyła metoda, co klasteryzowała, jak mierzyła różnice i które składniki zachowywały układ sceny. Dopiero wtedy określić adaptację do podobieństwa; nie zakładać, że wystarczy odwrócić znak dotychczasowego wyniku. Ten etap nie uruchamia teraz wyszukiwania źródeł.

SAM i segmenty mogą wnosić cechy części obrazu, ich położenia, wielkości i relacji do reprezentacji pełnej sceny. Agregacja ma zachować kontekst całego obrazu oraz narzędzie i tło. Dopasowanie wyłącznie jednego segmentu narzędzia lub jego cropu nie spełnia wymagania full-image retrieval.

### Opcjonalne ramię: pełnoobrazowe DINO i SSIM

Porównać DINO kodujące całą klatkę oraz SSIM liczone na całych wyrównanych obrazach. Możliwe warianty: kandydaci z pełnoobrazowego DINO i reranking przez pełnoobrazowe SSIM albo ranking z dwoma jawnie zapisanymi kryteriami. Oba etapy mają działać na całych obrazach. Ustalić liczbę kandydatów, regułę łączenia i normalizację na części rozwojowej, zachowując oryginalne wyniki obu miar.

Nie gwarantować, że połączenie poprawi ranking lub uczenie. Odległość DINO i SSIM są wynikami konkretnych miar, nie procentem zgodności sceny ani prawdopodobieństwem trafnego dopasowania. Oceniać je niezależną oceną pełnej sceny, efektem uczenia i kosztem, z uwzględnieniem wrażliwości na przesunięcia, oświetlenie i odbicia.

## Etap 3. Uczciwe porównanie i wybór

- [ ] Na początku użyć tych samych zachowanych pełnych obrazów SD jako zapytań dla wszystkich metod. To oddziela jakość retrieval od losowych różnic generacji.
- [ ] W każdym ramieniu wybierać tyle samo nowych klatek: w odtworzeniu kampanii 134 na obrót; w pilotażu z góry ustalony wspólny limit. Zachować te same wykluczenia już użytych danych, filmów kontrolnych/testowych oraz regułę odległości czasowej ±30 klatek. Stosować te same reguły duplikatów i uzupełniania braków.
- [ ] Część rozwojową i końcową odłożyć po filmach lub scenach, nie losowo po sąsiednich klatkach. Przed końcową oceną zamrozić preprocessing, metrykę, model, wagi mieszanki i regułę wyboru. Nie dobierać metryki na końcowej walidacji detektora.
- [ ] W ocenie offline zachować całe rankingi i galerie pełnych obrazów zapytanie-kandydaci. Oceniać zgodność całej sceny i layoutu, także przez ocenę wizualną z ukrytą nazwą metody; kryteria i sposób rozstrzygania remisów zapisać przed oceną. Nie definiować sukcesu wyłącznie przez własną odległość używaną w rankingu.
- [ ] Raportować również pokrycie różnych filmów, liczbę bliskich duplikatów i to, czy ranking wybiera wspólne tło kosztem narzędzia. Pomocnicza ocena narzędzia nie zastępuje oceny całej sceny.

## Etap 4. Efekt uczenia i koszt

Po audycie full-image oraz pilotażu offline przygotować osobny przyszły eksperyment uczenia. Wspólny punkt startu sieci, liczba aktualizacji, budżet klatek, przepis losowania i ewaluacji. Najpierw wspólne zapisane zapytania dla porównania retrieval; pełną pętlę z nowymi generacjami badać oddzielnie, bo wybór danych zmienia kolejne stany sieci i zapytania.

Raportować mAP i pozostałe wcześniej ustalone miary detektora na odłożonych filmach, zmiany względem crop retrieval i losowego doboru oraz rozrzut pomiędzy ziarnami. Ziarna są jednostką powtórzeń; klatki jednego filmu nie są niezależnymi eksperymentami. Lepsze podobieństwo wizualne nie gwarantuje lepszego uczenia.

Koszt obejmuje przygotowanie indeksu pełnych klatek, jego rozmiar na dysku, pamięć CPU/GPU, czas kodowania zapytania, czas wyszukania i rerankingu oraz amortyzację dla kolejnych obrotów. Raportować jakość względem kosztu; nie przyjmować, że jedna najdroższa miara jest automatycznie najlepsza.

## Warunek sukcesu i falsyfikacja

Wymaganie techniczne spełnia tylko porównanie full-image vs full-image. Sukces jakościowy wymaga powtarzalnej poprawy zgodności całej sceny i użyteczności wybranych klatek przy tym samym budżecie danych i akceptowalnym, jawnie zmierzonym koszcie. Próg minimalnej użytecznej poprawy ustalić przed kampanią, na podstawie pilotażu i budżetu.

Brak poprawy detektora, wybór głównie podobnego tła, wysoka wrażliwość na ekspozycję/przesunięcie lub koszt bez przyrostu jakości obalają przewagę danej metody. Ujemny wynik nie uchyla wymagania pełnoobrazowego: oznacza potrzebę sprawdzenia innej miary, a nie powrót do cropu pod nazwą pełnej sceny.

## Przyszłe artefakty i źródła kodu

Przyszła realizacja ma własny katalog eksperymentu: PLAN.md, README.md, kod preprocessing/rankingu/oceny, konfigurację zamrożonej miary, manifest pełnych zapytań i klatek, rankingi z metadanymi, raport jakości wizualnej, koszt, wyniki uczenia i WYNIKI.md. Ten zapis nie zakłada jeszcze takiego katalogu ani nie uruchamia obliczeń.

Źródła stanu obecnego:

- `../experiments/maly_vit_petla_sd_rownanie/kod/prowadzenie.py`
- `../experiments/maly_vit_petla_sd_rownanie/kod/uklady.py`
- `../experiments/maly_vit_petla_sd_rownanie/kod/petla4.py`
- `../experiments/maly_vit_petla_sd_rownanie/kod/wybor.py`

Plan Langevina pozostaje osobnym dokumentem; ten plan nie zmienia celu prognozy reprezentacji.
