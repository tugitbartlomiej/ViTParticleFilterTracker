# Kontrola wpływu edycji LaMa przy pozostawionym narzędziu

2026-10-01. Wygenerowano 24 z 24 kontroli w 65,8 sekundy. To pary kontrolne istniejących obrazów, nie 24 dodatkowe niezależne obserwacje. Wybór dwóch indeksów z każdego z 12 przedziałów orientacji był deterministyczny i nie używał wyników ViT.

Każda kontrola przesuwa rzeczywistą maskę edycji odpowiadającej pary narzędzie–usunięte. Zachowuje kształt i liczbę pikseli bez obcięcia. Przesunięcie wynosi co najmniej 48 pikseli; maska nie przecina oryginalnej maski narzędzia poszerzonej o 10 pikseli. Ten sam model LaMa i preprocessing uzupełniają przesunięty obszar. Wynik jest wklejony wyłącznie wewnątrz tej maski.

Skrypt dla wszystkich 24 obrazów sprawdził brak zmian poza maską edycji oraz brak zmian pikseli anotowanego narzędzia. SHA256 obrazów i masek, przesunięcia, udział edycji w 600 tokenach i ścieżki zapisano w `MANIFEST.json`. Schemat pliku: obiekt zawierający listę `records`; analizować rekordy ze `status: generated`.

Przejrzano wizualnie osiem pierwszych trójek z `SHAM_PREVIEW.jpg`. Nie jest to pełny audyt wszystkich 24 edycji. Wykluczenie dotyczy wyłącznie anotowanego cystotome: maska kontrolna może zawierać inny przyrząd, inną tkankę albo odbłysk. Dlatego kontrola mierzy również różnicę treści i położenia edytowanego obszaru.

Po ustaleniu podprzestrzeni na danych treningowych należy porównać dwie odpowiedzi: na tokenach niezmienionego narzędzia oraz na tokenach faktycznie edytowanego obszaru kontrolnego. Pierwsza mierzy wpływ kontekstu i atencji, druga bezpośrednią reakcję na edycję. Wyniki podawać także według roli filmu w podziale train/validation/test. Kontrole nie mogą służyć do dostrajania na części testowej.

Ta kontrola nie dowodzi, że syntetyczny obraz jest prawdziwym tłem, i sama nie rozdziela obecności narzędzia od wszystkich artefaktów jego usunięcia. Zapewnia sprawdzenie, czy samo zastosowanie LaMa w innym miejscu wywołuje porównywalny sygnał.

## Uzupełnienie wyłącznie na filmach testowych

Pierwsze 24 obrazy należały do części train/validation/test w liczbach 18/3/3. Dlatego przygotowano osobną kontrolę `heldout/MANIFEST.json`: 24 obrazy z 17 filmów części testowej, po dwa z każdego z 12 przedziałów orientacji. Skrypt uruchamia się z flagą `--test-only`; wybór to te same deterministyczne pozycje jednej i dwóch trzecich listy, już po odfiltrowaniu części testowej. Nie korzysta z trafności klasyfikatora, norm latentów ani wyników interwencji.

Uzupełnienie ukończono 2026-10-01 w 69,7 sekundy. Wszystkie 24 maski mają prawidłowe przesunięcie i zachowane pole; wszystkie kontrole zgodności pikseli poza edycją i w masce narzędzia przeszły. Przejrzano osiem pierwszych trójek z `heldout/SHAM_PREVIEW.jpg`, z tymi samymi ograniczeniami interpretacji co wyżej.

Podprzestrzeń i odczyt pozostają zamrożone. Ten zestaw służy do testu, nie do ponownego wyboru kierunków lub parametrów. Raportować oddzielnie od pierwszych 24 i nie agregować powtarzających się `source_index` jako nowych niezależnych obrazów.
