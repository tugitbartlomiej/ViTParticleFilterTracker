# Lokalna próba FastGAN

## Przeniesienie na Eden, 2026-09-23

Użytkownik zlecił trening na Edenie i pobranie modelu po ukończeniu. To kontynuacja tego samego biegu do 50 000 kroków, na tych samych 1044 wycinkach, batch 2 i niezmienionej funkcji straty. Lokalny proces zakończono przez plik STOP po kroku 16427; checkpoint zawiera generator, dyskryminator, EMA, optymalizatory, kolejność danych i stany losowania. SHA256 checkpointu: `01df623b1b7886b5e2dbb31eeb889f78f0dae201722d6091c807b22939cb0078`.

Eden: jedna karta H100 lub A100, jedno zadanie w partycjach hopper,short, cztery CPU, 24 GB RAM, limit sześciu godzin. Katalog współdzielony: `~/smallvit_runs/2026-09-23_fastgan`. Dane i identyczne bajtowo skrypty treningowe są rozpakowywane do tymczasowego katalogu w pamięci węzła; trap usuwa wyłącznie ten katalog. Checkpointy i raporty od razu trafiają na dysk współdzielony. Paczka danych, checkpoint i wagi VGG są sprawdzane SHA256 przed treningiem.

Nie zmieniamy istniejących środowisk RF-DETR. LPIPS jest zainstalowany bez zależności w prywatnym podkatalogu tego eksperymentu. Istniejący torch na Edenie ma wersję 2.9.1+cu128, wobec lokalnej 2.6.0+cu124. Sprzęt i biblioteki zmieniają się, więc kontynuacja nie jest deklarowana jako identyczna bitowo z pozostaniem na RTX. Stan uczenia i dane pozostają zachowane; poprawność wznowienia sprawdzamy na pierwszych krokach zadania.

`pakuj_wynik.py` pakuje tylko ukończony bieg: końcowy checkpoint, raporty, podglądy i opis środowiska, wraz z sumami plików. `pobierz_eden.py` pobiera go do `wyniki/eden_20260923_50k`, sprawdza sumę archiwum i każdego pliku oraz zapisuje `POBRANIE_OK.json`. Automatyzacja tej rozmowy sprawdza zakończenie i wywołuje pobieranie; nie uruchamia kolejnych treningów.

## Zatwierdzony dłuższy trening, 2026-09-23

Po udanym pilocie użytkownik napisał: „ok to mozesz puscic na mojej karcie”. Budżet dłuższego biegu: 50 000 kroków, batch 2, 1044 istniejące wycinki treningowe, czyli około 95,8 przejścia przez zbiór. Start od losowych wag. Szacunek z pilota: około 3,5 godziny plus zapis plików. Katalog: `wyniki/trening_20260923_50k`.

Skrypt `trening_lokalny.py` zachowuje architekturę, straty i augmentację pilota. Dodaje uśrednianie wag generatora EMA ze współczynnikiem 0,999 dla podglądów, zapis wag i optymalizatorów co 5000 kroków oraz podglądy co 1000 kroków. Checkpoint obejmuje też kolejność danych i stan generatorów losowych. Test zapisu i wznowienia przeszedł w katalogach `checkpoint_smoke_v2_20260923` i `resume_smoke_v2_20260923`. Wcześniejszy test wykrył niezgodny typ wersji torch w checkpointach; poprawiono go na zwykły tekst.

Stan biegu jest w `STATUS.json`, a wynik końcowy w `REPORT.json`. Plik `STOP` utworzony w katalogu biegu kończy pracę po bieżącym kroku i zapisuje checkpoint. Wznowienie wymaga `--resume` wskazującego checkpoint oraz nowego katalogu `--out`; `--steps` oznacza łączny budżet. Kod i manifest danych są identyfikowane sumami SHA256, kopia wykonywanego kodu jest w katalogu wyników.

Po treningu należy ocenić realizm obrazów oraz zgodność wygenerowanych masek z narzędziem. Sam przebieg nie testuje jeszcze doboru klatek ani skuteczności RF-DETR.

2026-09-23. Użytkownik zatwierdził uruchomienie lokalne: „puszczaj na lokalnym”. Wcześniejszy zakaz obliczeń nie obejmuje tej zatwierdzonej próby. Nie ma zgody na samodzielne wysyłanie kampanii na Eden.

## Cel i budżet

Próba techniczna, 200 kroków generatora i dyskryminatora na RTX 3070, batch 2, kwadrat 256 pikseli. Celem jest zmierzenie czasu, pamięci, skończoności strat i poprawnego zapisu checkpointu. To nie jest badanie zbieżności ani skuteczności doboru danych.

Przydział pamięci PyTorch ograniczony do połowy VRAM. Kontrola wolnej pamięci przed krokiem kończy próbę, gdy pozostaje mniej niż 512 MiB. Nie zatrzymujemy cudzych procesów. To nie jest rezerwacja sprzętowa dla RF-DETR: jednoczesne uruchomienie innego zadania może wciąż spowodować brak pamięci.

## Dane

Około tysiąca klatek z treningowych operacji, równy limit na film, ziarno 42, odstęp przynajmniej 20 klatek. Obrazy z lokalnej kopii lub oryginalnego ZIP; zgodność pikseli sprawdzona z manifestem. Maski tylko z `cystotome_seg`. Brak maski wyklucza klatkę.

Kwadrat obejmuje obrys z marginesem, bez rozciągania proporcji. RGB zmniejszane interpolacją powierzchniową, maska najbliższym sąsiadem. Dane RGBA w PNG: ostatni kanał jest maską, a nie przezroczystością używaną do zmiany RGB. Manifest zapisuje pochodzenie i transformację. Zbiory val i test nie służą treningowi.

## Model i różnice względem źródła

Generator i dyskryminator z pobranej, przypiętej wersji HF, szerokość 64, cztery kanały, losowa inicjalizacja. Wagi wytrenowane na polipach nie są używane. LPIPS-VGG obliczany tylko na RGB, dodatkowy L1 maski w rekonstrukcjach dyskryminatora. Jest to jawna adaptacja do segmentacji, nie dokładna reprodukcja wszystkich ustawień publikacji.

Wspólne odbicie poziome RGB i maski. Wyłączona augmentacja kolorów źródła, która obejmowała wszystkie kanały, w tym maskę. Dodatkowy forward generatora w kroku G ogranicza czas przechowywania grafu; koszt czasu jest ujęty w pomiarze. Bez AMP, bez DataParallel, bez dostrajania hiperparametrów do wyników.

## Raportowanie

Pomijamy pierwsze 10 kroków przy szacowaniu czasu ustalonego. Zapisujemy medianę i percentyl 90 czasu, maksymalną pamięć przydzieloną i zarezerwowaną przez PyTorch oraz ekstrapolację do 50 tysięcy kroków. Ekstrapolacja nie zawiera przygotowania danych i nie gwarantuje jakości generatora.

Checkpoint i losowe próbki służą kontroli wykonania. Nie traktujemy próbek po krótkiej próbie jako dowodu poprawnej anatomii, zgodności maski lub zachowania latentu RF-DETR. Kolejny etap zależy od wyniku pomiaru i decyzji użytkownika.
