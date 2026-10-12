# Sesja: przygotowanie eksperymentu bez liczenia

## Polecenie

Użytkownik najpierw poprosił o założenie i sprawdzenie eksperymentu. Przed wykonaniem obliczeń zawęził zakres: „mozesz założyc taki eksperyment i napisac kod na razie niczego nie licz!”. Obowiązuje drugie polecenie.

## Zapisany kod

Katalog `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\tozsamosc_czastki_bledy`.

Plan diagnostycznego porównania: pewność, geometria, przestrzenny sygnał narzędzia, pozostałe cechy i historia, EDMD, skończona pamięć oraz nominalny Kalman. Kontrola cyklicznie przesuwa cechy dynamiki wewnątrz filmu. Podział A/B/C po operacjach oddziela dopasowanie kierunku i dynamiki od predyktora błędów i oceny. Kierunek referencyjny służy wyłącznie porównaniu; kierunek oceniany jest dopasowany na A.

Przygotowano adapter zapisów cząstek i rzeczywistych masek `cystotome_seg`, eksport segmentacji z oryginalnych archiwów przy jawnym wzorcu nazw, konfigurację, skrypty i testy aparatu. Nie tworzyliśmy żadnych wejściowych tablic ani wyników. Nie wykonaliśmy nawet testów składni lub importów, zgodnie z ograniczeniem użytkownika. Kod jest niezweryfikowany wykonaniem.

## Granice

- Kod mierzy przewidywanie błędów, nie poprawę uczenia detektora.
- EDMD z ustalonym słownikiem nie jest deep Koopman autoencoderem.
- Przybliżenie pamięci nie identyfikuje pełnego równania Langevina ani wymuszeń kamery i ręki.
- Nominalny Kalman nie dostarcza automatycznie skalibrowanej niepewności detektora.
- Znany problem powtarzalności rejestratora i wcześniej dopasowana PCA pozostają ograniczeniami.
- Nie wdrożono jeszcze wyszukiwania dopasowanych par sukces–porażka ani interwencji w latent.
- Pliki innych sesji pozostawiono bez zmian. Nie wykonywano operacji na Edenie.

## Dalszy krok

Czekać na polecenie uruchomienia. Wtedy najpierw sprawdzić kontrakt identyfikatorów i masek, testy aparatu oraz dostępność ciągłych sekwencji przy ustalonym kroku. Nowy trening detektora wymaga przedstawienia i zatwierdzenia odrębnego planu.
