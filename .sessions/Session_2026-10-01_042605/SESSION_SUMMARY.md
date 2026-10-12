# Szczegóły sesji animacji

## Zakres

Przeczytano kod modelu, pętli odczytu LDA, prognozowania i doboru klatek oraz raporty o ruchu tokenów i równaniu latentu. Użytkownik poprosił o obszerną animację analogiczną do wcześniejszych filmów projektu Grant. Wcześniejsze filmy korzystały z ManimGL, mimo podanej przez użytkownika nazwy ManimML.

## Materiały

Katalog: `F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/docs/animacje_pipeline_vit`.

Skrypty: `scenariusz.py`, `przygotuj.py`, `sceny.py`, `render.py`, `narracja.ps1`, `zloz_i_sprawdz.py`. Scenariusz: `SCENARIUSZ.md`. Pochodzenie danych: `ZRODLA.json`. Wynik: `index.html`, osobne filmy i film zbiorczy w `videos/`, napisy SRT i WebVTT, manifest kontroli.

## Treść

Obraz 480 × 320, 600 tokenów, 192 cechy bloków i 768 aktywacji badanej warstwy MLP. Kod pozycji, attention, oddzielenie czasu filmu od czasu uczenia. Maski od jednego piksela, odczyt LDA, podprzestrzenie, osobne trajektorie tokenów, SVD i wyniki na odłożonych filmach. Historia surowa i średnie z pięciu, prognozowanie przyrostów, pamięć i reszty wariantu Langevina, równanie wpływu bieżącej paczki, historyczny FastGAN i wyszukiwanie prawdziwych klatek.

## Rzetelność

Rzeczywiste trajektorie tokenów pochodzą tylko z dwóch stanów próbnego biegu, kroki 60 i 120. Punkty pośrednie są interpolacją. Nowa gęsta kampania nie została sprawdzona na klastrze i jest przedstawiona jako plan. Kierunki SVD nie są ogłoszone przyczynowymi kierunkami tożsamości.

Dodatkowo dopasowano przykład prognozy jednej sondy na 100 surowych odczytach, kroki 10..1000, z oceną 1010..1500. RMSE: Koopman 1,1276, wariant Langevina 2,0911, stała ostatnia wartość 1,3193. To ilustracja jednego przypadku, nie ogólne porównanie skuteczności.

## Technika i kontrola

1920 × 1080, 24 klatki/s, H.264 i AAC. Polski głos zapisany do WAV przez System.Speech; bez odtwarzania przez głośniki. OmniVoice nie odpowiadał na porcie 8080. Wykorzystano istniejące środowiska, bez instalowania pakietów. Próbny render obejmował wszystkie 51 ujęć. Sprawdzono wizualnie plansze kontrolne i pełnowymiarowe ujęcia prognozy oraz LDA. Nieobsługiwane indeksy Unicode zastąpiono czytelnym zapisem tekstowym i ponownie wyrenderowano odpowiednie rozdziały. Skrypt montażu weryfikuje długość, liczbę ujęć, obecność audio, rozdzielczość oraz dekoduje wszystkie rozdziały.

Nie uruchomiono treningów, nowych zadań klastra ani instalacji. Nie usuwano danych, nie wykonywano commitów ani push.
