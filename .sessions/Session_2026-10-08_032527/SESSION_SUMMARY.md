# Session Summary: Session_2026-10-08_032527

## Metadata

- Date/time: 2026-10-08 03:25:27 Europe/Warsaw.
- Status: Completed. Oba opisane nowe eksperymenty zakończone, bez oczekującego treningu lub zadania SD.
- Type: Mixed — Analysis / Benchmark.
- Workspace: `F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation`.
- Jest to podsumowanie i zapis stanu pracy, nie pełny transkrypt rozmowy.

## Objective and context

Użytkownik bada tokeny jako cząsteczki we wspólnym latencie i chce nadać ich trajektoriom znaczenie: rozdzielić zmiany kodowania pozycji, koloru i kształtu oraz sprawdzić związek z poprawą detekcji. Zlecił najpierw eksperyment semantyczny, potem bardziej kontrolowane obrazy: jednolite kolory, warianty narzędzia i generatory FastGAN/SD. Oba polecenia wykonano. Nie modyfikowano GUI, wag modeli ani źródłowych danych treningowych.

W rozmowie rozróżniano odczytywalność cechy, jej wykorzystanie przez detektor i skuteczność ingerencji. Nazwa sondy nie dowodzi, że znalezione kierunki stanowią niezależny moduł koloru/kształtu. Generowanie przez SD nie jest samo w sobie precyzyjną weryfikacją geometrii lub przyczynowego znaczenia latentu.

## Terminology and constraints

- SmallViT w tym projekcie: 600 przestrzennych tokenów na obraz 480×320, siatka 20×30; mierzona reprezentacja ma 768 współrzędnych.
- Cząsteczka to reprezentacja jednego tokenu. Makrowektory to wspólne osie e_j; mikrowektory to składowe z_j e_j.
- Ruch treningowy mierzymy przy stałym obrazie i tym samym tokenie między checkpointami. Zmiany obrazu to osobny eksperyment odpowiedzi na bodziec.
- Pomiar dotyczy wyjścia GELU ostatniego MLP, przed fc2. Interwencja zmienia tę gałąź, a nie całą reprezentację z połączeniem rezydualnym.
- Nowy kod i plany: `SmallViTAblation/experiments/`; wyniki: `SmallViTAblation/outputs/`. Nie używać projektu Grant.
- Nie uruchamiać ponownie zakończonych eksperymentów tylko dla zmiany ścieżek. Nowe biegi mają osobne katalogi wyników.
- Istniejących zmian użytkownika w app ani innych katalogach nie nadpisywać. Nie wykonano commita.

## Sources

Wspólna rodzina wag i obrazów obu nowych eksperymentów:

`F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_ruch_czastek_gesta_historia/wyniki/pelny`

- `dane/obrazy.npy`: 12 obrazów RGB uint8.
- `dane/images.json`: role fit/holdout, filmy i prawdziwe poligony cystotome_seg; skalowanie z oryginalnej rozdzielczości do 480×320.
- `dane/LATENT_GESTY.npy`: 1001×12×600×768, float32, około 22 GB. Nie kopiowano całej historii.
- `wagi/k0000.pt` … `k1000.pt`: checkpointy co 10 aktualizacji; klucze net i krok.
- Model: `experiments/vit_sd_pelny_latent_filmy_eden_20261006/model_snapshot.py`, zgodny z tą rodziną wag.
- Osiem obrazów fit, cztery holdout diagnostyki (indeksy 8–11), każdy z innego filmu. Wszystkie były obrazami treningowymi samej sieci. To nie jest niezależny test generalizacji detektora.

Nie łączyć tej rodziny wag z siłami paczek z innego przebiegu na Edenie.

## Experiment 1: semantic motion probes

Code: `experiments/vit_semantyka_ruchu_20261008/`

- `PLAN.md`, `run.py`, `verify.py`, `interpret.py`.
- Completed output: `outputs/vit_semantyka_ruchu_20261008/pilot_v1/`.
- Wstępne sprawdzenie dwóch checkpointów: `outputs/vit_semantyka_ruchu_20261008/smoke/`.

21 checkpointów 0–1000 co 50. Czynniki: cyfrowa translacja całego kadru ±16 px; Lab a/b ±12 tylko w masce narzędzia; deformacja proporcji całego kadru wokół środka ramki z zachowaniem powierzchni. Ta ostatnia zmienia też tło, więc jest sondą geometrii/proporcji, nie czystego kształtu.

Rank-8 SVD odpowiedzi na ośmiu obrazach dopasowania; test na czterech pozostałych. Ruch stałych obrazów rzutowany na bazę wcześniejszego checkpointu. Lokalna interwencja: cofnięcie tylko tej części ruchu w ostatniej gałęzi MLP. Pięć losowych kontroli o równym wymiarze oraz kontrola tej samej normy ingerencji. IoU ramki o najwyższej pewności, nie mAP.

Wyniki:

- Średni udział energii ruchu: pozycja 0.2531, kolor 0.2481, proporcje 0.2797. Udziały nakładają się i nie sumują do jedności.
- Spearman udziału z ΔIoU: −0.0654, −0.0917, −0.0708. Po usunięciu średniej dla każdego checkpointu: −0.0399, −0.0753, 0.0003. Opisowe, bez naiwnego testu istotności dla zależnych kroków.
- Średnie nakładanie baz: pozycja–kolor 0.541, kolor–proporcje 0.553, pozycja–proporcje 0.682 (średnia kwadratów cosinusów kątów głównych).
- Sonda koloru przechwytuje własną perturbację 0.378, lecz pozycję 0.412; brak czystego rozdzielenia. Sonda pozycji jest częściowo selektywna.
- Dominujący przypadek: obraz 11, checkpoint 500. Cofnięcie części ruchu podniosło top-1 IoU z 0 do około 0.30285. Zmienił się ranking dwóch prawie równie pewnych ramek, nie tylko lokalizacja. Efekt średni jest mocno zależny od tego jednego przypadku.
- Obliczenia RTX 3070: około 54.24 s, maksymalna alokacja około 927.6 MiB.
- QA PASS: ortonormalność, niezależna weryfikacja udziału ruchu z archiwum, zachowanie tła/maski dla koloru, dokładna translacja i identyczny wynik zerowej interwencji.

Artefakty: REPORT.md, RESULT.json, QA.json, INTERPRETATION.json, MANIFEST.json, MOTION.csv, PROBES.csv, DETECTION.csv, OVERLAP.csv, STATES_BASES.npz, DIAGNOSTICS.png, PERTURBATIONS.png, kopia wykonanego kodu.

## Experiment 2: controlled images and FastGAN

Code: `experiments/vit_kontrolowane_obrazy_20261008/`

- `PLAN.md`, `run.py`, `transfer.py`, `report.py`, `control_latents.py`.
- Completed output: `outputs/vit_kontrolowane_obrazy_20261008/pilot_v1/`.

FastGAN rzeczywiście uruchomiono na CPU. Checkpoint:

`experiments/fastgan_narzedzie/wyniki/eden_20260923_50k/results_50k/checkpoint_050000.pth`

Wytrenowane EMA step=50000, nc=4, nz=256, ngf=64, im_size=256. Zweryfikowano strukturę wag i hash architektury. Checkpoint SHA256: `51895dcb216fc8b69f9b538461face293c418b7478b75812eb2130d7d3368fef`. Cztery ziarna 470–473; RGB i kanał maski zapisano. Maski FastGAN nie są prawdziwymi adnotacjami. SD nie załadowano: GPU i pamięć systemowa były zajęte innymi procesami; nie zatrzymywano ich.

Zestaw:

- 12 rzeczywistych narzędzi wyciętych według poligonów, cztery wycinki z FastGAN.
- Znane szare tło; stałe centrowanie i skala bazowa.
- Dla każdej z 16 kotwic: baseline, cztery translacje ±32 px, cztery zmiany Lab a/b ±12, dwie deformacje proporcji x1.25/x0.8. Łącznie 176 obrazów.
- Kolor nie zmienia maski ani tła; przesunięcie zachowuje wygląd obiektu; deformacja nie zmienia tła.
- 11 kontroli: czerń, biel, czerwony, zielony, niebieski, cztery szarości, gradient x i gradient y.
- Checkpointy 0, 250, 500, 750, 1000.
- SVD i regularizowane sondy kontrastowe, rank 8; dopasowanie wyłącznie na ośmiu rzeczywistych kotwicach fit.
- Kontrolowane testy na czterech rzeczywistych wycinkach holdout i czterech FastGAN. Dodatkowy test naturalnych klatek 8–11 na końcowym checkpointcie.
- Naturalny test koloru dotyczy tylko maski narzędzia; testy pozycji i proporcji zmieniają cały kadr. Ten transfer nie jest identyczną operacją jak ruch izolowanego obiektu na szarym tle.

Wyniki SVD, średnia z pięciu checkpointów na odłożonych rzeczywistych wycinkach:

- Pozycja: własna odpowiedź 62.6 procent energii; inne czynniki 38.8–43.2 procent.
- Kolor: własna odpowiedź 50.0 procent; inne 38.1–45.8 procent.
- Proporcje: własna 43.5 procent, przesunięcie aż 57.3 procent. Wciąż nie jest czystą sondą kształtu.
- FastGAN daje podobny schemat; sonda koloru przechwytuje własną odpowiedź 57.1 procent.
- Na naturalnych klatkach k1000 sonda koloru: własna odpowiedź 38.5 procent wobec 26.3–26.7 procent na pozostałych zmianach. To częściowa selektywność, nie odkrycie oddzielnego modułu koloru.
- Contrast bardziej selektywny na wycinkach, lecz przechwytuje mniej niż jeden procent energii odpowiedzi/ruchu. Średnie efekty ingerencji w gałąź MLP na IoU są małe i bliskie losowej kontroli tej samej normy.
- Rzadkie checkpointy dają tylko 16 zależnych obserwacji na metodę/czynnik. Korelacji nie interpretować jako dowodu dominującej przyczyny poprawy ani porównywać bezpośrednio z wcześniejszym próbkowaniem co 50 kroków.
- Kontrole k1000: najwyższa pewność 0.430 dla gradientu poziomego; żadna z dekodowanych ramek nie przekracza progu 0.5. To test negatywny na ustalonym progu, nie globalna gwarancja braku fałszywych alarmów.
- Zapisano również zmiany latentów wszystkich kontroli względem szarości 127 oraz 20 najsilniej reagujących kanałów. Podstawowe RGB/czerń/biel różnią się też jasnością: nie utożsamiać takich kanałów z czystą reprezentacją chromatyczności.
- Główny bieg CPU trwał 126.37 s; transfer i kontrolę latentów wykonano osobno.
- QA PASS: 720 wierszy sond, 96 pomiarów ruchu/interwencji, 72 testy transferu, 330 rzutów latentów kontroli; ortonormalność max błąd 2.15e−6, tożsamość zerowej interwencji i dokładne kontrole transformacji.

Artefakty: REPORT.md, RESULT.json, SUMMARY.json, QA.json, MANIFEST.json, GENERATOR.json, IMAGES.json, CONTROLLED_IMAGES.npy, TOKEN_REGIONS.npy, BASES.npz, PROBES.csv, DETECTIONS.csv, CONTROLS.csv, MOTION.csv, NATURAL_TRANSFER.csv, CONTROL_LATENTS.csv, CONTROL_CHANNELS.csv, CONTROL_FEATURES.npz, SELECTIVITY.png, EXAMPLES.png i katalog images/ z PNG oraz maskami generatora.

## Earlier conversation / separate experiment

Wcześniejsze przekazane wyniki Koopmana z wejściami są osobną rodziną danych i nie zostały ponownie przeliczone w tej części pracy. Kod: `experiments/vit_dmdc_paczki_20261008/run.py`; raport: `outputs/vit_dmdc_paczki_20261008/full_v2/REPORT.md`. DMDc z siłami i pamięcią był słabszy od prostego modelu siły/bezwładności; prognozy wielokrokowe używały rzeczywistych przyszłych sił, więc są warunkowe/oracle, nie przewidują nieznanych paczek. Nie mieszać jego źródeł Eden z lokalną rodziną semantycznych checkpointów.

## Commands used

Uruchamiano z katalogu SmallViTAblation, lokalny Python `F:/Python/3.13/python.exe`:

```powershell
python experiments/vit_semantyka_ruchu_20261008/run.py --out outputs/vit_semantyka_ruchu_20261008/pilot_v1
python experiments/vit_semantyka_ruchu_20261008/verify.py
python experiments/vit_semantyka_ruchu_20261008/interpret.py
python experiments/vit_kontrolowane_obrazy_20261008/run.py
python experiments/vit_kontrolowane_obrazy_20261008/transfer.py
python experiments/vit_kontrolowane_obrazy_20261008/report.py
python experiments/vit_kontrolowane_obrazy_20261008/control_latents.py
```

Nie wykonywać tych komend ponownie bez potrzeby. Drugi run.py odmawia nadpisania zakończonego biegu. report.py przebudowuje raport, więc po jego powtórzeniu trzeba odtworzyć dodatkowe sekcje z control_latents.py i interpretację dopisaną na końcu. Bieżący REPORT.md zawiera wszystkie sekcje.

## Next steps — suggestions, not started

- Przeczytać raporty i obejrzeć EXAMPLES.png / SELECTIVITY.png przed kolejnym eksperymentem.
- Testować nowe rzeczywiste klatki niewidziane przez sieć i kilka niezależnych przebiegów treningu.
- Oddzielić pozycję obiektu od ruchu całej kamery, chromatyczność od jasności oraz proporcje 2D od kategorii/pełnej geometrii kształtu.
- Badać nakładanie się sond i kontrolować wspólną odpowiedź, nie dzielić arbitralnie kanałów na moduły.
- Obok top-1 IoU mierzyć ciągłe błędy lokalizacji i ranking/prawdopodobieństwo ramek, aby nie mylić przeskoku rankingu ze zmianą geometrii.
- SD traktować jako opcjonalną wizualizację i kontrolować jego zgodność; obecnie nie uruchomiono generowania SD w tym eksperymencie.
- Publikacja: potrzebny powtarzalny sygnał, transfer i korzyść dla decyzji treningowych lub diagnozy, nie sama animacja cząsteczek. Nie ustalono pierwszeństwa naukowego.
