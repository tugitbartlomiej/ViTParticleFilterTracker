# Tożsamość, cząstki i błędy detektora

**Stan: kod przygotowany, niczego nie uruchomiono. Nie ma wyników.**

Eksperyment dotyczy istniejących cech zamrożonego detektora. Nie uruchamia RF-DETR, nie wysyła zadań na Eden i nie zmienia innych eksperymentów. Testy również nie zostały wykonane. Kod może wymagać poprawek po pierwszym uruchomieniu.

Plan i kryteria: [PLAN.md](PLAN.md). Deklaracja: [WYNIKI.json](WYNIKI.json). W tym repozytorium używamy lokalnej deklaracji rodziny; nie uruchamialiśmy narzędzi rejestru projektu grantowego.

## Pliki

- `dane.py`: jawne połączenie maski segmentacji, klatki i tokenów; przygotowanie wejścia.
- `modele.py`: kierunek narzędzia, przestrzenny opis cząstek, EDMD, pamięć, reszta, nominalny Kalman.
- `uruchom.py`: podział po operacjach, porównania, kontrola cykliczna i raport.
- `test_aparatu.py`: przygotowane testy granic czasu, masek, IoU, przyczynowości i kowariancji.
- `config.json`: ustawienia ustalone przed wynikiem.

## Co będzie mierzone

Czy znany rodzaj sygnału narzędzia pozwala przewidzieć błąd detektora, a jego rozkład przestrzenny i dynamika dodają informację ponad pewność, geometrię, pozostałe cechy i prostą historię. Kierunek jest dopasowywany na wydzielonych operacjach; stary wektor można podać tylko do porównania cosinusa. Tym samym unikamy użycia etykiet ocenianych filmów do ustalenia kierunku.

Nie jest to jeszcze dowód skutecznego doboru klatek do treningu. Plik CSV zawiera oceny poza próbą i może być podstawą następnego, osobno zatwierdzonego eksperymentu. W tym etapie nie wdrożono generowania obrazów, dopasowywania par sukces–porażka ani deep Koopman autoencodera. Pełne wymuszenie kamery i sterowanie wymagają dodatkowych pomiarów.

## Kontrakt segmentacji

Wejściowy JSONL maski musi mieć jeden rekord na klatkę:

```json
{"frame_id":"cataract_lmm/tracking/NAZWA_FILMU/00000006","width":720,"height":480,"polygons":[[340,208,561,208,561,260,340,260]]}
```

Powyższy wielokąt jest wyłącznie przykładem formatu. W eksperymencie musi pochodzić z rzeczywistego `cystotome_seg`, nie z obwiedni. Można wykorzystać istniejący indeks albo przygotowany eksporter z ZIP i `annotation.csv`. W eksporterze trzeba podać dokładny wzorzec `filename` po sprawdzeniu źródłowego CSV. Program nie zgaduje przesunięcia indeksów i nie pobiera obrazów.

## Polecenia na później — nie zostały wykonane

Uruchamianie z katalogu `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`.
Zależności: Python 3.11, numpy, scipy, scikit-learn, opencv-python. Niczego nie instalowano.

```powershell
# Po utworzeniu prawdziwego indeksu masek MASKI.jsonl:
py -3.11 -m SmallViTAblation.experiments.tozsamosc_czastki_bledy.dane prepare `
  --recording "F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\results\koopman_langevin_epochs\czastki\CZASTKI_pool_b0500" `
  --manifest "F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\feature_cache\eden_static36\data\tracking_static36\pool_manifest.jsonl" `
  --masks "F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\tozsamosc_czastki_bledy\dane\MASKI.jsonl" `
  --split train `
  --out "F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\tozsamosc_czastki_bledy\dane\WEJSCIE.npz"

py -3.11 -m unittest SmallViTAblation.experiments.tozsamosc_czastki_bledy.test_aparatu

py -3.11 -m SmallViTAblation.experiments.tozsamosc_czastki_bledy.uruchom `
  --data "F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\tozsamosc_czastki_bledy\dane\WEJSCIE.npz" `
  --out "F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\tozsamosc_czastki_bledy\wyniki\bieg_001"
```

Pula bywa rzadka w czasie. Domyślny krok wynosi trzy klatki; kod odmówi pomiaru, jeśli zabraknie pełnych sekwencji. Nie zmieniać kroku po obejrzeniu wyników. Do istniejących gęstych klipów trzeba przygotować manifest o tym samym kontrakcie, bez mieszania podziałów. W przykładzie `train` diagnostyka dotyczy puli treningowej; nie jest oceną generalizacji detektora na niewidziane podczas jego treningu operacje. Podział A/B/C dotyczy dopasowania diagnostyki.

## Oczekiwane wyjście

Nowy katalog biegu: `REPORT.json`, `MODELE.json`, `PREDYKCJE.csv`. Raport zapisuje konfigurację, wersje bibliotek, sumy kontrolne danych i kodu, podziały operacji, widmo oraz ograniczenia. `MODELE.json` jest audytem parametrów i widma, nie checkpointem wszystkich predyktorów. Po rzeczywistym biegu należy opisać wynik i zmienić status deklaracji. Obecnie plików wynikowych nie ma.
