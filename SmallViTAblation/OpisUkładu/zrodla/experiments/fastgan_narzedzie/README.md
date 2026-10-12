# FastGAN do przyszłych eksperymentów z narzędziem

**Zakończenie:** trening na Edenie ukończono na kroku 50 000, model i podglądy pobrano lokalnie oraz sprawdzono sumami SHA256. Wykonano też zatwierdzoną próbę zmiany otoczenia prawdziwego narzędzia przy zamrożonych wagach. Plan: [PLAN_PROBY_TRUDNYCH.md](PLAN_PROBY_TRUDNYCH.md), wynik: [WYNIKI_PROBY_TRUDNYCH.md](WYNIKI_PROBY_TRUDNYCH.md), kod: `proba_trudnych.py`. Pierwsze porównanie dotyczy 12 wycinków; nie dowodzi poprawy uczenia detektora. Poniżej zachowano zapis etapów przygotowania.

**Aktualny stan:** lokalna próba zakończona poprawnie; 200 kroków, mediana 0,248 s/krok. Szczegóły i granice: [WYNIKI_PILOT.md](WYNIKI_PILOT.md). Dłuższy trening zatrzymano lokalnie po kroku 16427 na prośbę o przeniesienie na Eden. Kontynuację do 50 000 kroków zgłoszono jako zadanie **1809936**. Lokalny raport `stopped_requested` oznacza planowe przeniesienie, nie awarię.

Parametry połączenia, lokalne ścieżki i identyfikatory: [EDEN_RUN.json](EDEN_RUN.json). Zdalny stan: `~/smallvit_runs/2026-09-23_fastgan/results_50k/STATUS.json`. Automatyzacja rozmowy kontroluje ukończenie i uruchamia `pobierz_eden.py`; docelowe pliki lokalne: `wyniki/eden_20260923_50k/results_50k/`, a potwierdzenie weryfikacji: `wyniki/eden_20260923_50k/POBRANIE_OK.json`. Plan i zasady wznowienia: [PLAN.md](PLAN.md).

Źródło: https://huggingface.co/deepsynthbody/deepfake_gi_fastGAN

Przypięta wersja: `5dcd21b18d67339076b41355e682ed6b5a592b45`.
Pobieranie: `pobierz.ps1`. Manifest pobrania i sumy kontrolne: `POBRANIE.json`.
Kod i wagi są w `vendor/deepfake_gi_fastGAN`, poza Git. Pomijamy cudze wyniki, cache i ustawienia Git.

## Zakres

Użytkownik 2026-09-23 polecił pobrać FastGAN lokalnie, z możliwością późniejszego treningu na Edenie, jeżeli lokalny wariant będzie zbyt wolny.
Przy pobieraniu nie uruchamiano treningu ani wczytywania checkpointu do Pythona.
Następnie użytkownik zatwierdził lokalną próbę słowami „puszczaj na lokalnym”. Plan: `PLAN.md`.
Następnie użytkownik jawnie zlecił przeniesienie na Eden oraz pobranie gotowego modelu; kontynuacja ma osobny zapis w `PLAN.md`.

## Zawartość modelu

Model generuje obraz RGB i maskę jako dodatkowy kanał. Konfiguracja źródła: `ngf=64`, `noise_dim=256`, `nc=4`, `im_size=256`.
Wagi są wyuczone na obrazach polipów, nie cystotomu. Nie stanowią gotowego generatora naszych operacji.
Sam generator nie odwraca jeszcze latentu RF-DETR i nie gwarantuje zachowania sygnału tożsamości. To wymaga osobnej funkcji celu i optymalizacji wejścia generatora.

## Lokalna próba

`przygotuj_wycinki.py` tworzy dane na podstawie prawdziwej segmentacji i weryfikuje skróty oryginalnych pikseli. `pilot_lokalny.py` uruchamia ograniczoną próbę z jawnymi poprawkami obsługi maski. Oryginalnego kodu w `vendor` nie zmieniano.

Środowisko: `F:\Studia\PhD_projekt\VIT\venvs\fastgan_local`. Współdzieli bazowy torch tylko do odczytu przez `--system-site-packages`; LPIPS zainstalowano lokalnie z `--no-deps`. Nie zmieniano środowiska RF-DETR ani torch. Wagi VGG pobrano do standardowego cache torch na potrzeby funkcji straty.

Uruchomienie z katalogu repozytorium:

```powershell
& 'F:/Studia/PhD_projekt/VIT/venvs/fastgan_local/Scripts/python.exe' -u SmallViTAblation/experiments/fastgan_narzedzie/pilot_lokalny.py --steps 200 --batch 2 --out SmallViTAblation/experiments/fastgan_narzedzie/wyniki/NOWY_KATALOG
```

Nie nadpisywać istniejącego katalogu biegu. To trening od losowych wag, bez checkpointu polipów. Kolejny dłuższy trening musi mieć ocenę realizmu, poprawności końcówki i zgodności maski. Dostępność zasobów Eden trzeba sprawdzić ponownie przed ewentualnym uruchomieniem.
