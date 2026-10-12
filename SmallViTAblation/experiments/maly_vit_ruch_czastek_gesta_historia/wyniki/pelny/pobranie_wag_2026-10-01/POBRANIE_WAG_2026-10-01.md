# Pobranie 101 checkpointów małego ViT z Edena

Data: 2026-10-01. Status: **zakończone, integralność i struktura zweryfikowane**. Pobranie wykonano na wyraźne polecenie użytkownika po zakończeniu researchu o Koopmanie i wagach.

## Wynik

- Lokalnie znajduje się **101 plików** `k0000.pt` … `k1000.pt`, kroki **0–1000 co 10 aktualizacji**.
- Łączny rozmiar: **1 184 784 641 bajtów**, czyli **1,184784641 GB / 1,103416682 GiB**.
- **101/101 rozmiarów i sum SHA-256 zgadza się z niezależnie obliczonym manifestem zdalnym.**
- **101/101 wartości pola `krok` odpowiada nazwie pliku.** Każdy checkpoint zawiera klucze `net` i `krok`; stan sieci ma 98 tensorów o zgodnych nazwach, kształtach i typach.
- W każdym stanie są trzy komplety buforów BatchNorm: running mean, running variance oraz licznik paczek. Nie wczytywano ani nie pobierano pełnych stanów AdamW.
- Transfer i sprawdzenie SHA-256: **16:30:35–16:33:04 czasu Europe/Warsaw** (14:30:35–14:33:04 UTC), około 150 sekund. Odczyt struktury zakończono około 16:34.
- Jeden przebieg SFTP, bez ponowień i bez plików wymagających izolacji. Po transferze wolne miejsce na F: wynosiło około **23,96 GB**.

Katalog lokalny:

```text
F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_ruch_czastek_gesta_historia\wyniki\pelny\wagi
```

Przykładowe checkpointy: [k0000.pt](../wagi/k0000.pt), [k0500.pt](../wagi/k0500.pt), [k1000.pt](../wagi/k1000.pt).

Źródło, dostęp przez istniejący alias SSH `eden-cluster`:

```text
/home2/faculty/bpiotrowski/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/wagi
```

## Weryfikacja i możliwość wznowienia

Najpierw na Edenie sprawdzono dokładny zbiór nazw oraz rozmiary i obliczono SHA-256 każdego pliku. Podczas hashowania sprawdzono także niezmienność rozmiaru i czasu modyfikacji. Manifest zapisano lokalnie przed rozpoczęciem transferu.

Transport korzystał z OpenSSH SFTP `reget`. Pobieranie odbywało się do plików `.pt.part`, dzięki czemu przerwaną kopię można wznowić. Plik otrzymywał nazwę `.pt` dopiero po zgodności rozmiaru i SHA-256. Skrypt sprawdza istniejące pliki końcowe i pomija poprawne, a przy różnicy przerywa bez ich nadpisania. Nie wykonuje operacji zapisu po stronie Edena.

Po zakończeniu ponownie zweryfikowano komplet 101 lokalnych plików i odczytano ich metadane przez `torch.load(..., weights_only=True, map_location="cpu")`, PyTorch 2.10.0+cu126. Weryfikacja struktury nie wykonywała forward sieci ani numerycznego audytu wartości wszystkich tensorów.

## Manifesty i narzędzia tego pobrania

- [Manifest zdalny: nazwy, kroki, rozmiary, SHA-256](MANIFEST_ZDALNY_2026-10-01.json).
- [Manifest lokalny z porównaniem 101 hashy](MANIFEST_LOKALNY_20261001_163035.json).
- [Lista lokalnych SHA-256](SHA256SUMS_20261001_163035.txt).
- [Weryfikacja numerów kroków, kluczy i tensorów](WERYFIKACJA_STRUKTURY_2026-10-01.json).
- [Wznawialny skrypt pobrania i hashowania](pobierz_i_sprawdz.ps1).
- [Skrypt odczytu struktury checkpointów](sprawdz_strukture_checkpointow.py).
- [Log SFTP](sftp_20261001_163035_attempt1.log) i [dokładna lista poleceń pobrania](sftp_20261001_163035_attempt1.txt).

Nie pobrano `LATENT_GESTY.npy` ani katalogu pełnych stanów AdamW. Nie uruchomiono rekonstrukcji latentów, treningów, zadań SLURM ani commitu. Zdalne checkpointy zachowano. Wcześniejsza informacja w README eksperymentu, że wagi pozostają wyłącznie na Edenie, opisuje stan sprzed tego autoryzowanego pobrania.
