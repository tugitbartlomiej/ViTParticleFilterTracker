# Kierunki ruchu tokenów narzędzia z maską od jednego piksela

Plan: [PLAN.md](PLAN.md). Dane i kod należą do repozytorium ViTParticleFilterTracker, nie projektu grantowego.

Osobny lokalny eksperyment na istniejących checkpointach małego ViT. Pełne wejście 480 × 320, 600 tokenów po 768 cech, bez średnich po tokenach i bez wygładzania czasowego. Kod przygotowania pomiaru i manifest określą faktyczną dostępną historię.

Narzędzia rejestru wyników z projektu grantowego nie są importowane do tego niezależnego repozytorium. Plan, skrypty, manifest danych i raport przechowywane są razem, zgodnie z lokalnym układem eksperymentów.

Status: wykonano wstępny pomiar na lokalnej RTX i analizę. [WYNIKI.md](WYNIKI.md) opisuje rezultat i ograniczenia.

Istotna granica: dostępne dwa stany jednej krótkiej próby (kroki 60 i 120), nie sto stanów rozgrzewki do kroku 1000. Nie da się odzyskać osobnych cząstek ze starych średnich. Ten bieg nie wyznacza prognozy Koopmana ani Langevina.

Uruchomienie przez Python środowiska rfdetr_1_10_1:

```powershell
& 'F:\Studia\PhD_projekt\VIT\venvs\rfdetr_1_10_1\Scripts\python.exe' -B 'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_ruch_czastek_maska_1px\pomiar.py'
& 'F:\Studia\PhD_projekt\VIT\venvs\rfdetr_1_10_1\Scripts\python.exe' -B 'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_ruch_czastek_maska_1px\analiza.py'
```

Pomiar odmawia nadpisania istniejącego latent.npy. Analizę można ponowić na zachowanym pomiarze bez używania GPU.

`wyniki/dane/latent.npy` zawiera wszystkie tokeny, `maski.npz` przypisania i pokrycie, `metadata.json` źródła oraz kroki.
`wyniki/KIERUNKI_RUCHU.npz` zawiera osie SVD i podpisany kierunek średniego ruchu w 768 współrzędnych.
`wyniki/REPORT.json` zawiera wyniki poza dopasowaniem i sumy SHA256.
`wyniki/MASKI_600_TOKENOW.png` pokazuje dodatkowe tokeny po zmianie progu.
`wyniki/RUCH_TOKENOW.png` pokazuje energię kierunków, kontrolę tła oraz osobne przemieszczenia.
