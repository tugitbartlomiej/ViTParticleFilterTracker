# Generator obrazów z pełnego latentu ViT K8000

**Zapisany model: A_control, najlepszy checkpoint po 10 500 aktualizacjach SD.** To model użyty w podglądzie „A_control · świeże wczytanie najlepszego checkpointa”. Otrzymuje pełny latent ViT i generuje obraz od losowego szumu. Wybrano go na podstawie najmniejszego względnego błędu pełnego latentu w porównaniu czterech wariantów na Edenie.

Model generuje rozpoznawalne obrazy operacyjne oka, ale nadal zniekształca narzędzie i nie odtwarza wiernie jego położenia. Jest checkpointem badawczym; nie stanowi gotowego generatora poprawnie oznaczonych danych treningowych.

## Co oznacza nazwa

- **SD**: własny model dyfuzji w przestrzeni VAE, nazywany SD w eksperymencie. To niestandardowa architektura, a nie gotowy model Stable Diffusion 1.5 do wczytania przez `StableDiffusionPipeline.from_pretrained`.
- **ViT K8000**: zamrożony detektor MałyViT po 8000 krokach, który dostarcza reprezentację warunkującą. Jego wag nie trenowano podczas tego eksperymentu SD.
- **A_control**: kontrolny wariant kontynuacji; sekwencyjne próbkowanie co 600 klatek, learning rate 5e-5, bez EMA i bez dodatkowej straty ROI.
- **K10500**: licznik aktualizacji generatora SD. Własny bazowy model pochodził z kroku 7500, a najlepszy zapis A_control uzyskano po kolejnych 3000 krokach. Cały wariant wykonał 4000 nowych kroków, ale późniejszy zapis nie był najlepszy.

## Pochodzenie i zawartość

VAE, UNet i adapter rozpoczęły pierwotny trening 6 października 2026 od losowych wag. Porównanie z 7 października kontynuowało własny model z kroku 7500. W tej kontynuacji VAE oraz ViT były zamrożone, a uczono denoiser z adapterem. Użyto nowego optymalizatora, ponieważ bazowy plik best nie zawierał jego stanu.

Klatki odczytywano bezpośrednio z filmów. Obrazów ani latentów nie ekstrahowano na dysk. Trening porównawczy działał na czterech GPU H100 na Edenie; job produkcyjny 1823544 zakończył się poprawnie.

Plik `best_sd.pt` zawiera wagi **VAE, UNet, adaptera i uczonego kodowania pozycji**, skalę VAE oraz metadane wyboru checkpointa. Nie zawiera wag ViT ani stanu optymalizatora pozwalającego dokładnie odtworzyć kontynuację treningu.

| Plik | Zawartość |
|---|---|
| `best_sd.pt` | Kompletny model generatora do inferencji, 546 432 766 bajtów, około 521,1 MiB |
| `kod/model.py` | Niezmieniony zapis architektury generatora i funkcji obsługi teachera |
| `kod/model_snapshot.py` | Niezmieniona architektura MałegoViT |
| `PROVENANCE.json` | Pochodzenie pliku, lokalizacje, sumy kontrolne i zgodność modeli |
| `PROVENANCE_TRENINGU.json` | Kopia oryginalnych metadanych wariantu A_control |
| `METRYKI.json` | Zebrane pomiary walidacyjne i podsumowanie wcześniejszego testu lokalnego |
| `SHA256SUMS` | Sumy kontrolne modelu i plików pakietu |

SHA256 modelu:

```text
1a3fbac4234c7a16c1da036dce208971b579b81d012513c96f2938f697492e41
```

## Wejście, wyjście i wymagany ViT

Wejście warunkujące ma kształt **[batch, 600, 768]**. Jest to pełna reprezentacja po GELU w MLP ostatniego bloku konkretnego ViT K8000, zachowująca kolejność przestrzennych tokenów. Nie jest to średnia, CAV, wektor DINO ani wybrane 128 kanałów.

Wymagany teacher: [MALY_VIT_K8000_s47_losowe.pth](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/assets/MalyViT_K8000_s47_losowe/MALY_VIT_K8000_s47_losowe.pth).

```text
SHA256 ViT: 0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b
```

ViT otrzymuje RGB 480 × 320; normalizacja jest częścią jego implementacji. Generator zwraca RGB 576 × 384. Adapter działa na wszystkich 600 tokenach; warunek jest podawany przez cross-attention. Generator nie wymaga promptu tekstowego ani enkodera tekstu.

Sprawdzone ustawienie generowania: **DDIM, 50 kroków, spacing `leading`, eta=0**, start z szumu Gaussa o kształcie [batch, 4, 48, 72]. Harmonogram treningowy ma 1000 kroków, `squaredcos_cap_v2`, predykcję `epsilon`, `clip_sample=False`. Przy dekodowaniu trzeba używać skali zapisanej w checkpointcie, a nie stałej z innego modelu SD.

## Wynik na odłożonych filmach

Walidacja obejmuje 25 filmów, po dwie klatki na film i dwa seedy, czyli 100 generacji na ocenę. Wyniki uśredniano po seedach, klatkach, a następnie filmach. Checkpoint wybrano na tej walidacji; nie jest to osobny, nietknięty zbiór testowy.

| Miara A_control | Wynik |
|---|---:|
| Pełny względny MSE latentu, mniej = lepiej | 0,519988 |
| Cosine pełnego latentu, więcej = lepiej | 0,713380 |
| Względny MSE obszaru narzędzia (ROI) | 0,976266 |
| Pełny względny MSE przy warunku z innego filmu | 0,711378 |
| PSNR wygenerowanego obrazu względem źródła | 10,07 dB |
| SSIM wygenerowanego obrazu względem źródła | 0,2391 |
| IoU względem ramki obrazu źródłowego | 0,1718 |

W tej samej walidacji model bazowy miał względny MSE 0,623735. A_control zmniejszył go o **16,63%** i uzyskał lepszy wynik od wariantów B_mixed, C_optim oraz D_roi. Nie oznacza to wykazanej istotności statystycznej ani najlepszej jakości wizualnej.

Względny MSE to `mean((latent_obrazu - latent_celu)^2) / max(mean(latent_celu^2), 1e-6)`. Cosine jest osobną miarą podobieństwa kierunku, nie procentem zgodności wartości. ROI oznacza obszar przestrzenny pokrywający ramkę narzędzia, nie J128.

## Co sprawdzono lokalnie i jakie są ograniczenia

7 października świeżo wczytano model oraz VAE z `strict=True` na lokalnym RTX 3070, używając przenoszenia modułów między GPU i CPU. Potwierdzono hash modelu i właściwego teachera. Wygenerowano cztery obrazy: dwa odłożone filmy, dwa seedy; dodatkowo porównano zamienione latenty przy identycznym szumie. Naturalny forward ViT działał bez podmiany aktywacji; wszystkie wartości były skończone.

W tym małym teście względny MSE wyniósł średnio 0,494920, cosine 0,735686, a względny MSE przy innym warunku 0,830999. To kontrola kopii i działania warunkowania, nie niezależny wynik generalizacji. Obrazy pozostały w RAM.

Oględziny wykazały artefakty oraz niewierny kształt i położenie narzędzia. Nie wolno automatycznie przenosić etykiety źródłowej na wygenerowany obraz. Podane IoU mierzy zachowanie układu względem źródła, nie trafność wobec niezależnej etykiety generacji. Model nie osiągnął wcześniejszego celu względnego MSE ≤ 0,10 i cosine ≥ 0,90.

Nie potwierdzono testu ze sztucznym latentem utworzonym przez zgodny K8000. Dawne sztuczne cele z K1000 lub K6402 mają inne pochodzenie i nie należy traktować ich jako zgodnego wejścia bez osobnej oceny.

## Wczytanie wag

Kod wymaga środowiska z `torch`, `diffusers` i `timm`. Przykład odtwarza architekturę i wczytuje wagi; nie uruchamia generowania:

```python
from pathlib import Path
import sys
import torch

asset = Path(r"F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/assets/SD_ViT_K8000_A_control_K10500")
sys.path.insert(0, str(asset / "kod"))
from model import Denoiser, new_vae

checkpoint = torch.load(asset / "best_sd.pt", map_location="cpu", weights_only=False)
net = Denoiser().eval()
vae = new_vae().eval()
net.load_state_dict(checkpoint["net"], strict=True)
vae.load_state_dict(checkpoint["vae"], strict=True)
vae_scale = float(checkpoint["scale"])
```

Do generacji należy użyć zgodnego samplera z archiwum eksperymentu. Denoiser przewiduje szum jako `net(h, timestep, full_vit_latent)`. Po zakończeniu odszumiania dekodowanie RGB ma postać `((vae.decode(h / vae_scale).sample + 1) / 2).clamp(0, 1)`.

## Materiały źródłowe i przeniesienie

- [Końcowy raport porównania](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_k8000_ulepszenia_20261007/RAPORT_KONCOWY.md).
- [Raport testu lokalnego](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_k8000_ulepszenia_20261007/TEST_LOKALNY.json).
- [Archiwum kodu, logów i metryk](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_sd_k8000_ulepszenia_20261007/eden_kopia).

8 października 2026 na polecenie użytkownika przeniesiono lokalny plik z `outputs/vit_sd_k8000_ulepszenia_20261007/eden_kopia/wybrany_model` do tego katalogu `assets`. SHA256 i rozmiar sprawdzono przed i po przeniesieniu. Wagi są identyczne z wcześniej przetestowanymi. Samo przeniesienie nie uruchamiało nowego treningu ani nowej generacji. Historyczne raporty JSON zachowują ścieżki obowiązujące w chwili testu.
