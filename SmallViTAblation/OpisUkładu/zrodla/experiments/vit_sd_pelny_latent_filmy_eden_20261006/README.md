# SD warunkowany pełnym naturalnym latentem ViT

Kod modelu i treningu, bez SSH, deploy ani submit. SD jest dostrajany z istniejącego
`ema_020000`, nie uczony od zera. Domyślnie uczymy adapter i wszystkie parametry
cross-attention `attn2` UNet: query/key/value/output. Reszta UNet jest zamrożona,
ale przepuszcza gradient warunku. `--train-unet-mode full` odblokowuje cały UNet.
Nie ma PEFT, LoRA, EMA, kodera tekstu ani masek zależnych od przykładu.

## Pliki należące do modelu/treningu

- `model.py`: adapter 600x768 -> 600x768, wczytanie SD/ViT, straty pomocnicze.
- `model_snapshot.py`: niezmieniona kopia modelu z `vit_kierunek_nadzorowany_detekcja128`.
- `run.py`: CLI, torchrun DDP, BF16, akumulacja, ocena, checkpoint, wznowienie, smoke.
- `config.json`: ustawienia dla czterech GPU, global batch 32, 20000 aktualizacji.
- `PLAN.md`: hipoteza, kontrole, falsyfikacja i ograniczenia.
- `generate.py`: wczytanie checkpointu i DDIM -> tensor RGB w RAM, bez zapisów.

`dane.py`, `build_manifest.py` i pliki uruchomienia należą do innych agentów.
Nie są zastępowane przez ten kod.

## Dane i warunek

`dane.VideoFrames(manifest, split='train', rank=..., world_size=..., seed=..., repeat=True)`
zwraca słownik: `image` RGB float32 [0,1] 3x384x576, `vit_image` RGB [0,1]
3x320x480, `bbox` znormalizowane xyxy, `film` string, `frame` int, `has_tool` bool.
Manifest jest listą JSON wpisów `zip_path/member/video_id/split/annotation_zip`.
Sharding po filmach i workerach wykonuje dataset; nie dodajemy DistributedSampler.
Wszystkie klatki TRAIN służą rekonstrukcji, także bez bbox; wtedy `has_tool=False`
wyłącza wyłącznie pomocniczą stratę detekcji. Obecny dataset odtwarza MP4 w RAM.

Zamrożony ViT wykonuje zwykły forward na rzeczywistej klatce. Hook tylko odczytuje
naturalne aktywacje po GELU ostatniego MLP, nigdy ich nie podmienia. Rzeczywisty
`MODEL_VIT_K1000.pt` zawiera dokładnie `net` i `krok=1000`; fc1 ostatniego MLP
ma kształt 768x192. Loader używa strict=True.

Adapter: stałe skalowanie wejścia `--input-scale` (domyślnie 1), Linear/GELU/Linear
768 -> 768 -> 768 niezależnie na każdym tokenie. Uczone pozycje wierszy 20 i kolumn
30. Bez poolingu, PCA i LayerNorm: amplituda latentu nie jest normalizowana away.
UNet dostaje 600 tokenów i cztery kanały zaszumionego VAE + trzy kanały zer.
Bbox i `has_tool` służą tylko obliczeniu straty, nie są wejściem UNet.

## Straty

Główna: DDPM epsilon MSE, 1000 timestepów, scaled_linear beta .00085 -> .012,
zgodnie z `dyfuzja_sd_filmy/kod/trenuj_sd.py`. VAE encode online w FP32,
posterior sample x scaling_factor (SD 0.18215). Zamrożone ViT i VAE nie zbierają
gradientów parametrów.

Domyślnie co 8 aktualizacji, na ostatnim microbatchu i jednym przykładzie na GPU:
dodatkowy timestep <=250, estymacja x0, różniczkowalny frozen VAE decode i naturalny
frozen ViT forward obrazu. Strata = .01 x (zgodność latentu + .1 x strata detekcji).
Zgodność to średnia po tokenach z sumy kwadratów błędu / energii naturalnego
tokenu (min 1e-6), zachowuje informację o amplitudzie. Detekcja to istniejący
CenterNet/FCOS loss ze snapshotu dla `has_tool=True`, bbox przeskalowane do 480x320.
Gałąź pomocnicza zachowuje gradient obrazu do UNet i adaptera, mimo zamrożenia
parametrów ViT/VAE. Jedno wywołanie DDP UNet łączy główną i pomocniczą paczkę,
aby uniknąć wielokrotnego oznaczania tych samych parametrów gotowymi przez DDP.
`--aux-every 0` wyłącza gałąź; podczas smoke jest wymuszana w obu krokach, jeśli >0.

## Uruchomienie przez rodzica

Ścieżki poniżej to argumenty istniejącego środowiska, kod niczego nie pobiera.
`--sd-root` wskazuje BEZPOŚREDNIO folder EMA020000 z config.json i wagami UNet;
`--vae-root` osobno wskazuje sd-vae-ft-mse. Wagi muszą być lokalne.

```bash
torchrun --standalone --nproc_per_node=4 run.py \
  --data-manifest /path/FILMY.json \
  --sd-root /path/old/wyniki/trening/ema_020000 \
  --vae-root /path/old/modele/sd-vae-ft-mse \
  --vit-checkpoint /path/MODEL_VIT_K1000.pt \
  --output /path/wyniki/smoke --steps 2 --smoke --batch-size 2 --workers 2

# Dopiero po SMOKE.json passed=true, w osobnym output (nie wznawia smoke):
torchrun --standalone --nproc_per_node=4 run.py \
  --config config.json --data-manifest /path/FILMY.json \
  --sd-root /path/old/wyniki/trening/ema_020000 \
  --vae-root /path/old/modele/sd-vae-ft-mse \
  --vit-checkpoint /path/MODEL_VIT_K1000.pt \
  --output /path/wyniki/trening --steps 20000 --batch-size 2 --workers 2
```

CLI nadpisuje config. W manifeście potrzebne są TRAIN 118 i osobne VAL 25;
builder `--include-val` zachowuje ich istniejące etykiety. Alternatywnie
`--eval-manifest /path/VAL.json`. Nie tworzymy nowych splitów. Nieznane 27 filmów
nie trafia do TRAIN. Trening odmawia startu przy przecięciu splitów lub braku VAL.

Ocena to wyłącznie skalary DDPM epsilon MSE na held-out VAL, per-film i średnie,
bez generowania podglądów. Domyślnie 32 paczki na rank; `--eval-batches 0` ocenia
wszystkie VAL. Ograniczona ocena może nie obejmować każdego filmu i jest jawnie
oznaczona jako truncated; MSE odszumiania nie jest miarą jakości wygenerowanych obrazów.

## Checkpointy i czas

`checkpoint_XXXXXXXX.pt` zawiera tylko uczone wagi, AdamW, krok, RNG każdego ranka,
liczbę zużytych paczek i sygnaturę wejść. Nigdy ViT, VAE, zamrożone wagi ani dane.
Zostają najwyżej dwa opublikowane checkpointy; zapis tymczasowy fsync + os.replace,
po publikacji usuwany starszy. Chwilowo miejsce na trzeci checkpoint jest konieczne.
Brak EMA oraz eksportów kolejnych pełnych UNet. W trybie cross-attention checkpoint
jest znacznie mniejszy od pełnego UNet+optimizer. Dokładny rozmiar w LATEST.json,
liczba parametrów w CONFIG.json. Full mode wymaga większego RAM, VRAM i miejsca.

Auto-resume używa najwyższego opublikowanego numeru; pliki .tmp są ignorowane.
Wznawia dokładne uczone parametry/AdamW/RNG; deterministyczny iterator jest
odtwarzany przez ponowne dekodowanie i pominięcie zużytych paczek, bez cache.
To może długo trwać dla późnego checkpointu. Liczba ranków, workerów, batch,
akumulacja, seed, straty, źródłowe wagi i manifesty muszą pozostać takie same.
Można zmienić końcową liczbę kroków, limit czasu i częstotliwość logów/oceny/zapisu.
Powtarzane lokalne shardy mogą kończyć epoki w różnych chwilach; żaden film
nie jest usuwany ze zbioru, ale na ograniczonym biegu częstość ekspozycji zależy
od długości filmów. Dataset nie zapewnia globalnie równomiernego samplingu klatek.

SIGTERM/SIGINT/SIGUSR1, plik STOP lub `--max-hours 23.5` zatrzymują na granicy
aktualizacji z zapisem. Budżet czasu liczymy od uruchomienia procesu, razem z replay;
hashowanie/wczytanie/replay i ostatnia ocena/zapis też wymagają zapasu w SLURM.
Zapis jest synchroniczny; przy błędzie ranka torchrun powinien zakończyć pozostałe.

## Smoke i zależności

`--smoke` używa PRAWDZIWYCH danych, pełnego pretrained UNet/VAE i oryginalnego
ViT, wykonuje 2 aktualizacje DDP. Sprawdza skończone i niezerowe gradienty adaptera
i UNet, brak gradientów parametrów ViT/VAE, wpływ zerowania warunku i przestawienia
tokenów na predykcję, checkpoint z optimizerem i identyczną predykcję po odczycie.
Każdy rank raportuje liczby do SMOKE.json. Potrzebny nowy output.

Wymagane: Python >=3.10, PyTorch >=2.6 (BF16, SDPA, NCCL na Linux), diffusers 0.39.0
(wersja zapisana w źródłowym UNet), timm 0.9.16 (oryginalny model), numpy,
opencv-python, av, huggingface-hub, safetensors oraz zależności diffusers.
Żadne PEFT, CLIP/text encoder, EMA, Accelerate launcher ani biblioteki SSH nie
są potrzebne do algorytmu. `from_pretrained` może używać zainstalowanego accelerate
do oszczędnego ładowania, ale kod nie wymaga go jako launcher.

Rdzeń cross-attention opiera się na nazwach `attn2` istniejącego UNet diffusers.
Oficjalna dokumentacja:
https://huggingface.co/docs/diffusers/api/models/unet2d-cond

Stan: kod przygotowany do smoke na Eden przez rodzica. Kontrola składni i helperów
nie dowodzi działania pełnego modelu na 4 GPU; rozstrzyga to SMOKE.json z Eden.

## API generowania w RAM

Uruchamiane ze ścieżką tego katalogu w sys.path:

```python
from generate import load_trained, generate
trained = load_trained(sd_root, vae_root, checkpoint, device='cuda')
# condition: Tensor[B,600,768], naturalny albo sztuczny, już w RAM.
rgb = generate(trained, condition, steps=50, seed=42)
# Równoważnie trained.generate(condition, steps=50, seed=42).
# rgb: float32 Tensor[B,3,384,576], [0,1], na tym samym urządzeniu.
```

Zasiew dotyczy początkowego szumu Bx4x48x72; DDIM eta=0. Nie ma CFG, dodatkowego
tekstu ani wejściowych masek. Wczytanie sprawdza hash bazowego UNet i VAE zgodnie
z checkpointem; odczytany optimizer nie jest zatrzymywany w obiekcie inference.
Sztuczny warunek poza rozkładem naturalnych aktywacji nie ma gwarancji rekonstrukcji.
Funkcje nigdy nie zapisują obrazu/latentu/modelu na dysk.
