# Konwerter latent ViT -> DINOv2 na filmach (Eden)

Plan, kontrole i kryteria: `PLAN.md`. Wyniki: `outputs/vit_konwerter_dino_filmy_eden` (lokalnie) oraz
`~/smallvit_runs/2026-10-06_vit_konwerter_dino_filmy/wyniki/{smoke,trening}` na Edenie.

## Pliki

- `dane.py` - strumień klatek MP4 z ZIP do RAM, trzy obrazy na klatkę (ViT 480x320, DINO CLS 224, DINO tokeny 420x280).
- `model.py` - konwerter, kontrola liniowa, wczytanie ViT i DINOv2 z kontrolą SHA256, `KonwerterGotowy` do użycia po treningu.
- `model_vit.py` - niezmieniona kopia `MalyViT` (ta sama co w `vit_sd_pelny_latent_filmy_eden_20261006/model_snapshot.py`).
- `run.py` - uczenie DDP (torchrun), test dymny `--smoke`, ocena VAL, checkpointy z wznowieniem, STOP/SIGTERM.
- `config.json`, `trening.slurm` - 4 x A100, partycja `short`; najpierw test dymny, uczenie tylko po `passed=true`.
- `deploy.py` - wysyłka kodu, twarde dowiązania wejść, kontrola środowiska; `--submit` zgłasza zadanie.

## Lokalny test dymny

```bash
MSYS_NO_PATHCONV=1 F:/Python/python.exe -X utf8 experiments/vit_konwerter_dino_filmy_eden/run.py \
  --manifest FILMY_smoke.json --zamiana "/mnt/evafs/faculty/home/bpiotrowski/smallvit_data/cataract_lmm/3_Object_Tracking=F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Dataset/Cataract-LMM/3_Object_Tracking" \
  --vit-checkpoint outputs/vit_latent128_jedno_zdjecie_90/MODEL_VIT_K1000.pt \
  --dino-repo C:/Users/bartl/.cache/torch/hub/facebookresearch_dinov2_main \
  --dino-weights C:/Users/bartl/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth \
  --output <katalog> --smoke --steps 30 --batch-size 16 --workers 2 --bufor 64 --stats-batches 3 --eval-krok 60 --eval-max-filmow 2
```

2026-10-06, RTX 3070, 2 filmy TRAIN i 2 VAL: `passed=true`, TF32 kontra FP32 cosinus CLS 0,999999,
tokenów 0,99992, błąd względny latentu 0,0003; ok. 100 klatek/s na jednym GPU. Jeden proces dekodujący: ok. 115 klatek/s.

## Użycie po treningu

```python
from model import KonwerterGotowy
k = KonwerterGotowy('KONWERTER.pt', device='cuda')
cls, tokeny = k.predict(latent)   # latent: N x 600 x 768 -> CLS N x 384 (L2), tokeny N x 600 x 384 (L2)
```

## Wyniki (2026-10-06, Eden dgx-2, 4 x A100 40 GB)

Lokalne kopie: `outputs/vit_konwerter_dino_filmy_eden/v1`, `.../v2` (CONFIG, HISTORIA, OCENA, SMOKE) oraz
`v2/KONWERTER.pt` (SHA256 `5230af4e...7112`, 46 MB). Checkpointy z optymalizatorem tylko na Edenie.

| bieg | zadanie | kroki | czas | uwagi |
|---|---|---|---|---|
| v1, bez augmentacji i regularyzacji | 1823170 | 10 120 z 20 000 | 1 h 06 min | zatrzymany plikiem STOP po przeuczeniu |
| v2, augmentacje natywnej klatki + dropout 0,1, drop-path 0,1, maska tokenów 0,25, wd 0,1 | 1823174 | 20 000 | 2 h 16 min | pełny harmonogram |

VAL: 25 filmów, co 15. klatka, 4435 klatek; galeria wyszukiwania = cele DINO tych samych klatek.

| miara VAL | v2 koniec | v1 najlepsza (krok 6000) | v1 koniec | kontrola liniowa v2 | średnia TRAIN |
|---|---|---|---|---|---|
| cosinus CLS | 0,710 | 0,645 | 0,634 | 0,737 | 0,829 |
| cosinus po odjęciu średniej TRAIN | **0,435** | 0,378 | 0,359 | 0,395 | 0 |
| mediana po filmach (wycentrowany) | **0,446** | 0,375 | - | 0,395 | - |
| mediana pozycji właściwej klatki | **101** | 227 | 303 | 190 | - |
| top1 / top5 | **1,5 / 6,3 %** | 0,7 / 3,0 % | 0,6 / 2,7 % | 0,7 / 3,1 % | - |
| top1 z tolerancją sąsiedniej próbki | 3,6 % | 1,8 % | - | 2,1 % | - |
| najbliższa klatka z właściwego filmu | **81 %** | 64 % | - | 68 % | 6 % losowo |
| cosinus tokenów DINO 20x30 | 0,781 | 0,772 | - | - | - |

Wnioski:

- v1 przeuczała się od kroku ~6000 (TRAIN cos 0,92, VAL coraz gorszy) i na końcu przegrywała z kontrolą liniową.
- v2 rosła na VAL do końca, bez przeuczenia; transformer na tokenach wyraźnie bije liniową głowę na uśrednionym
  latencie (mediana pozycji 101 wobec 190, top5 6,3 wobec 3,1 procent), więc układ przestrzenny latentu niesie
  informację o cechach DINO.
- Surowy cosinus CLS jest niższy niż sama średnia TRAIN (0,71 wobec 0,83): składnik InfoNCE odpycha predykcje od
  wspólnego kierunku. Do zastosowań wymagających bliskości w surowym cosinusie (np. FAISS IP) trzeba to sprawdzić.
- Konwerter dobrze trafia film i scenę (81 procent), ale dokładna klatka z ponad 4 tys. jest trafiana rzadko (1,5 procent).

Czego nie sprawdzono: zgodności z indeksem FAISS 35000 zbudowanym z JPEG-ów, działania na sztucznym latencie,
innych checkpointów ViT, drugiego ziarna. VAL nie służył do wyboru modelu (stały harmonogram), ale konfigurację v2
wybrano po obejrzeniu krzywej VAL v1, więc VAL nie jest już w pełni niezależnym testem dla porównania v1 i v2.
