# ViT Model Zoo — ViT-S/16 na ImageNet-1k (checkpointy co epokę)

Źródło: Falk, Meynent, Pfammatter, Schürholt, Borth, „A Model Zoo of Vision Transformers” (2025), https://doi.org/10.48550/arXiv.2504.10231,
repozytorium https://github.com/ModelZoos/ViTModelZoo, licencja CC BY 4.0 (wg artykułu, dodatek C.4).
Serwer z plikami (Uniwersytet St. Gallen, katalog HTTP z listingiem JSON):
`https://ai.unisg.ch/files/weight_space_learning/model_zoos/vit-modelzoo/vit_imagenet_pretrained/<przebieg>/checkpoint_XX/checkpoints`
(narzędzie `pip install wsl-modelzoo` → `modelzoo fetch` tylko składa te adresy i woła `wget`; pobieramy wprost `curl`).

## Co jest na serwerze (sprawdzone 2026-10-13)
- 10 przebiegów uczenia wstępnego, każdy 90 katalogów `checkpoint_00` … `checkpoint_89` (po jednym na epokę) + `config.json`, `result.json`:
  - nadzorowane, 1 karta: `sl_vit_s_16_224_seed_{42,6312}_bs_256_lr_0.001_wd_0.0001_me_90`
  - nadzorowane, DDP (paczka 512): `sl_vit_s_16_ddp_512_seed_{1957,2517,3753,6312,7853}_bs_256_lr_0.001_wd_0.0001_me_90`
  - kontrastowe (SimCLR): `cl_vit_s_16_seed_{42,2517,6312}_bs_{96,128}_lr_0.001_wd_0.0001_me_90`
- w każdym `checkpoint_XX` dwa pliki ~88 MB: `checkpoints` (sam `state_dict`) i `model_name=0-epoch=XX-…ckpt` (Lightning); pobieramy tylko `checkpoints`.
- jeden przebieg = 90 × 88 MB ≈ 7,9 GB; całe archiwum `_zips/zoos_vit.zip` (z 240 modelami dostrojonymi na CIFAR-100) ma 2,2 TB.
- `result.json`: lista 90 wpisów, epoki 1–90 (train_loss, val_loss, val_acc) → **`checkpoint_00` to stan po 1. epoce, nie inicjalizacja**
  (brak checkpointu z kroku 0). Seed 42: val_acc 0,047 po epoce 1, 0,675 po epoce 90.
- `config.json` (seed 42): vit_s_16, GELU, init kaiming_normal, AdamW lr 1e-3 stały (bez harmonogramu), wd 1e-4, paczka 256,
  label smoothing 0,1, MixUp 0,8, CutMix 1,0, RandomErasing 0,25, dropout 0; 1 epoka ≈ 5005 kroków.

## Format
`state_dict` w formacie timm z prefiksem `vit.` (`vit.blocks.N.mlp.fc1.weight` 1536×384, `fc2` 384×1536, `norm2`, `attn.qkv`, …),
22 050 664 parametrów, 12 bloków, d = 384, MLP 1536. Ładować wyłącznie `torch.load(..., weights_only=True)` (pliki z zewnątrz).

## Pobrane lokalnie (dysk F: ma ~28 GB wolnego — pełne przebiegi pobierać z głową albo liczyć na Edenie)
- `sl_vit_s_16_224_seed_42/`: `config.json`, `result.json`, `checkpoint_00`, `checkpoint_01`, `checkpoint_89` (2026-10-13, ~265 MB).
