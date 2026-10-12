# Mały ViT po 1000 aktualizacjach — model warunkujący SD

Plik: `MODEL_VIT_K1000.pt`.

To oryginalny checkpoint detektora narzędzia MalyViT po **1000 aktualizacjach**. Jego naturalne pełne latenty były używane podczas treningu Stable Diffusion na filmach na Edenie w eksperymencie `2026-10-06_vit_sd_pelny_latent_filmy`. Nie jest to checkpoint RF-DETR Nano z 21. epoki ani późniejszy model dopasowany przez Lorentza do jednego obrazu.

## Architektura i format

Klasa: `MalyViT`, źródło w projekcie: `experiments/vit_sd_pelny_latent_filmy_eden_20261006/model_snapshot.py`.

Mierzona reprezentacja po końcowym GELU: **600 tokenów × 768 współrzędnych** dla obrazu wejściowego **480 × 320**. Wagi tego konkretnego checkpointa definiują przestrzeń reprezentacji, której uczył się adapter SD. Plik nie zawiera sztucznie podmienionych aktywacji.

Format kontenera PyTorch: `net` zawiera state_dict, `krok` wynosi 1000. Ładowanie:

```python
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
assert checkpoint["krok"] == 1000
model.load_state_dict(checkpoint["net"], strict=True)
model.eval()
```

## Pochodzenie i integralność

Poprzednia lokalna ścieżka:

```text
outputs/vit_latent128_jedno_zdjecie_90/MODEL_VIT_K1000.pt
```

Kopia wejściowa użyta na Edenie:

```text
/home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_vit_sd_pelny_latent_filmy/wejscia/MODEL_VIT_K1000.pt
```

Przeniesiono do assets 6 października 2026. Pod dawną ścieżką lokalną pozostawiono hardlink do tego samego pliku, aby działały istniejące skrypty oraz historyczne manifesty. Nie zajmuje on miejsca drugiej kopii modelu. Checkpoint traktować jako niezmienny; przyszłe modele zapisywać pod inną nazwą.

- Rozmiar: **11 730 541 bajtów**.
- SHA256:

```text
3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e
```

Hash odpowiada zapisowi `HASHES.json` wcześniejszego eksperymentu i provenance checkpointa SD. Potwierdzono zgodność po przeniesieniu.

## Wyniki i ograniczenia

Na pojedynczym obrazie TR_0154_S1_P03, klatka 1926, zapisany naturalny wynik przed ingerencją wynosił confidence **86,79%**, IoU **85,66%**. To pomiar jednego obrazu, nie mAP zbioru. Nie przypisywać temu checkpointowi AP50 61,12% — ten wynik dotyczy osobnego modelu po 8000 krokach.

Sztuczny latent zapisany wcześniej w `outputs/vit_latent128_jedno_zdjecie_90/LATENT_PELNY.npz` uzyskano przez ingerencję w aktywacje przy zamrożonych wagach tego modelu. Poprawa po podmianie latentu nie jest wynikiem naturalnego działania checkpointa.

Do kontrolnego testu SD należy obliczać naturalne latenty właśnie tym modelem. Inne checkpointy MalyViT mają tę samą architekturę, lecz nie mają automatycznie potwierdzonej zgodności reprezentacji z adapterem SD.
