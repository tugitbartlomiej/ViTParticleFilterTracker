# Session 2026-07-23 14:34:35 — EdenBackup: weryfikacja kopii YOLO i kasowanie zrodel

## Typ sesji
`results` / infrastruktura. Artykul (.tex) NIE byl dotykany.

Sesja zapisana rownolegle w `DETR_IEEE\.sessions\` oraz
`ViTParticleFilterTracker\.sessions\`.

## Cel sesji
Potwierdzic, ze lokalna kopia zapasowa (folder "add&backup", fizycznie
`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\EdenBackup`) zawiera
komplet checkpointow, logow i wynikow z Edenu — a nastepnie skasowac
zweryfikowane zrodla z klastra.

## Workerzy
Brak — praca jednoagentowa (Windows, solo).

## USUNIETE Z EDENU

| Katalog | Rozmiar | Podstawa |
|---|---|---|
| `~/Yolo` | 21 GB | porownanie 1:1, 344 pliki, zero roznic |
| `~/YOLO_Checkpoints` | 167 MB | md5 zgodne bit w bit |
| `~/YOLO_20k` | 48 KB | 4 pliki, rozmiary zgodne |

Zwolniono ~21,2 GB. Stan konta po operacji: **136 GB / 200 GB, wolne 65 GB**.

## POBRANE DO BACKUPU (stan kopii)

| Katalog | Plikow | Rozmiar | Weryfikacja |
|---|---|---|---|
| `DETR` | 4748 | 39,80 GB | tylko manifest |
| `Yolo` | 344 | 20,25 GB | 1:1, zero roznic |
| `archives` | 3 | 23,85 GB | tylko manifest |
| `models` | 3 | 0,63 GB | tylko manifest |
| `YOLO_Checkpoints` | 1 | 0,16 GB | md5 |
| `YOLO_20k` | 4 | 33 KB | 1:1 |
| `=` | 409 | 60,28 GB | artefakt bledu kopiowania |

## Kluczowe ustalenia

1. **Cala czesc YOLO jest zabezpieczona** — komplet wag, logow i wynikow.
2. **`epoch170.pt` to NIE trening video-split.** To epoka 170 biegu bazowego
   `train_out_5gpu_hopper` (2025-06-09) na `unified_dataset` z losowym podzialem
   klatkowym. Sluzyl jako inicjalizacja dla fine-tune na 20k.
   Video-split to `train_out_scratch_90k_vs`.
3. **Braki wag w `train_out_ep200_to_500`, `train_out_8gpu` i `YOLO_20k`** to
   stan po stronie klastra, nie luka w kopiowaniu.

## Do zrobienia dalej
Porownac 1:1 `~/DETR` (40 GB) PRZED skasowaniem. Uratowac 3 unikalne pliki z
katalogu `=` i odzyskac 60 GB na dysku F. Zweryfikowac sha256 archiwum JEPA.
