# SESSION SUMMARY — 2026-07-23 14:34:35

Sesja zapisana rownolegle w dwoch miejscach:
- `F:\Studia\Articles\Moj\IEEE\Overleaf\DETR_IEEE\.sessions\`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\`

## Cel sesji
Zwolnic miejsce na koncie Eden przez skasowanie artefaktow treningowych YOLO —
ale dopiero po twardym potwierdzeniu, ze wszystko jest w lokalnej kopii
zapasowej. Sesja dokumentuje: co zweryfikowano, co pobrano, co usunieto z
klastra i ile miejsca zostalo.

## Kontekst

### Czym jest folder "add&backup"
Lokalny folder kopii zapasowej Edenu:

```text
F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\EdenBackup
```

Do niego kopiowane byly WSZYSTKIE artefakty treningowe z klastra: checkpointy
YOLO (wszystkie dotychczas wytrenowane wagi), checkpointy DETR, logi SLURM, logi
treningu, wykresy `results.png`, `predictions.json`, `results.csv`, `args.yaml`,
podglady `train_batch*.jpg` / `val_batch*.jpg` oraz skrypty `.slurm` i `.py`.
Kopia powstala 2026-07-22 (manifesty w `EdenBackup\_manifests\`, stemple
`20260722_0431` do `20260722_0434`).

### Procedura
Kasowanie na Edenie idzie katalog po katalogu i tylko dla pozycji wczesniej
potwierdzonych jako skopiowane. Zadna pozycja nie jest usuwana bez porownania.

---

## CO ZOSTALO POBRANE DO BACKUPU

Stan `EdenBackup` na 2026-07-23:

| Katalog | Plikow | Rozmiar | Status weryfikacji |
|---|---|---|---|
| `DETR` | 4748 | 39,80 GB | tylko manifest, BRAK porownania 1:1 |
| `Yolo` | 344 | 20,25 GB | **porownanie 1:1 z Edenem, zero roznic** |
| `archives` | 3 | 23,85 GB | tylko manifest, sha256 niesprawdzone |
| `models` | 3 | 0,63 GB | tylko manifest |
| `YOLO_Checkpoints` | 1 | 0,16 GB | **md5 zgodne bit w bit** |
| `YOLO_20k` | 4 | 33 KB | **porownanie 1:1, rozmiary zgodne** |
| `=` | 409 | 60,28 GB | artefakt bledu, patrz "Otwarte problemy" |

### Zawartosc czesci YOLO (istotne dla odtwarzalnosci artykulu)

- `Yolo\train_out_5gpu_hopper\exp\weights\` — pelny bieg bazowy: `best.pt`,
  `last.pt`, `emergency_checkpoint.pt`, epoki co 10 od 0 do 190 (23 pliki).
- `Yolo\20k_finetune\final_ep170_lr0001\` (13 wag),
  `final_ep170_to_500\` (8), `final_scratch_90k_vs\` (21, epoki 0-180),
  `train_out_20k_finetune_short\` (8), `train_out_resume_ep170\` (1).
- Modele bazowe `yolov8l.pt`, `yolo11n.pt`; model inferencyjny
  `models\YOLO\yolo_inference_model_final\yolo_inference_model.pt`;
  `YOLO_Checkpoints\epoch170.pt`.
- Wszystkie `results.csv`, `predictions.json`, `args.yaml`, krzywe
  `PR_curve.png` / `F1_curve.png` / `P_curve.png` / `R_curve.png`, macierze
  pomylek, 19 logow z `20k_finetune\logs`, logi z `Yolo\logs` i `Yolo\out`,
  oraz `Yolo\Ablation\` z `yolo_ep170_el2n_scores.json`.

---

## CO ZOSTALO USUNIETE Z EDENU

W trakcie tej sesji, po weryfikacji, usunieto z
`/mnt/evafs/faculty/home/bpiotrowski/`:

| Katalog | Rozmiar | Podstawa decyzji |
|---|---|---|
| `Yolo` | 21 GB | porownanie 1:1 — 344 pliki i 20,25 GB po obu stronach, zero roznic |
| `YOLO_Checkpoints` | 167 MB | md5 `4744db6d9c86ba99d7dcc85879d9614b` zgodne po obu stronach |
| `YOLO_20k` | 48 KB | 4 pliki, rozmiary co do bajta zgodne |

Lacznie zwolniono okolo 21,2 GB.

### Stan konta Eden po kasowaniu (2026-07-23, ~15:00)

```text
quota:  135,7 GB uzyte, 128 658 plikow (brak twardego limitu na koncie)
df:     200G total / 136G used / 65G avail / 68% na /mnt/evafs
```

Co zostalo w home:

| Pozycja | Rozmiar |
|---|---|
| `DETR` | 40 GB |
| `datasets_20250606.tar.gz` | 39 GB |
| `archives` | 24 GB |
| `20kSelectedImages.tar` | 8,5 GB |
| `models` | 642 MB |
| `prom_ai` | 217 MB |
| reszta | ponizej 50 MB lacznie |

---

## Wykonane kroki

1. **Weryfikacja wzgledem manifestu.** Przejsto po
   `_manifests\manifest_all_20260722_434.txt` (413 pozycji), sprawdzajac
   istnienie i rozmiar bajtowy kazdego pliku. Wynik: 410 OK, 0 z niezgodnym
   rozmiarem, 3 pozycje nieobecne w korzeniu (`merged_20k_annotations.json`,
   `merged_20k_annotations_fixed.json`, `slurm-1618300.out`) — wyladowaly w
   katalogu `=`, wiec fizycznie sa.

2. **Nieudane transfery.** `failed_all_20260722_434.txt` wymienia 4 pozycje,
   wszystkie nieistotne: trzy pliki
   `DETR/external_benchmark/*/_annotations.coco — kopia.json` (duplikaty;
   oryginaly obecne, transfer poleglal na myslniku w nazwie) oraz
   `DETR/logs/ablation_4294967294_1669339.log`.

3. **Porownanie 1:1 katalogu `Yolo`.** Zdalny listing
   (`find . -type f -printf '%s|%p\n'`) porownany ze zbiorem sciezek i rozmiarow
   lokalnych:

   ```text
   Eden:     344 plikow, 20,25 GB
   Lokalnie: 344 plikow, 20,25 GB
   Brak lokalnie: 0 | Tylko lokalnie: 0 | Rozny rozmiar: 0
   ```

4. **Porownanie `YOLO_20k`.** Cztery pliki po obu stronach:
   `launch_tensorboard.sh` 1355 B, `run_yolo_20k_finetune_dgx4_2gpu.slurm`
   8545 B, `yolo_train_20k_finetune.py` 17299 B,
   `logs/yolo_20k_finetune_1468484.log` 667 B.

5. **Porownanie `YOLO_Checkpoints` z md5.** Jeden plik `epoch170.pt`,
   175 036 386 B, md5 `4744db6d9c86ba99d7dcc85879d9614b` identyczne po obu
   stronach. Jedyny katalog zweryfikowany hashem, nie samym rozmiarem.

6. **Kasowanie zrodel** — patrz tabela wyzej.

7. **Identyfikacja `epoch170.pt`** (osobne ustalenie, patrz nizej).

---

## Najwazniejsze ustalenia

### Cala czesc YOLO jest zabezpieczona
344 pliki, komplet wag i logow. Zrodla skasowane po weryfikacji.

### Trzy pozorne luki to stan klastra, nie blad kopiowania
Potwierdzone bezposrednim listingiem Edenu:
- `Yolo/20k_finetune/train_out_ep200_to_500/` — tylko `args.yaml` i podglady
  `train_batch*.jpg` / `labels.jpg`, zero wag. Bieg nie zapisal checkpointow.
- `Yolo/train_out_8gpu/exp/weights/` — tylko `best.pt`, `epoch0.pt`, `last.pt`,
  wszystkie o identycznym rozmiarze 175 014 690 B. Brak epok posrednich.
- `YOLO_20k/` — sam kod i log na 667 B. Wagi z tego eksperymentu leza w
  `Yolo/20k_finetune/`.

### `epoch170.pt` to NIE jest trening video-split

Ustalenie istotne, bo ten plik latwo pomylic z wynikiem leakage-free.

Cztery pliki maja identyczne md5 `4744db6d...`:
`YOLO_Checkpoints\epoch170.pt`, `Yolo\train_out_5gpu_hopper\exp\weights\epoch170.pt`,
`Yolo\20k_finetune\epoch170.pt`, `Yolo\20k_finetune\ckpt_resume_ep170_final\last.pt`.
Checkpointy video-split maja inny hash (`c2ee2e2f...`), wiec to inny model.

Zrodlem jest bieg bazowy `train_out_5gpu_hopper` z 2025-06-09
(`args.yaml` + `out\yolo_8gpu_python_1030337.out`):

| Parametr | Wartosc |
|---|---|
| Dataset | `unified_dataset`, 91 260 obrazow, 75 867 sparowanych par |
| Podzial | LOSOWY, KLATKOWY — 60 693 / 11 380 / 3794 (80/15/5) |
| Start | `yolov8l.pt` (wagi COCO), nie od zera |
| Budzet | 200 epok, batch 96, 8 GPU |
| Wynik koncowy | mAP50 0,995 / mAP50-95 0,9497 |

To jest zbior NIEWYCZYSZCZONY, z losowym podzialem klatkowym — czyli ten z
wyciekiem miedzy klatkami tego samego wideo. Stad zawyzone metryki.

**Po co ten plik istnieje osobno:** `epoch170.pt` jest punktem startowym dla
fine-tune na wyselekcjonowanych 20k. W `args.yaml` biegow
`train_out_ep170_lr0001` i `train_out_ep170_to_500` pole `model` wskazuje wprost
na `20k_finetune/epoch170.pt`.

**Trening video-split to inny katalog:** `Yolo\20k_finetune\train_out_scratch_90k_vs\`
oraz kopia `final_scratch_90k_vs\`. `data` wskazuje na
`/tmp/bpiotrowski_1729466_yolo90k_vs/`, budzet 190 epok, job 1729466 wznowiony
jako 1733562 (sesja z 2026-07-12). Checkpointy do epoki 180 plus `best.pt`
i `last.pt`.

---

## Pliki zmienione / utworzone

- **Eden (zdalnie):** usuniete `~/Yolo` (21 GB), `~/YOLO_Checkpoints` (167 MB),
  `~/YOLO_20k` (48 KB).
- **Lokalnie:** bez zmian w `EdenBackup` — sesja byla weryfikacyjna.
- **Ta sesja:** `README.md` + `SESSION_SUMMARY.md`, zapisane w dwoch
  lokalizacjach (DETR_IEEE i ViTParticleFilterTracker).
- Artykul `.tex` i pliki repo DETR_IEEE: NIE zmieniane.

## Otwarte problemy

1. **`DETR` (40 GB) nie jest porownany 1:1 z Edenem.** Zweryfikowany tylko
   wzgledem manifestu z 2026-07-22. Przed skasowaniem `~/DETR` nalezy powtorzyc
   procedure z pkt. 3. To samo dotyczy `models` i `archives`.

2. **Katalog `=` w `EdenBackup` — 60,28 GB zdublowanych danych.** Skutek bledu
   skladni w komendzie kopiujacej (argument zinterpretowany jako katalog
   docelowy). Zawiera pelny drugi egzemplarz `DETR`, `Yolo`, `models`,
   `YOLO_20k`, `YOLO_Checkpoints`. UWAGA: trzy pliki istnieja WYLACZNIE tam —
   `merged_20k_annotations.json`, `merged_20k_annotations_fixed.json`,
   `slurm-1618300.out`. Najpierw przeniesc je do korzenia, dopiero potem usunac `=`.

3. **Suma kontrolna archiwum JEPA niesprawdzona.**
   `archives\JEPA_2026-04-25.tar.gz` (23,85 GB) ma obok plik `.sha256`, ktory nie
   zostal zweryfikowany po skopiowaniu.

4. **Dla `Yolo` weryfikowano rozmiary, nie hashe.** Zrodlo juz nie istnieje, wiec
   tej kontroli nie da sie wykonac wstecz. Dla `YOLO_Checkpoints` md5 policzono
   przed skasowaniem.

## Nastepne kroki

1. Porownac 1:1 `~/DETR` z `EdenBackup\DETR` PRZED kasowaniem — to najcenniejsza
   pozostala czesc (40 GB, ~90 checkpointow). Rozwazyc md5 po obu stronach.
2. To samo dla `models` i `archives`.
3. Uratowac trzy unikalne pliki z katalogu `=`, potem usunac `=` i odzyskac
   60 GB na dysku F.
4. Zweryfikowac `JEPA_2026-04-25.tar.gz` wzgledem jego `.sha256`.
5. Rozwazyc, czy `datasets_20250606.tar.gz` (39 GB) i `20kSelectedImages.tar`
   (8,5 GB) maja odpowiednik lokalny — to razem 47,5 GB na Edenie, a nie ma ich
   w `EdenBackup`.
