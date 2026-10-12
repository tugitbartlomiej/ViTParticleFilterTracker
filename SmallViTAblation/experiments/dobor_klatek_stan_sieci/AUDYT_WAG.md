# Audyt wag i krzywych walidacji — 22 września 2026

Sprawdzenie wykonano tylko do odczytu. Checkpointy ładowano na CPU, bez uruchamiania modelu ani GPU. Ten dokument zapisuje wynik wcześniej wykonanej kontroli; jego utworzenie nie uruchamia jej ponownie.

## Potwierdzone fakty

- **5/5 ramion:** `last.pth` ma `epoch=9` i `checkpoint_role=last_epoch_regular_weights_only`. Zbiór kluczy oraz **każdy tensor** `model` są identyczne z `epoch_checkpoints/epoch_009.pth` (`torch.equal`).
- **25/25 checkpointów okresowych:** `checkpoint_role=periodic_regular_weights_only`. Pliki `epoch_001/003/005/007/009.pth` odpowiadają ukończeniu epok **2/4/6/8/10**. Pole `epoch` jest indeksowane od zera.
- Zwykłe wagi są eksportowane z `state_dict` Lightninga przez wybór kluczy `model.*`. Stan EMA callbacka nie jest wybierany. Zgodność zwykłych wag wynika z kodu eksportu i cyklu callbacków; równość `last` z ostatnim checkpointem potwierdzono dodatkowo tensorowo.
- **5/5 konfiguracji efektywnych:** `use_ema=true`, `eval_base_model=false`, `epochs=10`, `checkpoint_interval=2`, `eval_interval=1`, `best_model_metric=map`.
- **50/50 zapisów epok:** w rzeczywistych `training_logs/**/metrics.csv` zachodzi dokładna równość liczb `val/ema_mAP_50_95 == val/mAP_50_95 == epoch_val_map.json[epoch]`. Krzywa to **AP EMA z walidacji treningowej**, nie maksimum regular/EMA ani krzywa zwykłych checkpointów.
- Końcowy wrapper niezależnie ocenia **`last.pth`**, czyli zwykłe wagi. `best.pth` jest zachowany, ale nie stanowi głównego wyniku tego pilota.

| Ramię | Krok po ep. 2 | po ep. 4 | po ep. 6 | po ep. 8 | po ep. 10 / last |
|---|---:|---:|---:|---:|---:|
| `random` (1500) | 188 | 376 | 564 | 752 | 940 |
| `readout_loss` (1500) | 188 | 376 | 564 | 752 | 940 |
| `koopman_langevin` (1500) | 188 | 376 | 564 | 752 | 940 |
| `placebo` (1500) | 188 | 376 | 564 | 752 | 940 |
| `random_large` (3000) | 376 | 752 | 1128 | 1504 | 1880 |

To rzeczywiste `global_step` zapisane w checkpointach. Wykres ekspozycji używa osobnego przybliżenia `ukończone_epoki × liczba_klatek`, nie tych kroków.

## Źródła lokalne

Repozytorium: `F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation`.

- [pipeline/train/train_rfdetr.py](../../pipeline/train/train_rfdetr.py), linie 505–536: eksport końcowych zwykłych wag z `last.ckpt` i zapis `last.pth`; linie 562–582: eksport okresowych zwykłych wag; linie 595–608: odczyt historii z pierwszego niepustego klucza `val/*mAP_50_95` w CSV.
- [trening.py](trening.py), linie 49–52: checkpoint co dwie epoki; linie 60–68: kontrola pięciu indeksów i hashy; linie 69–75: jawna niezależna ewaluacja `last.pth` na val.

## Źródła zainstalowane na Edenie

Wersja potwierdzona przez `importlib.metadata.version('rfdetr')`: **1.10.1**.

Środowisko: `/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1`.

Katalog pakietu: `/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/lib/python3.11/site-packages/rfdetr`.

- `training/trainer.py:320–345`: zwykłe callbacki Lightning `ModelCheckpoint` zapisują `last.ckpt` i `checkpoint_{epoch}.ckpt`.
- `training/module_model.py:1036–1070`: walidacja kieruje forward do osobnego modelu EMA, gdy `eval_base_model=false`; nie podmienia wag modelu uczonego.
- `training/callbacks/coco_eval.py:475–481`: predykcje trafiają do metryki EMA. Linie 659–691 i 805–812: wynik EMA jest kopiowany także pod główne klucze, w tym `val/mAP_50_95`.
- `training/callbacks/ema.py:277–280`: dopiero `on_train_end` pozostawia model w pamięci w stanie EMA. Eksport zwykłych wag z zapisanych checkpointów omija tę podmianę.

Rzeczywiste artefakty odczytano pod:

```text
/home2/faculty/bpiotrowski/smallvit_runs/2026-09-22_dobor_klatek_stan_sieci/pilot/results/
  stan_<ramię>_b<1500_lub_3000>_sel42_s42/
```

## Wykonane kontrole

Poniżej fragment wykonanego skryptu kontroli tensorów, z pominięciem końcowego wypisywania kodu źródłowego pakietu. Polecenie uruchomiono z PowerShell przez SSH. `weights_only=False` służyło do odczytu zaufanych checkpointów utworzonych przez ten pilot; `map_location='cpu'` wymuszało CPU.

```powershell
@'
from pathlib import Path
import json
import torch
torch.set_num_threads(1)
root=Path('/home2/faculty/bpiotrowski/smallvit_runs/2026-09-22_dobor_klatek_stan_sieci/pilot/results')
for run in sorted(root.iterdir()):
 last_path=run/'last.pth'
 if not last_path.exists(): print(run.name,'NOT_READY'); continue
 last=torch.load(last_path,map_location='cpu',weights_only=False)
 records=[]; equality=None
 for p in sorted((run/'epoch_checkpoints').glob('epoch_*.pth')):
  payload=torch.load(p,map_location='cpu',weights_only=False)
  records.append({'file':p.name,'epoch':payload.get('epoch'),'role':payload.get('checkpoint_role'),'global_step':payload.get('global_step')})
  if payload['epoch']==9:
   equality=set(last['model'])==set(payload['model']) and all(torch.equal(last['model'][k],payload['model'][k]) for k in last['model'])
  del payload
 print(json.dumps({'run':run.name,'last_epoch':last.get('epoch'),'last_role':last.get('checkpoint_role'),'last_global_step':last.get('global_step'),'last_source_sha':last.get('source_checkpoint_sha256'),'periodic':records,'last_equal_epoch009_all_tensors':equality}),flush=True)
 del last
'@ | ssh -o BatchMode=yes eden-cluster /home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/bin/python -
```

Odrębny wykonany skrypt wypisał dla każdego rzeczywistego rekordu CSV: indeks epoki, niepuste klucze `val/*mAP_50_95`, ich wartości oraz wartość z `epoch_val_map.json`. Porównanie wszystkich 50 wypisanych rekordów dało zgodność opisaną wyżej. Istotny fragment tej kontroli:

```python
import json, csv
for run in sorted(root.iterdir()):
 history_path=run/'epoch_val_map.json'
 if not history_path.exists(): continue
 h=json.loads(history_path.read_text())['per_epoch_map_50_95_val']
 metricfiles=list((run/'training_logs').rglob('metrics.csv'))
 selected=[]
 for p in metricfiles:
  with p.open() as stream:
   for row in csv.DictReader(stream):
    keys=[k for k in row if k.startswith('val/') and 'mAP_50_95' in k and row[k]]
    if keys: selected.append({'epoch':row.get('epoch'),'keys':keys,'values':[row[k] for k in keys], 'history':h.get(str(int(float(row['epoch']))))})
 print('ACTUAL_LOG',run.name,selected)
```

Audyt rozstrzyga pochodzenie wag i metryk; sam nie ocenia przewagi selektora ani istotności statystycznej wyników AP.
