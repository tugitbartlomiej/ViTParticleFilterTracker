# DeiT-Tiny na CIFAR-10: drugi model i drugi zbiór (2026-10-13)

**Po co.** Wszystkie wyniki układu laserów, pułapek i koloidów (opis: `outputs/porownanie_szklo_20261010/film_krysztal/opis_ukladu/OPIS_UKLADU_OPTYCZNEGO_VIT.pdf`)
pochodzą z jednego biegu MałegoViT (s47) na jednym zbiorze. Recenzent zapyta o drugi model i drugi zbiór oraz o to, czy klatki
i starzenie igieł nie są artefaktem obcięcia gradientu (w s47 działało w 100 % kroków). Ten eksperyment daje oba.

## Model i przepis
- **DeiT-Tiny** (timm `VisionTransformer`, jak `deit_tiny_patch16_224`): wymiar 192, 12 bloków pre-LN, 3 głowy, MLP 192 → 768 → 192, GELU,
  token CLS, uczone kodowanie pozycji, głowa na CLS. Pod CIFAR: **łatka 4**, siatka 8×8 = **64 tokeny** + CLS. Od zera, bez tokenu destylacji.
  Ten sam wymiar (192) i ta sama liczba neuronów MLP (768) co MałyViT, więc „768 reflektorów na blok” przenosi się wprost.
- **Przepis jak s47:** paczka 32, AdamW 5e-4, wd 0,05 (tensory o wymiarze ≥ 2: wagi, ale też CLS i kodowanie pozycji), rozgrzewka
  300 kroków, potem stały lr, bf16, odbicie 0,5, jasność/kontrast ±20 %, ziarno 47, 60 epok (1563 kroki na epokę, 93 780 kroków).
- **Dwa biegi:** `s47_obciecie` (obcięcie normy gradientu do 1, jak s47) i `s47_bez_obciecia` (`--obciecie 0`).
- **Trzeci bieg, MałyViT na CIFAR-10** (`maly_s47_obciecie`, `--model malyvit`): model użytkownika bez zmian (łodyga
  4 konwolucji z krokiem 2 = krok 16 px, sin-cos 2D, 6 bloków, głowa na średniej tokenów jak w biegach na Imagenette, 2 890 402
  parametry), obraz 32 px powiększony dwuliniowo ×4 do 128 px, więc też 64 tokeny. Przepis jak s47 z obcięciem, stan AdamW co 10 epok
  (`--stan_co 10`, miejsce), wagi ~12 MB. Rozdziela wpływ zbioru (s47 → MałyViT CIFAR) od wpływu modelu (MałyViT CIFAR → DeiT CIFAR).

- **Biegi kontrolne MałegoViT na CIFAR-10** (wagi rzadziej: co 10 do 1000, potem co 1000, `--wagi_co_po1k 1000 --wagi_co_po10k 1000`):
  `maly_s48_obciecie` (drugie ziarno), `maly_s47_wd0` (`--wd 0`: czy weight decay to sprężyna pułapki),
  `maly_s47_losowe_etykiety` (`--losowe_etykiety`: kontrola zerowa, czy klatki i starzenie biorą się z treści, czy z samej optymalizacji).
  Do zrobienia, gdy będzie miejsce: `maly_s49_obciecie`, `deit_s48`, `deit_s49`.

## Co się zapisuje (na bieg)
| ścieżka | zawartość |
|---|---|
| `wagi/k<krok>.pt` | wagi co 10 kroków do 1000, potem co 100, plus końce epok — jak s47 (~1030 plików × 21,5 MB ≈ 22 GB); bieg bez obcięcia po kroku 10 000 co 500 (`--wagi_co_po10k 500`, ≈ 8 GB) |
| `pelny/stan_eNN.pt` | wagi + **stan AdamW** (oba momenty) co 5 epok i 2 ostatnie — tego brakowało w s47 |
| `gesty/` | obserwator RP128: 24 sondy × 64 tokeny × 768 GELU MLP ostatniego bloku, co krok w 1. epoce, potem co 20 |
| `dane/NORMY_GRAD.npz` | norma gradientu **przed** obcięciem w każdym kroku |
| `dane/OCENY.json`, `LOG.json`, `metadata.json` | dokładność top-1 na zbiorze testowym, strata, przepis, SHA-256 kodu |

## Pliki
- `kod/model_deit_cifar.py` — model z interfejsem MałegoViT (`bloki`, `zapisuj_latent`, `forward -> (logity, latent)`).
- `kod/trening_cifar_deit.py` — trener (dane z oryginalnego `cifar-10-python.tar.gz`; reszta jak `trening_klasyfikacja.py`).
- `kod/deit_cifar.slurm` — zadanie na Edenie (hopper-2, H200).
- `kod/z_edena/` — kopie modułów z Edena do testów lokalnych (`trening.py`, `model.py`, `obserwator_gesty.py` z biegu s47;
  `trening_klasyfikacja.py`, `obserwator_rp.py`, `model_cnn.py` z biegów z pokrętłami). Na Edenie te same moduły bierze `PYTHONPATH`.
- `kod/przygotuj_cifar10.py` — CIFAR-10 z HuggingFace (`uoft-cs/cifar10`, parquet) → `cifar10_uint8.npz`
  (SHA-256 68e07d955b627c168b2265659ef8c62593c7548b33e1cb988ba924e24b466ada; 50 000 + 10 000 obrazów, po 5000/1000 na klasę).
  Oryginał z cs.toronto.edu szedł ~40 KB/s; gdy dojdzie, `--sprawdz cifar-10-python.tar.gz` porównuje piksele i etykiety.
- Dane lokalnie: `assets/Dane_CIFAR10/` (parquet, npz); na Edenie `~/smallvit_runs/2026-10-13_deit_cifar10/dane/cifar10_uint8.npz`.

## Uruchomienie
Lokalny test kodu (2 epoki na 1024 obrazach, kilka minut):
```
F:/Python/3.13/python.exe -X utf8 kod/trening_cifar_deit.py --wyniki <scratch>/proba --dane ../../assets/Dane_CIFAR10/cifar10_uint8.npz --proba --obserwuj --historia_wag
```
Eden (`~/smallvit_runs/2026-10-13_deit_cifar10/`, kod w `kod/`, dane w `dane/`, wyniki w `wyniki/<nazwa>/`, logi w `logi/`):
```
sbatch deit_cifar.slurm s47_obciecie --epoki 60 --obserwuj --historia_wag
sbatch deit_cifar.slurm s47_bez_obciecia --epoki 60 --obserwuj --historia_wag --obciecie 0 --wagi_co_po10k 500
sbatch --job-name=maly_cifar --partition=hopper-2,hopper deit_cifar.slurm maly_s47_obciecie --model malyvit --epoki 60 --obserwuj --historia_wag --stan_co 10
```
Pobranie wyników: strumień `ssh eden-cluster "tar cf - ..." | tar xf -` (nie listy ścieżek scp z Git Bash), docelowo do
`assets/DeiTTiny_CIFAR10_s47_<wariant>/` z README.
