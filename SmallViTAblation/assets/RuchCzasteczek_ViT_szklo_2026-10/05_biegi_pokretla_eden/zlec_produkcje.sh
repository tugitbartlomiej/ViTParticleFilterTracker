#!/bin/bash
# Fala 4 pokręteł (2026-10-11): ziarna w rzucie 128D, SGD (lr i paczka), drugi model (CNN), skan po warstwach, drugi zbiór (Imagenette),
# zapis f32 w oknie po 60. epoce. Trzy tory (--job-name=laneK --dependency=singleton: jeden bieg na tor naraz, ogranicza zużycie kwoty).
# Zmienne ANALIZY/ZOSTAW/WARSTWY idą przez środowisko (sbatch domyślnie eksportuje całe środowisko; opcja --export jest na Edenie zabijana).
#   bash zlec_produkcje.sh [A|B|C|all]
set -euo pipefail
B=/home2/faculty/bpiotrowski
EXP="$B/smallvit_runs/2026-10-10_szklo_pokretla"
E35="$B/smallvit_runs/2026-10-09_maly_vit_od_zera_35000_21epok"
M35="$E35/wejscia/MANIFEST_35000.jsonl"; S100="$E35/wejscia/SONDY_100.jsonl"
cd "$EXP"
CO=${1:-all}
z() { local lane=$1; shift; sbatch --job-name="$lane" --dependency=singleton "$@" | sed "s/^/[$lane] /"; }

if [ "$CO" = A ] || [ "$CO" = all ]; then
  # --- tor A: ziarna (3 lr × 2 nowe ziarna), f16, okno wczesne
  for lr in 5e-4 1.25e-4 2e-3; do
    for zi in 48 49; do
      ANALIZY=300:2188 z laneA pokretla_pelny.slurm "lr_${lr}_rp_z${zi}" "$lr" 3 "$M35" "$S100" 12 --rzut rp --ziarno "$zi"
    done
  done
fi
if [ "$CO" = B ] || [ "$CO" = all ]; then
  # --- tor B: SGD z pędem 0,9 (temperatura ∝ lr/paczka): lr przy paczce 32, potem paczka przy lr 0,1
  for lr in 0.03 0.1 0.3; do
    ANALIZY=300:2188 z laneB pokretla_pelny.slurm "sgd_lr_${lr}_rp" "$lr" 3 "$M35" "$S100" 12 --rzut rp --opt sgd
  done
  ANALIZY=300:2188 z laneB pokretla_pelny.slurm sgd_lr_0.1_b16_rp 0.1 2 "$M35" "$S100" 12 --rzut rp --opt sgd --paczka 16
  ANALIZY=300:2188 z laneB pokretla_pelny.slurm sgd_lr_0.1_b64_rp 0.1 5 "$M35" "$S100" 12 --rzut rp --opt sgd --paczka 64
  ANALIZY=300:2188 z laneB pokretla_pelny.slurm sgd_lr_0.1_b128_rp 0.1 9 "$M35" "$S100" 12 --rzut rp --opt sgd --paczka 128
fi
if [ "$CO" = C ] || [ "$CO" = all ]; then
  # --- tor C: drugi model (CNN) przy trzech lr, skan po warstwach, drugi zbiór (Imagenette: ViT i CNN, krótkie i długi bieg), f32 po 60. epoce
  for lr in 5e-4 1.25e-4 2e-3; do
    ANALIZY=300:2188 z laneC pokretla_pelny.slurm "cnn_lr_${lr}_rp" "$lr" 3 "$M35" "$S100" 12 --rzut rp --model cnn
  done
  ANALIZY=300:2188 WARSTWY=1 z laneC pokretla_pelny.slurm warstwy_lr_5e-4_rp 5e-4 3 "$M35" "$S100" 8 --rzut rp --warstwy --warstwy_n 4
  for lr in 5e-4 1.25e-4 2e-3; do
    ANALIZY=300:2188 z laneC klasyfikacja.slurm "klas_lr_${lr}" "$lr" 8 12
  done
  ANALIZY=300:2188 z laneC klasyfikacja.slurm klas_cnn_lr_5e-4 5e-4 8 12 --model cnn
  ANALIZY=300:2188,10746:11840 z laneC klasyfikacja.slurm klas_dlugi_lr_5e-4 5e-4 40 12 --gesto_od 10746
  # f32 kontra f16 w oknie po 60. epoce (wznowienie ze stanu e60 biegu 35 000)
  ST=$(ls "$EXP"/wyniki/okno60_rp/pelny/stan_e60.pt "$E35"/wyniki/*/pelny/stan_e60.pt 2>/dev/null | head -1)
  if [ -n "$ST" ]; then
    mkdir -p wyniki/okno60_rp_f32/pelny && cp -n "$ST" wyniki/okno60_rp_f32/pelny/
    ANALIZY=65640:66734 z laneC pokretla_pelny.slurm okno60_rp_f32 5e-4 61 "$M35" "$S100" 12 --rzut rp --zapis f32 --wznow --gesto_od 65640
  else
    echo "brak stan_e60.pt w $E35/wyniki/*/pelny - pomijam okno60_rp_f32"
  fi
fi
squeue -u "$USER" -o "%.9i %.8j %.2t %.8M %R" | head -40
