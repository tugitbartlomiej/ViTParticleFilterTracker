#!/bin/bash
# Wysyłka kodu i przygotowanie środowiska na Edenie (PLAN.md, krok 2). NIE zgłasza żadnego zadania.
# Na węźle logowania: diffusers 0.39.0 do $MY/deps (bez zależności; reszta jest w venv), wagi SD 1.5 (UNet i koder
# tekstu w fp16, tokenizer, harmonogram) i VAE sd-vae-ft-mse do $MY/modele (~2,3 GB). Kontrola kwoty przed i po.
#   bash eden/wyslij.sh            (z katalogu dyfuzja_sd_filmy, Git Bash)
set -euo pipefail
H=eden-cluster
TU=$(cd "$(dirname "$0")/.." && pwd)
B=/home2/faculty/bpiotrowski
MY=$B/smallvit_runs/2026-09-30_dyfuzja_sd_filmy
PY=$B/venvs/rfdetr_nano_clean_1_10_1/bin/python
MAN="$TU/../../feature_cache/eden_static36/data/tracking_static36/pool_manifest.jsonl"
[ -f "$MAN" ] || { echo "BRAK $MAN"; exit 2; }
cicho() { grep -v "^\*\|SSH server\|Faculty\|Individuals\|excess\|all of\|recorded\|monitoring\|authorized\|this system\|advised\|activity\|such mon\|^$" || true; }
echo "== kwota przed"; ssh $H "lfs quota -p 1299 /mnt/evafs | tail -1" 2>/dev/null | cicho
ssh $H "mkdir -p $MY/kod $MY/eden $MY/logi $MY/wyniki $MY/wejscia $MY/modele $MY/deps" 2>/dev/null
scp -q "$TU"/kod/*.py "$H:$MY/kod/"
scp -q "$TU"/eden/*.slurm "$H:$MY/eden/"
scp -q "$MAN" "$H:$MY/wejscia/pool_manifest.jsonl"
ssh $H "sed -i 's/\r\$//' $MY/eden/*.slurm $MY/kod/*.py" 2>/dev/null
ssh $H "set -e
  [ -f $MY/deps/diffusers/__init__.py ] || $PY -m pip install -q --no-deps --target $MY/deps diffusers==0.39.0
  PYTHONPATH=$MY/deps $PY -c 'import diffusers, torch; from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler; print(\"diffusers\", diffusers.__version__, \"torch\", torch.__version__)'
  cd $MY/modele && HF_HOME=$MY/modele/.hf PYTHONPATH=$MY/deps $PY - <<'EOP'
from huggingface_hub import snapshot_download
snapshot_download('stable-diffusion-v1-5/stable-diffusion-v1-5', local_dir='sd15',
                  allow_patterns=['model_index.json', 'scheduler/*', 'tokenizer/*', 'unet/config.json',
                                  'unet/diffusion_pytorch_model.fp16.safetensors', 'text_encoder/config.json',
                                  'text_encoder/model.fp16.safetensors'])
snapshot_download('stabilityai/sd-vae-ft-mse', local_dir='sd-vae-ft-mse',
                  allow_patterns=['config.json', 'diffusion_pytorch_model.safetensors'])
print('wagi pobrane')
EOP
  rm -rf $MY/modele/.hf $MY/modele/sd15/.cache $MY/modele/sd-vae-ft-mse/.cache
  du -sh $MY/modele/* $MY/deps" 2>/dev/null | cicho
echo "== kwota po"; ssh $H "lfs quota -p 1299 /mnt/evafs | tail -1" 2>/dev/null | cicho
echo "Zgłoszenia (po zgodzie):"
echo "  próba:   ssh $H 'cd $MY/eden && sbatch --gres=gpu:2 przygotuj_sd.slurm train 1 2 && sbatch --gres=gpu:2 przygotuj_sd.slurm val 10 2'"
echo "           ssh $H 'cd $MY/eden && sbatch --gres=gpu:2 trenuj_sd.slurm 200 16 proba'"
echo "  całość:  ssh $H 'cd $MY/eden && sbatch --gres=gpu:4 przygotuj_sd.slurm train 1 && sbatch --gres=gpu:4 przygotuj_sd.slurm val 10'"
echo "           ssh $H 'cd $MY/eden && sbatch --gres=gpu:8 trenuj_sd.slurm 20000 32'"
