#!/bin/bash
#SBATCH -A transformers_vsc               # upewnij się, że to konto ma prawo korzystać z GPU
#SBATCH --qos=long-part                   # explicitly select QOS, który pozwala na GRES
#SBATCH -p long                           # partycja z węzłami DGX
## opcjonalnie: usuń -w, żeby scheduler wybrał wolny węzeł
## #SBATCH -w dgx-4

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3                      # 3 GPU na jednym węźle
#SBATCH --mem=45G
#SBATCH --time=24:00:00
#SBATCH --job-name=detr_annotator
#SBATCH --output=out/%x_%j.out

module load cuda/11.4                     # lub inna dostępna wersja
# aktywuj środowisko, jeśli potrzebne:
# source activate detr_env_py310

# przejdź do katalogu z kodem
cd /mnt/evafs/faculty/home/bpiotrowski/DETR/DETR_fo_a100

# uruchomienie treningu
python detr_annotator_train.py \
  --train_images_dir "/mnt/evafs/.../images" \
  --train_annotations_file "/mnt/evafs/.../augmented_coco_20250417_030014.json" \
  --checkpoint_dir "./checkpoints" \
  --best_model_dir "./detr_tool_tracking_model_best" \
  --num_epochs 10 \
  --learning_rate 5e-5 \
  --batch_size 8 \
  --image_size 800 800 \
  --val_split 0.1 \
  --resume_training
