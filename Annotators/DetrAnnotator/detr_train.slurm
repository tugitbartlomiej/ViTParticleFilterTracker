#!/bin/bash
#SBATCH -A transformers_vsc       # Twoja grupa
#SBATCH -p short                  # Kolejka (np. short)
#SBATCH --gres=gpu:10             # 6 GPU
#SBATCH --mem=64G                # Ilość pamięci (64 GB przykładowo)
#SBATCH --time=02:00:00          # Limit czasu, 2h
#SBATCH --job-name=yolo_train
#SBATCH --output=yolo_train_%j.out

# Załaduj moduł Anacondy i aktywuj środowisko
module load anaconda/4.0
source $CONDA_SOURCE
conda activate yolo_env_311_gpu

cd /mnt/evafs/faculty/home/bpiotrowski/DETR

/home2/faculty/bpiotrowski/.conda/envs/yolo_env_311_gpu/bin/python detr_annotator_train.py
