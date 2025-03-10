#!/bin/bash
#SBATCH -A transformers_vsc
#SBATCH -p short
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --job-name=coco_archiver
#SBATCH --output=coco_archiver_%j.out

module load anaconda/4.0
source $CONDA_SOURCE

cd /mnt/evafs/faculty/home/bpiotrowski/DETR

# tutaj używasz python z environmentu yolo_env_311_gpu
/home2/faculty/bpiotrowski/.conda/envs/yolo_env_311_gpu/bin/python coco_augmentation_archivier.py
