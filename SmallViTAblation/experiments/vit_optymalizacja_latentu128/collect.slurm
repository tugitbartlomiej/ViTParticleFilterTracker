#!/bin/bash
#SBATCH --job-name=vit128report
#SBATCH --account=transformers_vsc
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
set -euo pipefail
cd /home2/faculty/bpiotrowski/smallvit_runs/2026-10-04_vit_optymalizacja_latentu128
/usr/bin/python3 collect.py results
