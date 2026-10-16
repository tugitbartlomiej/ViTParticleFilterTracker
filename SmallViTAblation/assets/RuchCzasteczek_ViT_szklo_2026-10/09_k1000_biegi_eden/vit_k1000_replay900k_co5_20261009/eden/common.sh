#!/usr/bin/env bash
set -euo pipefail

ROOT=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_replay900k_co5_20261009
CODE="$ROOT/code"
PROVENANCE="$ROOT/provenance"
RESULTS="$ROOT/results"
LOGS="$ROOT/logs"
SUBMISSION="$ROOT/submission"
PY=/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/bin/python
TEST_PY=/home2/faculty/bpiotrowski/venvs/nndmd_expanded_20261002/bin/python
CHECKPOINT=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/wagi/k1000.pt
CHECKPOINT_ADAMW=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/pelny/k1000.pt
COCO=/home2/faculty/bpiotrowski/smallvit_runs/2026-09-28_maly_vit_od_zera/wejscia/pretrain/train/_annotations.coco.json
ARCHIVE=/home2/faculty/bpiotrowski/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar
OLD_ROOT=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_1500_pelne_latenty_90_20261008
OLD_PRODUCTION="$OLD_ROOT/results/production"
MANIFEST="$PROVENANCE/MANIFEST_1500.jsonl"
OLD_INDEX="$PROVENANCE/OLD_ORACLE_INDEX.jsonl"
PCA="$PROVENANCE/PCA3_FIXED.npz"
PCA32="$PROVENANCE/PCA32_ANALYTICAL.npz"
PCA32_MANIFEST="$PROVENANCE/PCA32_ANALYTICAL_MANIFEST.json"
PANEL="$PROVENANCE/PANEL_64.json"
SMOKE_INDICES="$PROVENANCE/SMOKE_INDICES.json"

export ROOT CODE PROVENANCE RESULTS LOGS SUBMISSION PY TEST_PY CHECKPOINT CHECKPOINT_ADAMW COCO ARCHIVE
export OLD_ROOT OLD_PRODUCTION MANIFEST OLD_INDEX PCA PCA32 PCA32_MANIFEST PANEL SMOKE_INDICES
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$RESULTS" "$LOGS" "$SUBMISSION"
