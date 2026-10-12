#!/usr/bin/env bash
set -euo pipefail

ROOT=/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_1500_pelne_latenty_90_20261008
CODE="$ROOT/code"
PROVENANCE="$ROOT/provenance"
RESULTS="$ROOT/results"
LOGS="$ROOT/logs"
PY=/home2/faculty/bpiotrowski/venvs/rfdetr_nano_clean_1_10_1/bin/python
TEST_PY=/home2/faculty/bpiotrowski/venvs/nndmd_expanded_20261002/bin/python
CHECKPOINT=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/wagi/k1000.pt
CHECKPOINT_ADAMW=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/pelny/k1000.pt
COCO=/home2/faculty/bpiotrowski/smallvit_runs/2026-09-28_maly_vit_od_zera/wejscia/pretrain/train/_annotations.coco.json
ARCHIVE=/home2/faculty/bpiotrowski/smallvit_runs/2026-09-15_tracking_static36/data/tracking_static36.tar
MANIFEST="$PROVENANCE/MANIFEST_1500.jsonl"
SMOKE_INDICES="$PROVENANCE/SMOKE_INDICES.json"
CONTROL_PANEL="$PROVENANCE/CONTROL_PANEL.json"

export ROOT CODE PROVENANCE RESULTS LOGS PY TEST_PY CHECKPOINT CHECKPOINT_ADAMW COCO ARCHIVE MANIFEST SMOKE_INDICES CONTROL_PANEL
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "$LOGS" "$RESULTS"
