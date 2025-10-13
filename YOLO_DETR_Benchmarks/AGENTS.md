# Repository Guidelines

## Project Structure & Module Organization
- `Advanced_Analysis/`: Main benchmarking and inference code (`run_inference.py`, `run_benchmark.py`, `evaluate_metrics.py`, `visualize_results.py`, `config.yaml`).
- `scripts/`: Training and orchestration (DETR DDP trainer `detr_train_optimized.py`, YOLO trainer `yolo-train.py`, SLURM job files, background finetune pipeline under `DETR_Background_Training/`).
- `models/`: Pretrained and converted weights (`models/DETR/*.pth`, `models/YOLO/*.pt`).
- `simple_test/`: Ad‑hoc sanity checks, sample frames, and result images.
- `DETR/`: Exported inference model artifacts.
- `Datasets/`: Local dataset structure (YOLO/COCO style data; not versioned upstream).

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r Advanced_Analysis/requirements.txt`
- Inference: `python Advanced_Analysis/run_inference.py --config Advanced_Analysis/config.yaml`
- Benchmark: `python Advanced_Analysis/run_benchmark.py --config Advanced_Analysis/config.yaml`
- Metrics: `python Advanced_Analysis/evaluate_metrics.py --config Advanced_Analysis/config.yaml`
- DETR training (DDP): `torchrun --nproc_per_node=4 scripts/detr_train_optimized.py --images_dir <images> --annotations_path <coco.json> --output_dir out/ --checkpoint_dir ckpts/`
- YOLO training: `python scripts/yolo-train.py --dataset_yaml_path Datasets/Yolo/dataset.yaml --epochs 50`

## Coding Style & Naming Conventions
- Python 3.10+, PEP 8, 4‑space indentation; line length ≤ 100.
- Files/functions: `snake_case`; classes: `CamelCase`; modules live under `Advanced_Analysis/` or `scripts/`.
- Prefer type hints and docstrings; keep CLI via `argparse`. Config via YAML (`config.yaml`, `pipeline_config*.yaml`).

## Testing Guidelines
- No formal test suite; use quick smoke tests:
  - Run inference on `simple_test/test_frames/` and verify outputs in `simple_test/*results*/`.
  - Validate metrics with `evaluate_metrics.py` and attach a short summary.
- Add new sanity scripts under `simple_test/` and keep input/output paths configurable.

## Commit & Pull Request Guidelines
- Commits: imperative mood with scope, e.g., `DETR: add DDP trainer`, `YOLO: fix dataset YAML`. Group related changes.
- PRs must include: purpose, key commands run, configs used, data path patterns, hardware (GPU count), and before/after artifacts (images or metrics).
- Large files: do not commit bulky checkpoints; prefer external storage or Git LFS. Avoid secrets in configs.

## Security & Configuration Tips
- Keep datasets and keys out of VCS; use local paths and environment variables.
- SLURM jobs live in `scripts/*.slurm`; mirror CLI flags from the Python entrypoints and document cluster specifics in the PR.

