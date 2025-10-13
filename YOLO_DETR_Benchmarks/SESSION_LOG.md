Benchmark session log — 2025-10-13

Summary
- Goal: Run YOLO vs DETR benchmark with visible console progress and fix loading issues on PyTorch 2.6.
- Critical Discovery: DETR converted model lost Query81 specialization (96.3% → 0% hit rate)
- Outcome: Config updated, robust path resolution added, PyTorch 2.6 safe-pickle allowlist integrated, DETR now loads from original checkpoint to preserve Query81.

Key Changes
- Advanced_Analysis/config.yaml
  - models.yolo.path: models/YOLO/epoch100.pt
  - models.detr.path: models/DETR/checkpoint_epoch_100.pth (CHANGED to original checkpoint)
  - dataset.images_dir: ../BackgroundFinetuned/Datasets/TooltipMining/train
  - dataset.annotations_path: ../BackgroundFinetuned/Datasets/TooltipMining/annotations/tool_train_annotations.json
  - label_map: { yolo: {0: 1}, detr: {0: 1} } to match COCO category id=1 (surgical_tool)

- Advanced_Analysis/run_inference.py
  - Added PyTorch 2.6 allowlist for Ultralytics and torch.nn classes to avoid UnpicklingError when loading .pt weights.
  - Added resolve_path() to robustly locate model folders/files (prints chosen path). Prefers YOLO epoch100.pt.
  - Added informative prints: shows resolved paths and which YOLO weights are being loaded.
  - **CRITICAL**: Added load_detr_from_checkpoint() function to load original checkpoint instead of converted model
  - **QUERY81 PRESERVATION**: Original checkpoint preserves Query81 specialization (96.3% hit rate vs 0% in converted model)

Paths used
- YOLO weights (preferred): models/YOLO/epoch100.pt
- DETR checkpoint (with Query81): models/DETR/checkpoint_epoch_100.pth
- Images: ../BackgroundFinetuned/Datasets/TooltipMining/train
- Annotations: ../BackgroundFinetuned/Datasets/TooltipMining/annotations/tool_train_annotations.json

Query81 Specialization (CRITICAL FOR PERFORMANCE)
- Original checkpoint: Query 81 achieved 96.3% hit rate (182/189 detections, 97.3% avg confidence)
- Converted model: Query 81 had 0% hit rate (0/30 detections, 0.5% avg confidence)
- Root cause: Model conversion incorrectly reinitialized model, destroying learned query embeddings
- Solution: Load directly from checkpoint_epoch_100.pth using strict=True state_dict loading

How to run (Windows PowerShell, Python 3.11)
- Ensure dependencies:
  - py -3.11 -m pip install -r Advanced_Analysis/requirements.txt
  - If pycocotools fails on Windows: py -3.11 -m pip install pycocotools-windows
- Run benchmark:
  - py -3.11 Advanced_Analysis/run_benchmark.py --config Advanced_Analysis/config.yaml
- Optional visualizations (after predictions exist):
  - py -3.11 Advanced_Analysis/visualize_results.py

What you should see on console
- "Running YOLOv8 inference..." and tqdm progress bar (YOLO Inference)
- "Running DETR inference..." and tqdm progress bar (DETR Inference)
- COCOeval summary table (mAP / AR), temporal stability metrics, and final report path

Known issues addressed
- PyTorch 2.6 safe-pickle breaks Ultralytics .pt loading with UnpicklingError.
  - Fix: allowlisted common classes (ultralytics.nn.tasks + ultralytics.nn.modules + torch.nn containers).
- Relative path mismatch when running from repo root.
  - Fix: resolve_path() tries as-is, then repo-relative, then known candidates; logs the chosen path.

Troubleshooting
- If YOLO still fails to load due to an unsupported class, re-run and share the exact class name; the allowlist can be extended similarly.
- To benchmark DETR from a .pth checkpoint instead of HF export, add a fallback loader in run_inference.py (can be implemented on request).

Commands recap
- Benchmark: py -3.11 Advanced_Analysis/run_benchmark.py --config Advanced_Analysis/config.yaml
- Visualize: py -3.11 Advanced_Analysis/visualize_results.py

