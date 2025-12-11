# Suggested Commands for Development

## Python Execution
```bash
# Use Python 3.11 explicitly (project requirement)
py -3.11 script.py

# Run with arguments
py -3.11 -m module_name

# Install packages
py -3.11 -m pip install package_name
```

## Main Entry Points

### YOLO Detection
```bash
cd Annotators/Yolo/
py -3.11 yolo_predict.py
```

### DETR Inference
```bash
cd Annotators/DetrAnnotator/
py -3.11 detr_annotator_inference_range_images.py
```

### Background Frame Selection
```bash
cd YOLO_DETR_Benchmarks/Intelligent_Background_Selector_2025-07-17/
py -3.11 intelligent_background_frame_selector.py
```

### DINO Frame Analysis
```bash
cd YOLO_DETR_Benchmarks/DINO_Frame_Selection/
py -3.11 dino_frame_selector.py
```

### Full Pipeline
```bash
cd YOLO_DETR_Benchmarks/scripts/DETR_Background_Training/
py -3.11 main_pipeline.py --config pipeline_config.yaml
```

### Simple Pipeline
```bash
cd YOLO_DETR_Benchmarks/scripts/DETR_Background_Training/
py -3.11 simple_pipeline.py
```

## Windows System Commands
```bash
# List directory
dir /b
ls -la  # Git Bash

# Find files
dir /s /b *.py  # Windows CMD
find . -name "*.py"  # Git Bash

# File operations
copy source dest
move source dest
del filename
mkdir dirname

# Process management
tasklist
taskkill /PID xxx /F
```

## Git Commands
```bash
# Status and diffs
git status
git diff
git log --oneline -10

# Branching
git checkout -b feature/new-feature
git checkout master
git merge feature/new-feature

# Commits
git add .
git commit -m "message"
git push origin branch-name
```

## GPU/CUDA
```bash
# Check GPU status
nvidia-smi

# Watch GPU usage
nvidia-smi -l 1
```

## Task Master (if configured)
```bash
task-master list
task-master next
task-master show <id>
task-master set-status --id=<id> --status=done
```
