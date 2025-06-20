@echo off
echo Uruchamianie skryptu wyboru znaczących obrazów...
python run_image_selector.py --keep_temp_files --annotation_weight 0.3 --labels_dir Annotators/Datasets/Yolo/yolo_dataset_20250218/labels/train
pause
