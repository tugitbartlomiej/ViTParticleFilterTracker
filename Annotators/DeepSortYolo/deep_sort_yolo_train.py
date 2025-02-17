import os

import torch
from ultralytics import YOLO


def main():
    # Wyłączanie wandb:
    os.environ["WANDB_DISABLED"] = "true"

    data_yaml = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Annotators\OpencvTrackerAnnotator\output\deepsort_yolo_dataset\dataset.yaml"
    base_model = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Annotators\Yolo\surgical_tool_detection\exp14\weights\best.pt"
    epochs = 50
    batch = 16
    imgsz = 640
    project = "tooltip_detection"
    name = "exp"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[TRAIN] Using device: {device}")

    model = YOLO(base_model)
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        pretrained=True,
        verbose=True
        # w nowszych wersjach można dodać: use_wandb=False
    )
    print("[TRAIN] YOLO training completed.")


if __name__ == "__main__":
    main()
