import os
import torch
from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Sprawdzenie dostępności CUDA
print("CUDA is available:", torch.cuda.is_available())

# Wczytanie modelu YOLO
model = YOLO("yolov8n.pt")  # Wersja 'n' (nano) modelu
print("YOLOv8 is ready.")

# Wyświetlenie wersji bibliotek
import torchvision
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"torchvision version: {torchvision.__version__}")


class YOLOTrainer:
    def __init__(
        self,
        dataset_yaml_path: str,
        model_size: str = 'n',  # np. 'n', 's', 'm', 'l', 'x'
        epochs: int = 50,
        batch_size: int = 16,
        imgsz: int = 640,
        project_name: str = 'surgical_tool_detection'
    ):
        """
        Inicjalizacja trenera YOLO.
        """
        # Wyłączenie logowania wandb
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"

        # Przypisanie parametrów treningowych
        self.dataset_yaml_path = dataset_yaml_path
        self.model_size = model_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.project_dir = project_name

        # Inicjalizacja modelu YOLO
        self.model = YOLO(f'yolov8{self.model_size}.pt')

    def train(self):
        print("Starting YOLO training...")  # Punkt kontrolny
        results = self.model.train(
            data=self.dataset_yaml_path,
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.imgsz,
            device=self.device,
            project=self.project_dir,
            name="exp",
            pretrained=True,
            verbose=True
        )
        print("Training completed successfully.")  # Punkt kontrolny

        # Wyświetlenie podsumowania wyników dla każdej epoki
        for epoch, result in enumerate(results.epoch_results):
            print(f"Epoch {epoch+1}/{self.epochs} - "
                  f"box_loss: {result['box_loss']:.4f}, "
                  f"cls_loss: {result['cls_loss']:.4f}, "
                  f"dfl_loss: {result['dfl_loss']:.4f}")


def main():
    # Inicjalizacja klasy trenera z odpowiednimi parametrami
    trainer = YOLOTrainer(
        dataset_yaml_path="../OpencvTrackerAnnotator/output/yolo_dataset/dataset.yaml",  # Zaktualizuj ścieżkę do swojego zbioru danych
        model_size='s',  # Zmiana rozmiaru modelu, np. 'n', 's', 'm', 'l', 'x'
        epochs=50,       # Liczba epok
        batch_size=16,   # Rozmiar batcha
        imgsz=640        # Rozmiar obrazu
    )

    # Rozpoczęcie procesu treningu
    trainer.train()


if __name__ == "__main__":
    main()
