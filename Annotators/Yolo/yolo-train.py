import os
import signal
import sys
import warnings

import torch
from ultralytics import YOLO

# Wyłączenie ostrzeżeń FutureWarning dotyczących biblioteki torch
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Sprawdzenie dostępności CUDA
print("CUDA is available:", torch.cuda.is_available())

# Wczytanie modelu YOLO (upewnij się, że plik modelu jest dostępny w ścieżce)
model = YOLO("yolov8l.pt")
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
        model_size: str = 'n',  # Dostępne opcje: 'n', 's', 'm', 'l', 'x'
        epochs: int = 50,
        batch_size: int = 16,
        imgsz: int = 640,
        project_name: str = 'surgical_tool_detection'
    ):
        """
        Inicjalizacja trenera YOLO.
        """
        # Wyłączenie logowania przez wandb
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"

        self.dataset_yaml_path = dataset_yaml_path
        self.model_size = model_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.project_dir = project_name

        # Inicjalizacja modelu YOLO dla określonego rozmiaru (np. yolov8l.pt)
        self.model = YOLO(f'yolov8{self.model_size}.pt')

        # Ustawienie obsługi sygnałów przerwania (CTRL+C, SIGTERM)
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, sig, frame):
        print("\nInterrupt signal received. Saving checkpoint...")
        # Zapis bieżącego stanu modelu do pliku checkpoint
        self.model.save("checkpoint.pt")
        print("Checkpoint saved. Exiting.")
        sys.exit(0)

    def train(self):
        print("Starting YOLO training...")
        try:
            results = self.model.train(
                data=self.dataset_yaml_path,
                epochs=self.epochs,
                batch=self.batch_size,
                imgsz=self.imgsz,
                device=self.device,
                project=self.project_dir,
                name="exp",
                pretrained=True,
                verbose=True,
                workers=0,    # Multiprocessing wyłączony przy ładowaniu danych
                resume=False  # Ustawione na False, aby uniknąć ładowania niepoprawnego checkpointu
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving checkpoint...")
            self.model.save("checkpoint.pt")
            print("Checkpoint saved. Exiting training loop.")
            sys.exit(0)

        print("Training completed successfully.")

        # Wyświetlenie podsumowania wyników treningu
        if hasattr(results, 'mean_results'):
            mean_results = results.mean_results()
            # Jeśli zwrócony wynik jest listą
            if isinstance(mean_results, list):
                print("Mean training results per epoch:")
                for epoch, result in enumerate(mean_results, start=1):
                    # Sprawdzenie, czy element listy jest słownikiem
                    if isinstance(result, dict):
                        print(f"Epoch {epoch}: box_loss: {result.get('box_loss', float('nan')):.4f}, "
                              f"cls_loss: {result.get('cls_loss', float('nan')):.4f}, "
                              f"dfl_loss: {result.get('dfl_loss', float('nan')):.4f}")
                    else:
                        # Jeśli nie, wypisujemy wartość bez indeksowania
                        print(f"Epoch {epoch}: {result}")
            elif isinstance(mean_results, dict):
                print("Mean training results:")
                print(f"box_loss: {mean_results.get('box_loss', float('nan')):.4f}, "
                      f"cls_loss: {mean_results.get('cls_loss', float('nan')):.4f}, "
                      f"dfl_loss: {mean_results.get('dfl_loss', float('nan')):.4f}")
            else:
                print("Mean training results:", mean_results)
        elif hasattr(results, 'results_dict'):
            results_dict = results.results_dict()
            print("Training results (summary):")
            for key, value in results_dict.items():
                print(f"{key}: {value}")
        else:
            print("Unable to retrieve detailed epoch results.")


def main():
    trainer = YOLOTrainer(
        dataset_yaml_path=r"/mnt/evafs/faculty/home/bpiotrowski/datasets/yolo_dataset_20250218/data.yaml",
        model_size='l',  # Możliwe opcje: 'n', 's', 'm', 'l', 'x'
        epochs=40,       # Liczba epok
        batch_size=16,  # Rozmiar batcha
        imgsz=640       # Rozmiar obrazu
    )
    trainer.train()


if __name__ == "__main__":
    main()
