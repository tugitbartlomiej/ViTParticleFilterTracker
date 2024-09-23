# main_script.py

import torch
from torch.utils.data import DataLoader

from ToolTipTracking.utils.detr_tracker import DETRTracker
from utils.tool_tip_data_processor import ToolTipDataProcessor
from tool_tip_trainer import train_model
from utils.tool_tip_dataset import ToolTipDataset
from torchvision import transforms

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # Ścieżki do danych
    video_path = 'E:/DETR_DATASET/ToolTip/ToolTipDataset/annotations/RetinaThickNeedle.mp4'
    json_path = 'E:/DETR_DATASET/ToolTip/ToolTipDataset/annotations/instances_default.json'
    output_dir = 'E:/DETR_DATASET/ToolTip/ToolTipDataset/annotations/output'
    debug_dir = 'E:/DETR_DATASET/ToolTip/ToolTipDataset/debug'

    # Tworzenie i używanie ToolTipDataProcessor
    processor = ToolTipDataProcessor(video_path, json_path, output_dir, debug_dir)
    processor.extract_frames()
    processor.load_annotations()

    # Definicja transformacji
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # Średnie wartości dla ImageNet
                             [0.229, 0.224, 0.225])  # Odchylenia standardowe dla ImageNet
    ])

    # Tworzenie datasetu z transformacją
    try:
        dataset = ToolTipDataset(processor.output_dir, processor.json_path, transform=transform)
        print(f"Liczba próbek w zbiorze danych: {len(dataset)}")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    if len(dataset) == 0:
        print("Zbiór danych jest pusty. Sprawdź, czy pliki zostały poprawnie wyodrębnione i adnotacje są prawidłowe.")
        return

    # Podział na zbiór treningowy i walidacyjny
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"Zbiór treningowy: {len(train_dataset)} próbek")
    print(f"Zbiór walidacyjny: {len(val_dataset)} próbek")

    # Tworzenie dataloaderów
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Inicjalizacja modelu DETR
    num_classes = 1  # Przykład: 1 klasa obiektu + tło
    model = DETRTracker(num_classes=num_classes)

    # Konfiguracja treningu
    num_epochs = 20
    learning_rate = 1e-4

    # Trening modelu
    trained_model = train_model(model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate)

    # Model z najlepszymi wagami został już zapisany w funkcji train_model
    print("Trening zakończony!")

    # Opcjonalnie: zapisz ostatni stan modelu
    torch.save(trained_model.state_dict(), 'E:/DETR_DATASET/ToolTip/ToolTipDataset/models/final_detr_tracker.pth')

if __name__ == "__main__":
    main()
