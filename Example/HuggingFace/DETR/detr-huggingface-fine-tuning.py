import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from datasets import load_dataset
from tqdm.auto import tqdm
import os
import json

class SurgicalToolDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, annotations_file, image_size=(800, 800)):
        """
        Inicjalizacja zestawu danych.

        Args:
            dataset: Zbiór danych wczytany za pomocą `datasets.load_dataset`.
            processor: Procesor obrazów z biblioteki Transformers.
            annotations_file: Ścieżka do pliku z adnotacjami w formacie COCO.
            image_size: Krotka określająca rozmiar obrazów po przetworzeniu (wysokość, szerokość).
        """
        self.dataset = dataset
        self.processor = processor
        self.image_size = image_size
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.filename_to_id = {ann['file_name']: ann['id'] for ann in self.annotations['images']}
        self.id_to_annotations = self._map_id_to_annotations()

    def _map_id_to_annotations(self):
        """
        Tworzy mapowanie od `image_id` do listy adnotacji.

        Returns:
            Słownik mapujący `image_id` na listę adnotacji.
        """
        id_to_ann = {}
        for ann in self.annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in id_to_ann:
                id_to_ann[image_id] = []
            # Oblicz obszar, jeśli nie jest podany
            if 'area' not in ann:
                ann['area'] = self.calculate_area(ann['bbox'])
            id_to_ann[image_id].append({
                'bbox': ann['bbox'],
                'category_id': ann['category_id'],
                'area': ann['area']
            })
        return id_to_ann

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Pobiera jeden element zestawu danych.

        Args:
            idx: Indeks elementu do pobrania.

        Returns:
            Słownik zawierający `pixel_values` i `labels`.
        """
        item = self.dataset[idx]
        image = item['image']
        image_filename = os.path.basename(item['image'].filename)

        # Znajdź odpowiednie adnotacje dla tego obrazu
        image_id = self.filename_to_id.get(image_filename)
        if image_id is None:
            print(f"No annotations found for image {image_filename}")
            annotations = {'image_id': idx, 'annotations': []}
        else:
            annotations = {
                'image_id': image_id,
                'annotations': self.id_to_annotations.get(image_id, [])
            }

        # Zastosuj przetwarzanie obrazu i adnotacji z określonym rozmiarem
        encoding = self.processor(
            images=image,
            annotations=[annotations],
            return_tensors="pt",
            size=self.image_size  # Ustawienie stałego rozmiaru
        )
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return {"pixel_values": pixel_values, "labels": target}

    @staticmethod
    def calculate_area(bbox):
        """
        Oblicza obszar zadanego bbox.

        Args:
            bbox: Lista lub krotka zawierająca [x, y, szerokość, wysokość].

        Returns:
            Obliczony obszar.
        """
        return bbox[2] * bbox[3]  # width * height

def collate_fn(batch):
    """
    Funkcja collate do łączenia batcha.

    Args:
        batch: Lista słowników zawierających `pixel_values` i `labels`.

    Returns:
        Słownik zawierający złożone `pixel_values` i listę `labels`.
    """
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    pixel_values = torch.stack(pixel_values)  # Teraz wszystkie mają ten sam rozmiar
    return {"pixel_values": pixel_values, "labels": labels}

def train_epoch(model, data_loader, optimizer, device):
    """
    Trenuje model przez jeden epoch.

    Args:
        model: Model do trenowania.
        data_loader: DataLoader zawierający dane treningowe.
        optimizer: Optymalizator.
        device: Urządzenie (CPU lub GPU).

    Returns:
        Średnia strata (loss) za epoch.
    """
    model.train()
    total_loss = 0

    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(data_loader)

def main():
    # Konfiguracja
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 10
    learning_rate = 1e-5
    batch_size = 4
    image_size = (800, 800)  # Stały rozmiar obrazów

    # Ładowanie modelu i procesora
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Ładowanie i przygotowanie danych
    dataset = load_dataset("imagefolder", data_dir="E:/wspolne_labelowanie/train/")
    annotations_file = "E:/wspolne_labelowanie/train/_annotations.coco.json"  # Ścieżka do pliku z adnotacjami

    # Wydrukuj pierwsze kilka elementów zbioru danych, aby zobaczyć ich strukturę
    print("Sample data items:")
    for i in range(min(5, len(dataset['train']))):
        print(f"Item {i}:")
        print(dataset['train'][i])
        print("/n")

    train_dataset = SurgicalToolDataset(dataset['train'], processor, annotations_file, image_size=image_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Przygotowanie modelu do treningu
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Pętla treningowa
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, train_dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    # Zapisywanie modelu
    model.save_pretrained("./surgical_tool_detector")
    processor.save_pretrained("./surgical_tool_detector")
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
