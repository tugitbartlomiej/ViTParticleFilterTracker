import os
from datetime import datetime

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import TimesformerForVideoClassification, AutoImageProcessor


class SurgicalVideoDataset(Dataset):
    def __init__(self, sequences_dir, processor, num_frames=8):
        """
        Dataset dla sekwencji chirurgicznych.
        
        Args:
            sequences_dir: Ścieżka do folderu z sekwencjami
            processor: Procesor obrazu TimeSformer
            num_frames: Liczba klatek w sekwencji
        """
        self.sequences_dir = sequences_dir
        self.processor = processor
        self.num_frames = num_frames
        self.samples = []
        self.classes = {}
        self._load_sequences()

    def _load_sequences(self):
        """Ładuje sekwencje i ich etykiety."""
        for seq_folder in os.listdir(self.sequences_dir):
            seq_path = os.path.join(self.sequences_dir, seq_folder)
            if not os.path.isdir(seq_path):
                continue

            # Odczytaj klasę z pliku description.txt
            desc_file = os.path.join(seq_path, 'description.txt')
            if not os.path.exists(desc_file):
                continue

            with open(desc_file, 'r', encoding='utf-8') as f:
                label = f.read().strip()
                if label not in self.classes:
                    self.classes[label] = len(self.classes)

            # Zbierz ścieżki do klatek
            frame_files = sorted([
                os.path.join(seq_path, f)
                for f in os.listdir(seq_path)
                if f.endswith(('.jpg', '.png'))
            ])

            if len(frame_files) >= self.num_frames:
                self.samples.append({
                    'frames': frame_files,
                    'label': label
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = []

        # Wybierz równomiernie rozłożone klatki
        step = len(sample['frames']) // self.num_frames
        selected_frames = sample['frames'][::step][:self.num_frames]

        for frame_path in selected_frames:
            image = Image.open(frame_path).convert('RGB')
            frames.append(image)

        # Przygotuj dane wejściowe dla modelu
        inputs = self.processor(
            images=frames,
            return_tensors="pt",
            padding=True,
            do_resize=True,
            size={"height": 224, "width": 224},
        )

        pixel_values = inputs.pixel_values.squeeze(0)  # [num_frames, channels, height, width]
        label = torch.tensor(self.classes[sample['label']], dtype=torch.long)

        return {
            'pixel_values': pixel_values,
            'label': label
        }

def train_model(model, train_loader, val_loader, num_epochs, device, writer=None):
    """
    Trenuje model TimeSformer.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Tryb treningu
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

            if writer:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/train', loss.item(), global_step)

        train_acc = 100. * train_correct / train_total
        
        # Walidacja
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)

                outputs = model(pixel_values=pixel_values).logits
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100. * val_correct / val_total

        if writer:
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')

        # Zapisz najlepszy model
        # if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, 'best_surgical_timesformer.pth')

def main():
    # Konfiguracja
    sequences_dir = './zapisane_sekwencje'
    num_frames = 8
    batch_size = 2
    num_epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Inicjalizacja procesora i modelu
    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    dataset = SurgicalVideoDataset(sequences_dir, processor, num_frames)

    if len(dataset) == 0:
        print("Brak danych treningowych!")
        return

    # Inicjalizacja modelu
    model = TimesformerForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k400",
        num_labels=len(dataset.classes),
        ignore_mismatched_sizes=True,
        label2id=dataset.classes,
        id2label={v: k for k, v in dataset.classes.items()}
    ).to(device)

    # Podział na zbiór treningowy i walidacyjny
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Inicjalizacja TensorBoard
    writer = SummaryWriter(f'runs/surgical_timesformer_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    # Trening modelu
    train_model(model, train_loader, val_loader, num_epochs, device, writer)
    writer.close()

if __name__ == '__main__':
    main()
