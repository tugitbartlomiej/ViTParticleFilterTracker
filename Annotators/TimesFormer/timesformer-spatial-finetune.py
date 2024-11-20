import os
from datetime import datetime

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from transformers import TimesformerForVideoClassification, AutoImageProcessor


class MedicalVideoFrameDataset(Dataset):
    def __init__(self, images_dir, processor, num_frames=8):
        """
        Dataset dla klatek medycznych, który symuluje sekwencje wideo.

        Args:
            images_dir: ścieżka do folderu z obrazami
            processor: procesor obrazu TimeSformer
            num_frames: liczba powtórzeń obrazu do symulacji sekwencji
        """
        self.images_dir = images_dir
        self.processor = processor
        self.num_frames = num_frames
        self.samples = []
        self.class_mapping = {
            'surgical_tool': 0,
            'eye': 1
        }

        # Wczytaj obrazy
        for class_name in self.class_mapping:
            class_path = os.path.join(images_dir, class_name)
            if not os.path.exists(class_path):
                raise ValueError(f"Folder {class_path} nie istnieje!")

            for img_file in os.listdir(class_path):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append({
                        'image_path': os.path.join(class_path, img_file),
                        'label': self.class_mapping[class_name]
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Wczytaj obraz
        image = Image.open(sample['image_path']).convert('RGB')

        # Powiel obraz do sekwencji o długości num_frames
        frames = [image] * self.num_frames

        # Przetwórz sekwencję używając procesora TimeSformer
        inputs = self.processor(
            images=frames,
            return_tensors="pt",
            padding=True,
            do_resize=True,
            size={"height": 224, "width": 224},
        )

        pixel_values = inputs.pixel_values.squeeze(0)  # [num_frames, channels, height, width]

        return {
            'pixel_values': pixel_values,
            'label': torch.tensor(sample['label'])
        }


def train_model(model_name='facebook/timesformer-base-finetuned-k400',
                data_dir='./dataset_classes',
                output_dir='./surgical_model',
                num_epochs=30,
                batch_size=8,
                learning_rate=2e-5):
    """
    Fine-tuning TimeSformera do klasyfikacji obiektów medycznych.
    """
    # Utwórz katalogi
    os.makedirs(output_dir, exist_ok=True)

    # Ustaw urządzenie
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Inicjalizacja procesora i modelu
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = TimesformerForVideoClassification.from_pretrained(
        model_name,
        num_labels=2,  # surgical_tool, eye
        ignore_mismatched_sizes=True,
        label2id={'surgical_tool': 0, 'eye': 1},
        id2label={0: 'surgical_tool', 1: 'eye'}
    ).to(device)

    # Przygotuj dataset
    dataset = MedicalVideoFrameDataset(data_dir, processor)

    # Podziel na train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Optymalizator i scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )

    # TensorBoard
    writer = SummaryWriter(
        log_dir=os.path.join(output_dir, 'logs', datetime.now().strftime("%Y%m%d_%H%M%S"))
    )

    # Training loop
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            # Log progress
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')

        train_acc = 100. * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)

                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                val_loss += loss.item()
                _, predicted = outputs.logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        # Update learning rate
        scheduler.step()

        # Log metrics
        writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/val', val_loss / len(val_loader), epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_mapping': dataset.class_mapping
            }, os.path.join(output_dir, 'best_model.pth'))

            # Zapisz również model w formacie Transformers
            model.save_pretrained(os.path.join(output_dir, 'best_transformers_model'))
            processor.save_pretrained(os.path.join(output_dir, 'best_transformers_model'))

    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    train_model(
        data_dir='./dataset_classes',  # ścieżka do folderu z danymi
        output_dir='./surgical_model',  # ścieżka do zapisu modelu
        num_epochs=30,
        batch_size=8,  # zmniejszony batch size ze względu na większe zużycie pamięci
        learning_rate=2e-5
    )