from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForVideoClassification, AutoImageProcessor

from Annotators.TimesFormer.video_dataset import VideoDataset


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        # Przenieś dane na odpowiednie urządzenie
        inputs = inputs.to(device)  # [batch_size, num_frames, channels, height, width]
        labels = labels.to(device)

        # Przygotuj dane w odpowiednim formacie dla TimeSformera
        inputs = inputs.transpose(1, 2)  # [batch_size, channels, num_frames, height, width]

        optimizer.zero_grad()
        outputs = model(pixel_values=inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def main():
    # Konfiguracja
    seq_dir = './zapisane_sekwencje/'
    num_frames = 8  # Zmieniono na 8 klatek
    batch_size = 2
    num_epochs = 5
    learning_rate = 1e-4

    # Inicjalizacja procesora i modelu
    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    dataset = VideoDataset(seq_dir, processor=processor, num_frames=num_frames)

    if len(dataset) == 0:
        print("Dataset jest pusty!")
        return

    # Przygotowanie urządzenia
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(dataset.class_mapping)

    # Inicjalizacja modelu
    model = AutoModelForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k400",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        label2id=dataset.class_mapping,
        id2label={v: k for k, v in dataset.class_mapping.items()}
    ).to(device)

    # Konfiguracja treningu
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Przygotowanie dataloaderów
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Inicjalizacja TensorBoard
    writer = SummaryWriter(f'runs/timesformer_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    # Główna pętla treningu
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Trening
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        # Logowanie do TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        # Zapisywanie najlepszego modelu
        if train_acc > best_val_acc:
            best_val_acc = train_acc
            torch.save(model.state_dict(), 'best_timesformer_model.pth')
            print(f"Saved best model with accuracy {best_val_acc:.4f}")

    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    main()