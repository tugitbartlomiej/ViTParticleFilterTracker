# tool_tip_trainer.py

import torch
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, targets in progress_bar:
            # Przeniesienie danych na urządzenie
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            outputs = model(images, targets)

            # Sumowanie wszystkich strat
            loss_dict = outputs
            losses = sum(loss for loss in loss_dict.values())

            losses.backward()
            optimizer.step()

            train_loss += losses.item()
            progress_bar.set_postfix({'train_loss': f"{losses.item():.4f}"})

        scheduler.step()

        # Walidacja
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs = model(images, targets)
                loss_dict = outputs
                losses = sum(loss for loss in loss_dict.values())

                val_loss += losses.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_detr_model.pth')
            print("Saved new best model")

    return model
