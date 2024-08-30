import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

class Trainer:
    def __init__(self, model, train_loader, test_loader, lr=0.001, device="cuda", checkpoint_dir="checkpoints"):
        self.model = model.to(device)  # Model przenoszony na urządzenie
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device  # Urządzenie, na którym będą wykonywane operacje
        self.checkpoint_dir = checkpoint_dir  # Katalog do zapisywania checkpointów

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def save_checkpoint(self, epoch):
        """Zapisz checkpoint modelu."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.criterion
        }, checkpoint_path)
        print(f"Model checkpoint saved at epoch {epoch} to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Załaduj checkpoint modelu."""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Model loaded from checkpoint {checkpoint_path}, epoch {epoch}")
        return epoch, loss

    def save_model(self, save_path):
        """Zapisz wytrenowany model do pliku."""
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def train(self, epochs=1000):
        for epoch in range(epochs):
            print(f"Starting Epoch {epoch}")  # Wypisanie numeru epoki
            self.model.train()
            train_losses = []
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)  # Przeniesienie danych na urządzenie
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            # Wypisanie wyniku treningu po każdej epoce
            print(f"Epoch {epoch}, Train Loss: {np.mean(train_losses)}")

            # Zapisywanie checkpointu co 40 epok
            if epoch % 40 == 0:
                self.evaluate(epoch)
                self.save_checkpoint(epoch)

    def evaluate(self, epoch):
        self.model.eval()
        test_losses = []
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)  # Przeniesienie danych na urządzenie
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_losses.append(loss.item())
        print(f"Epoch {epoch}, Test Loss: {np.mean(test_losses)}")

    def test(self):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)  # Przeniesienie danych na urządzenie
                outputs = self.model(inputs)
                preds = outputs.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())  # Przenoszenie wyników z GPU na CPU
                all_labels.extend(labels.cpu().numpy())  # Przenoszenie wyników z GPU na CPU
        return all_preds, all_labels
