import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Definicja modelu Vision Transformer (ViT)
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_size=32, img_size=144):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x):
        return self.projection(x)

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(x.shape[0], -1, self.heads, t.shape[-1] // self.heads).permute(0, 2, 1, 3), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(x.shape[0], -1, self.heads * (x.shape[-1] // self.heads))
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dim_head=64, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ResidualAdd(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                ResidualAdd(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_size=32, img_size=144, depth=6, heads=8, mlp_dim=64, num_classes=37):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.transformer = Transformer(emb_size, depth, heads, mlp_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)

# Przygotowanie danych
transform = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.ToTensor(),
])

train_dataset = datasets.OxfordIIITPet(root='.', split='trainval', download=True, transform=transform)
test_dataset = datasets.OxfordIIITPet(root='.', split='test', download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Definicja klasy Trainer
class Trainer:
    def __init__(self, model, train_loader, test_loader, lr=0.001, device="cuda", checkpoint_dir="checkpoints"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.criterion
        }, checkpoint_path)
        print(f"Model checkpoint saved at epoch {epoch} to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Model loaded from checkpoint {checkpoint_path}, epoch {epoch}")
        return epoch, loss

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def train(self, epochs=1000):
        for epoch in range(epochs):
            print(f"Starting Epoch {epoch}")
            self.model.train()
            train_losses = []
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            print(f"Epoch {epoch}, Train Loss: {np.mean(train_losses)}")

            if epoch % 40 == 0:
                self.evaluate(epoch)
                self.save_checkpoint(epoch)

    def evaluate(self, epoch):
        self.model.eval()
        test_losses = []
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_losses.append(loss.item())
        print(f"Epoch {epoch}, Test Loss: {np.mean(test_losses)}")

    def test(self):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                preds = outputs.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return all_preds, all_labels

# Mapowanie klas na etykiety i wyświetlanie obrazów z przewidywaniami
def get_class_labels():
    return {
        0: "Abyssinian", 1: "American Bulldog", 2: "American Pit Bull Terrier", 3: "Bengal",
        4: "Birman", 5: "Bombay", 6: "British Shorthair", 7: "Chihuahua",
        8: "Egyptian Mau", 9: "English Cocker Spaniel", 10: "English Setter",
        11: "German Shorthaired", 12: "Great Pyrenees", 13: "Havanese",
        14: "Japanese Chin", 15: "Keeshond", 16: "Leonberger", 17: "Maine Coon",
        18: "Miniature Pinscher", 19: "Newfoundland", 20: "Persian",
        21: "Pomeranian", 22: "Pug", 23: "Ragdoll", 24: "Russian Blue",
        25: "Saint Bernard", 26: "Samoyed", 27: "Scottish Terrier",
        28: "Shiba Inu", 29: "Siamese", 30: "Sphynx", 31: "Staffordshire Bull Terrier",
        32: "Wheaten Terrier", 33: "Yorkshire Terrier", 34: "British Shorthair",
        35: "Russian Blue", 36: "Ragdoll"
    }

def display_predictions(model, test_loader, device="cuda"):
    class_labels = get_class_labels()
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=-1)

            for i in range(len(inputs)):
                plt.imshow(inputs[i].cpu().permute(1, 2, 0).numpy())
                plt.title(f"Predicted: {class_labels[preds[i].item()]}, Actual: {class_labels[labels[i].item()]}")
                plt.show()
            break

# Przykładowe użycie
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ViT()
trainer = Trainer(model, train_loader, test_loader, device=device)

# Trening modelu (dla przykładu tylko 5 epok, zmień na więcej)
trainer.train(epochs=5)

# Wyświetlenie wyników predykcji na obrazach testowych
display_predictions(trainer.model, trainer.test_loader, device=trainer.device)
