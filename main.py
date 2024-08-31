from data_handler import DatasetHandler
from trainer import Trainer
from vision_transformer import VisionTransformer
import torch

# Sprawdzenie dostępności CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the dataset handler with appropriate image size for VisionTransformer
image_size = 256  # Upewnij się, że rozmiar obrazu jest zgodny z parametrami modelu
dataset_handler = DatasetHandler(img_size=image_size)
dataset_handler.show_images()

# Get data loaders
train_loader, test_loader = dataset_handler.get_data_loaders()

# Initialize the model with appropriate parameters
model = VisionTransformer(
    image_size=image_size,   # Rozmiar obrazu
    patch_size=32,           # Rozmiar patcha
    num_classes=37,          # Liczba klas (dopasowana do datasetu)
    dim=512,                 # Wymiar embeddingu
    depth=6,                 # Głębokość Transformera (liczba warstw)
    heads=8,                 # Liczba głów w warstwie Attention
    mlp_dim=1024,            # Wymiar warstwy FeedForward
    channels=3               # Liczba kanałów (3 dla obrazów RGB)
).to(device)                 # Upewnij się, że model jest na GPU

# Initialize the trainer with the device
trainer = Trainer(model, train_loader, test_loader, lr=0.001, device=device)

# Train the model
trainer.train(epochs=100)
trainer.save_model('final_model.pth')

# Test the model
preds, labels = trainer.test()

# Show and save sample images with predictions
dataset_handler.show_images(save_path='output_image.png')

print("Predictions:", preds)
print("Actual Labels:", labels)
