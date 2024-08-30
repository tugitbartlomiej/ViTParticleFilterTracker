from data_handler import DatasetHandler
from trainer import Trainer
from vision_transformer import VisionTransformer
import torch

# Sprawdzenie dostępności CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the dataset handler
dataset_handler = DatasetHandler()
dataset_handler.show_images()

# Get data loaders
train_loader, test_loader = dataset_handler.get_data_loaders()

# Initialize the model
model = VisionTransformer()

# Initialize the trainer with the device
trainer = Trainer(model, train_loader, test_loader, lr=0.001, device=device)

# Train the model
trainer.train(epochs=100)
trainer.save_model('final_model.pth')


# Test the model
preds, labels = trainer.test()

dataset_handler = DatasetHandler()
dataset_handler.show_images(save_path='output_image.png')

print("Predictions:", preds)
print("Actual Labels:", labels)
