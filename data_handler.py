import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import Resize, ToTensor, Compose

class DatasetHandler:
    def __init__(self, root=".", img_size=144):
        transform = Compose([Resize((img_size, img_size)), ToTensor()])
        self.dataset = OxfordIIITPet(root=root, download=True, transform=transform)

    def get_data_loaders(self, train_split=0.8, batch_size=32):
        train_size = int(train_split * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader

    def show_images(self, num_samples=40, cols=8, save_path=None):
        plt.figure(figsize=(15, 15))
        idx = int(len(self.dataset) / num_samples)
        for i in range(num_samples):
            img, _ = self.dataset[i * idx]
            plt.subplot(int(num_samples / cols) + 1, cols, i + 1)
            plt.imshow(img.permute(1, 2, 0))

        if save_path:
            plt.savefig(save_path)  # Zapisz obraz do pliku
            plt.close()  # Zamknij wykres po zapisaniu
        else:
            plt.show()  # Wyświetl obraz na ekranie
