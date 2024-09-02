import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)


# Patchify and model classes remain unchanged

def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[
                        :,
                        i * patch_size: (i + 1) * patch_size,
                        j * patch_size: (j + 1) * patch_size,
                        ]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches


class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.k_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.v_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class MyViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        super(MyViT, self).__init__()

        self.chw = chw
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        assert (
                chw[1] % n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        assert (
                chw[2] % n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        self.register_buffer(
            "positional_embeddings",
            get_positional_embeddings(n_patches ** 2 + 1, hidden_d),
            persistent=False,
        )

        self.blocks = nn.ModuleList(
            [MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)]
        )

        self.mlp = nn.Sequential(nn.Linear(self.hidden_d, out_d), nn.Softmax(dim=-1))

    def forward(self, images):
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)

        tokens = self.linear_mapper(patches)

        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        for block in self.blocks:
            out = block(out)

        out = out[:, 0]

        return self.mlp(out)


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = (
                np.sin(i / (10000 ** (j / d)))
                if j % 2 == 0
                else np.cos(i / (10000 ** ((j - 1) / d)))
            )
    return result


def main(start_checkpoint=None, test_checkpoint=None):
    # Configuring paths
    checkpoint_dir = "checkpoints"
    results_dir = "results"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Loading data
    transform = ToTensor()

    train_set = MNIST(
        root="./../datasets", train=True, download=True, transform=transform
    )
    test_set = MNIST(
        root="./../datasets", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Using device: ",
        device,
        f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "",
    )
    model = MyViT(
        (1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10
    ).to(device)
    optimizer = Adam(model.parameters(), lr=0.005)
    criterion = CrossEntropyLoss()

    # Load model from checkpoint if specified
    start_epoch = 0
    if start_checkpoint is not None:
        checkpoint = torch.load(start_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from checkpoint {start_checkpoint}, epoch {start_epoch}")

    # Test model directly from checkpoint if specified
    if test_checkpoint is not None:
        checkpoint = torch.load(test_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Testing model from checkpoint {test_checkpoint}")
        test(model, test_loader, criterion, device, results_dir)
        return

    # Training loop
    N_EPOCHS = 5
    for epoch in trange(start_epoch, N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(
                train_loader, desc=f"Epoch {epoch + 1} in training", leave=False
        ):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"))
        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

    # Test model after training
    test(model, test_loader, criterion, device, results_dir)


def test(model, test_loader, criterion, device, results_dir):
    model.eval()
    correct, total = 0, 0
    test_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
    accuracy = correct / total * 100
    print(f"Test loss: {test_loss:.2f}")
    print(f"Test accuracy: {accuracy:.2f}%")

    # Save results
    with open(os.path.join(results_dir, "test_results.txt"), "w") as f:
        f.write(f"Test loss: {test_loss:.2f}\n")
        f.write(f"Test accuracy: {accuracy:.2f}%\n")

    show_classification_results(model, test_loader, device, save_dir=results_dir)


def show_classification_results(model, test_loader, device, n_images=5, save_dir=None):
    model.eval()
    images_shown = 0

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            predictions = torch.argmax(y_hat, dim=1)

            for i in range(x.size(0)):
                if images_shown >= n_images:
                    return
                plt.imshow(x[i].cpu().squeeze(), cmap="gray")
                plt.title(
                    f"Predicted: {predictions[i].item()}, Actual: {y[i].item()}"
                )
                plt.axis("off")

                if save_dir:
                    plt.savefig(os.path.join(save_dir, f"classification_{images_shown + 1}.png"))
                else:
                    plt.show()
                images_shown += 1


if __name__ == "__main__":
    # Replace these with paths to your checkpoints or leave as None
    start_checkpoint_path = None  # Path to start checkpoint, or None to start fresh
    test_checkpoint_path = None  # Path to checkpoint for testing, or None to skip testing

    main(start_checkpoint=start_checkpoint_path, test_checkpoint=test_checkpoint_path)
