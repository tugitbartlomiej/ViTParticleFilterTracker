import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ToolTipTracker(nn.Module):
    def __init__(self, hidden_size=256, num_classes=2):
        super(ToolTipTracker, self).__init__()
        # Inicjalizacja ResNet18 z nowymi wagami
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()  # Usunięcie ostatniej warstwy

        self.lstm = nn.LSTM(512, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # 2 dla współrzędnych x i y

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        cnn_out = self.cnn(x.view(batch_size * seq_len, c, h, w))
        cnn_out = cnn_out.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(cnn_out)

        output = self.fc(lstm_out)
        return output
