import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights


class CNN_LSTM(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=3, num_classes=2):
        super(CNN_LSTM, self).__init__()
        cnn = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(cnn.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling to (B, 2048, 1, 1)

        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        batch_size, frames, C, H, W = x.shape  # (Batch, Frames, C, H, W)

        x = x.view(batch_size * frames, C, H, W)  # (Batch × Frames, C, H, W)
        features = self.feature_extractor(x)
        features = self.pool(features).squeeze(-1).squeeze(-1)  # (Batch × Frames, 2048)

        features = features.view(batch_size, frames, -1)  # (Batch, Frames, Feature_Dim)
        lstm_out, _ = self.lstm(features)
        out = self.fc(lstm_out[:, -1, :])
        return out