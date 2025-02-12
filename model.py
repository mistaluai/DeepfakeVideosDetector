import torch
import torch.nn as nn
import torchvision
from torch.jit import script, trace, trace_module
from torchvision.models import ResNet50_Weights

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        cnn = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(cnn.children())[:-1])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling to (B, 2048, 1, 1)

    def forward(self, x):
        batch_size, frames, C, H, W = x.shape
        feature_list = []

        for t in range(frames):
            frame = x[:, t, :, :, :]  # (Batch, C, H, W)
            features = self.feature_extractor(frame)  # (Batch, 2048, H', W')
            features = self.pool(features).squeeze(-1).squeeze(-1)  # (Batch, 2048)
            feature_list.append(features)

        return torch.stack(feature_list, dim=1)  # (Batch, Frames, 2048)


class LSTMPredictor(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=3, num_classes=2, example_input=None):
        super(LSTMPredictor, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )
        example_input = torch.randn(example_input['batch_size'],
                                    example_input['frames'],
                                    example_input['c'],
                                    example_input['H'],
                                    example_input['W'])


        self.feature_extractor_compiled = trace_module(self.feature_extractor, {'forward': example_input})

    def forward(self, x):
        print(x.shape)
        features = self.feature_extractor_compiled(x)  # (Batch, Frames, 2048)

        lstm_out, _ = self.lstm(features)  # (Batch, Frames, Hidden_Dim)
        out = self.fc(lstm_out[:, -1, :])  # Take the last LSTM output
        return out