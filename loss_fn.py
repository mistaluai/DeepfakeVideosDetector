import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, class_real=364, class_fake=3068):
        super(Loss, self).__init__()
        num_samples = class_real + class_fake
        self.real_weight = num_samples / (2 * class_real)
        self.fake_weight = num_samples / (2 * class_fake)
        weights = torch.tensor([self.real_weight, self.fake_weight], dtype=torch.float)
        self.loss = nn.CrossEntropyLoss(weight=weights)

    def forward(self, outputs, target):
        return self.loss(outputs, target)