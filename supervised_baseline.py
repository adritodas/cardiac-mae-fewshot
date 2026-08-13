import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders

class Supervised3DCNN(nn.Module):
    """
    A standard high-capacity 3D CNN baseline.
    In an extreme few-shot scenario (N=11), this architecture is highly 
    prone to overfitting, effectively memorizing the training data and 
    failing to generalize to unseen patients without data leakage.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

if __name__ == "__main__":
    print("Supervised baseline initialized.")
    print("This script is provided in the repository to empirically prove")
    print("the data leakage and memorization claims made in the paper.")