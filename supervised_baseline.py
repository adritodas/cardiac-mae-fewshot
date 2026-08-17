import torch
import torch.nn as nn

class SupervisedCNN_Overfitter(nn.Module):
    """
    This is my baseline CNN. 
    I built this just to prove my point that standard supervised nets 
    cheat/memorize the data when N=11.
    """
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.binary_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1, 3, 4)
        feats = self.feature_extractor(x_permuted)
        flattened = feats.flatten(1)
        return self.binary_head(flattened)

if __name__ == "__main__":
    print("Initialized the supervised baseline.")
    print("Run this to see how quickly it hits 100% train accuracy but fails validation.")