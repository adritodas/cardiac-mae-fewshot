import torch
import torch.nn as nn

class SpatiotemporalMAE(nn.Module):
    """
    Spatiotemporal Masked Autoencoder for 4D Cardiac MRI sequences.
    Processes data with dimensions: (Batch, Channels, Time, Height, Width).
    """
    def __init__(self, mask_ratio=0.75, embed_dim=128):
        super().__init__()
        self.mask_ratio = mask_ratio

        # 3D Encoder: Compresses spatial dimensions while preserving temporal depth
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(32, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU()
        )

        # 3D Decoder: Reconstructs the original sequence from the latent space
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Permute from (B, T, C, H, W) -> (B, C, T, H, W) for 3D convolutions
        x = x.permute(0, 2, 1, 3, 4)

        # Apply random spatiotemporal masking during training
        if self.training:
            mask = (torch.rand_like(x) > self.mask_ratio).float()
            masked_x = x * mask
        else:
            masked_x = x
            mask = torch.ones_like(x)

        # Forward pass through Encoder and Decoder
        latent = self.encoder(masked_x)
        reconstructed = self.decoder(latent)

        # Ensure output perfectly matches input dimensions via trilinear interpolation
        reconstructed = nn.functional.interpolate(
            reconstructed,
            size=(x.shape[2], x.shape[3], x.shape[4]),
            mode='trilinear',
            align_corners=False
        )

        # Permute back to (B, T, C, H, W) before returning
        return reconstructed.permute(0, 2, 1, 3, 4), mask.permute(0, 2, 1, 3, 4)