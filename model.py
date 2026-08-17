import torch
import torch.nn as nn

class SpatiotemporalMAE(nn.Module):
    # Processes data: (Batch, Channels, Time, Height, Width)
    def __init__(self, mask_ratio=0.75, embed_dim=128):
        super().__init__()
        self.mask_ratio = mask_ratio

        # 3D Encoder
        # Tried 3x3x3 kernel first, but it blurred the temporal axis way too much
        # self.encoder = nn.Sequential(nn.Conv3d(1, 32, kernel_size=3)...)
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(32, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU()
        )

        # 3D Decoder to rebuild the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 4D tensors are confusing... gotta permute to (B, C, T, H, W) for Conv3d to work
        x_permuted = x.permute(0, 2, 1, 3, 4)

        # apply random mask if training
        if self.training:
            # random tensor same shape as x
            binary_mask = (torch.rand_like(x_permuted) > self.mask_ratio).float()
            inputs_masked = x_permuted * binary_mask
        else:
            inputs_masked = x_permuted
            binary_mask = torch.ones_like(x_permuted)

        latent_feats = self.encoder(inputs_masked)
        recon_out = self.decoder(latent_feats)

        # Fix dimension mismatch - had a bug here for days
        # Force it back to original shape via interpolation
        recon_out = nn.functional.interpolate(
            recon_out,
            size=(x_permuted.shape[2], x_permuted.shape[3], x_permuted.shape[4]),
            mode='trilinear',
            align_corners=False
        )

        # permute back to (B, T, C, H, W)
        return recon_out.permute(0, 2, 1, 3, 4), binary_mask.permute(0, 2, 1, 3, 4)