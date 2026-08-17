import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import SpatiotemporalMAE

def generate_money_shot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, val_loader, _, _, _, _ = get_dataloaders(batch_size=1)
    
    mae_net = SpatiotemporalMAE(mask_ratio=0.75).to(device)
    
    if os.path.exists("mae_frozen_encoder.pth"):
        mae_net.load_state_dict(torch.load("mae_frozen_encoder.pth", map_location=device))
        print("Weights loaded successfully.")
    else:
        print("Yikes, no weights found. Are you sure you ran train.py?")
        
    mae_net.eval()
    os.makedirs('assets', exist_ok=True)
    
    with torch.no_grad():
        sample = next(iter(val_loader)).to(device)
        
        # generate a fresh mask just for this plot
        viz_mask = (torch.rand_like(sample) > 0.75).float()
        inputs_masked = sample * viz_mask
        
        # manual forward pass to grab the reconstruction
        latent = mae_net.encoder(inputs_masked.permute(0, 2, 1, 3, 4))
        recon_out = mae_net.decoder(latent)
        
        recon_out = nn.functional.interpolate(
            recon_out, 
            size=(sample.shape[2], sample.shape[3], sample.shape[4]),
            mode='trilinear', 
            align_corners=False
        ).permute(0, 2, 1, 3, 4)
        
        # grab the middle frame so we actually see the heart
        mid_frame = sample.shape[1] // 2
        
        img_orig = sample[0, mid_frame, 0].cpu().numpy()
        img_mask = viz_mask[0, mid_frame, 0].cpu().numpy()
        img_recon = recon_out[0, mid_frame, 0].cpu().numpy()
        
        img_masked_input = img_orig * img_mask
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # axes[0].imshow(img_orig, cmap='bone') # bone looked weird, sticking to gray
        axes[0].imshow(img_orig, cmap='gray')
        axes[0].set_title('Original Frame')
        axes[0].axis('off')
        
        axes[1].imshow(img_masked_input, cmap='gray')
        axes[1].set_title('Masked Input (75% Hidden)')
        axes[1].axis('off')
        
        axes[2].imshow(img_recon, cmap='gray')
        axes[2].set_title('MAE Reconstruction')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        out_path = os.path.join('assets', 'reconstruction_plot.png')
        plt.savefig(out_path, bbox_inches='tight')
        print(f"Saved the visualization to {out_path}!")

if __name__ == "__main__":
    generate_money_shot()