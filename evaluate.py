import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import SpatiotemporalMAE

def visualize_reconstruction():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, val_loader, _, _, _, _ = get_dataloaders(batch_size=1)
    
    # Initialize model
    model = SpatiotemporalMAE(mask_ratio=0.75).to(device)
    
    # Load the trained weights from Day 3
    if os.path.exists("mae_frozen_encoder.pth"):
        model.load_state_dict(torch.load("mae_frozen_encoder.pth", map_location=device))
        print("Loaded trained model weights.")
    else:
        print("Warning: No weights found. Model is untrained.")
        
    model.eval()
    
    # Create an assets folder to save the image
    os.makedirs('assets', exist_ok=True)
    
    with torch.no_grad():
        # Grab a single batch from the unseen validation patients
        sample_batch = next(iter(val_loader)).to(device)
        
        # Generate a new 75% mask strictly for visualization
        visual_mask = (torch.rand_like(sample_batch) > 0.75).float()
        masked_batch = sample_batch * visual_mask
        
        # Manually pass the masked batch through the encoder and decoder
        latent = model.encoder(masked_batch.permute(0, 2, 1, 3, 4))
        reconstructed = model.decoder(latent)
        
        # Interpolate to ensure exact dimension matching
        reconstructed = nn.functional.interpolate(
            reconstructed, 
            size=(sample_batch.shape[2], sample_batch.shape[3], sample_batch.shape[4]),
            mode='trilinear', 
            align_corners=False
        ).permute(0, 2, 1, 3, 4)
        
        # Select the middle frame of the sequence for the plot
        frame_idx = sample_batch.shape[1] // 2
        
        orig_img = sample_batch[0, frame_idx, 0].cpu().numpy()
        mask_img = visual_mask[0, frame_idx, 0].cpu().numpy()
        recon_img = reconstructed[0, frame_idx, 0].cpu().numpy()
        
        # Apply the mask to the original image for the middle plot
        masked_input = orig_img * mask_img
        
        # Plotting the 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(orig_img, cmap='gray')
        axes[0].set_title('Original Frame')
        axes[0].axis('off')
        
        axes[1].imshow(masked_input, cmap='gray')
        axes[1].set_title('Masked Input (75% Hidden)')
        axes[1].axis('off')
        
        axes[2].imshow(recon_img, cmap='gray')
        axes[2].set_title('MAE Reconstruction')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save the figure to the assets folder
        save_path = os.path.join('assets', 'reconstruction_plot.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualization saved successfully to {save_path}")

if __name__ == "__main__":
    visualize_reconstruction()