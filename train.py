import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import SpatiotemporalMAE

def train_mae():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}") # just to check if gpu is actually working

    train_loader, val_loader, _, _, _, _ = get_dataloaders(batch_size=1)

    mae_net = SpatiotemporalMAE(mask_ratio=0.75).to(device)
    loss_fn = nn.MSELoss()
    
    # Adam worked way better than SGD here
    # opt = optim.SGD(mae_net.parameters(), lr=0.01)
    opt = optim.Adam(mae_net.parameters(), lr=1e-3)

    total_epochs = 15
    print("Kicking off MAE Pre-training...")

    for epoch in range(total_epochs):
        mae_net.train()
        epoch_loss_tracker = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            opt.zero_grad()

            recon_out, mask_tensor = mae_net(batch)
            
            # Compute loss ONLY on the hidden 75%
            # (1 - mask_tensor) selects the masked pixels
            actual_loss = loss_fn(recon_out * (1 - mask_tensor), batch * (1 - mask_tensor))

            actual_loss.backward()
            opt.step()
            epoch_loss_tracker += actual_loss.item()

        print(f"--> Done with epoch {epoch+1}, average loss: {epoch_loss_tracker/len(train_loader):.5f}")

    print("Starting validation check on unseen patients...")
    mae_net.eval()
    total_val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon_out, mask_tensor = mae_net(batch)
            
            val_err = loss_fn(recon_out * (1 - mask_tensor), batch * (1 - mask_tensor))
            total_val_loss += val_err.item()

    final_val = total_val_loss / len(val_loader)
    print(f"Validation Loss (Unseen): {final_val:.5f}")

    # save the weights
    torch.save(mae_net.state_dict(), "mae_frozen_encoder.pth")
    print("Saved weights to mae_frozen_encoder.pth!")

if __name__ == "__main__":
    train_mae()