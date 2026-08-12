import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import SpatiotemporalMAE

def train_mae():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Data Splits
    train_loader, val_loader, _, _, _, _ = get_dataloaders(batch_size=1)

    # 2. Initialize Architecture & Loss
    model = SpatiotemporalMAE(mask_ratio=0.75).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 15
    print("Starting MAE Pre-training...")

    # 3. Pre-Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            reconstructed, mask = model(batch)
            # Compute loss ONLY on the masked regions (unseen 75%)
            loss = criterion(reconstructed * (1 - mask), batch * (1 - mask))

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {running_loss/len(train_loader):.4f}")

    # 4. Validation Evaluation (Unseen Patients)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            reconstructed, mask = model(batch)
            loss = criterion(reconstructed * (1 - mask), batch * (1 - mask))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"\nValidation Reconstruction Loss (Unseen Patients): {avg_val_loss:.4f}")

    # 5. Save Weights Checkpoint
    torch.save(model.state_dict(), "mae_frozen_encoder.pth")
    print("Model checkpoint saved to mae_frozen_encoder.pth")

if __name__ == "__main__":
    train_mae()