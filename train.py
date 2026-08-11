import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import SpatiotemporalMAE

def train_mae():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, _, _, _, _, _ = get_dataloaders(batch_size=1)

    model = SpatiotemporalMAE(mask_ratio=0.75).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 15
    print("Starting MAE Pre-training...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            reconstructed, mask = model(batch)
            # Calculate loss ONLY on the 75% masked pixels
            loss = criterion(reconstructed * (1 - mask), batch * (1 - mask))

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {running_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    train_mae()