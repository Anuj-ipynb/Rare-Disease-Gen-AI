import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from models.model_vae import CVAE
import os

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create checkpoint folder if not exists
os.makedirs("checkpoints", exist_ok=True)

# Transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Dataset
dataset = datasets.ImageFolder("dataset", transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

num_classes = len(dataset.classes)

# Model
model = CVAE(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(15):
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # One-hot encoding
        onehot = F.one_hot(y, num_classes).float()

        # Expand for image conditioning
        onehot_img = onehot.unsqueeze(-1).unsqueeze(-1)
        onehot_img = onehot_img.expand(-1, -1, 64, 64)

        # Encoder input
        encoder_input = torch.cat([x, onehot_img], dim=1)

        # Forward
        recon, mu, logvar = model(encoder_input, onehot)

        # Loss
        recon_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), "checkpoints/cvae.pth")

print("Training complete. Model saved.")