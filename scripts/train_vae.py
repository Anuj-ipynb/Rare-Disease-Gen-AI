import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.model_vae import CVAE
import os

# ===== SETUP =====
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("checkpoints", exist_ok=True)

# ===== TRANSFORMS =====
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()  # keep [0,1]
])

# ===== DATASET =====
dataset = datasets.ImageFolder("dataset", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

num_classes = len(dataset.classes)

# ===== MODEL =====
model = CVAE(num_classes=num_classes, latent_dim=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# ===== TRAINING =====
for epoch in range(20):
    model.train()
    total_loss = 0

    beta = 0.1 

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # One-hot encoding
        onehot = F.one_hot(y, num_classes).float()

    
        onehot_img = onehot.unsqueeze(-1).unsqueeze(-1)
        onehot_img = onehot_img.expand(-1, -1, 64, 64)

        # Concatenate image + label
        encoder_input = torch.cat([x, onehot_img], dim=1)

        # Forward pass
        recon, mu, logvar = model(encoder_input, onehot)

        # ===== LOSS (FIXED) =====
        recon_loss = F.binary_cross_entropy(recon, x, reduction='mean')

        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        loss = recon_loss + beta * kl_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

# ===== SAVE MODEL =====
torch.save(model.state_dict(), "checkpoints/cvae.pth")
print("Training complete. Model saved.")