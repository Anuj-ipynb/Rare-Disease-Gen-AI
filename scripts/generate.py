import torch
import os
from torchvision.utils import save_image
from models.model_vae import CVAE

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = CVAE(num_classes=2, latent_dim=16).to(device)
model.load_state_dict(torch.load("checkpoints/cvae.pth", map_location=device))
model.eval()

os.makedirs("generated", exist_ok=True)

# Generate samples
for i in range(10):
    z = torch.randn(1, model.latent_dim).to(device)

    # Try both classes
    label = torch.tensor([[1, 0]]).float().to(device)  # benign
    img = model.sample(z, label)
    save_image(img, f"generated/benign_{i}.png")

    label = torch.tensor([[0, 1]]).float().to(device)  # melanoma
    img = model.sample(z, label)
    save_image(img, f"generated/melanoma_{i}.png")

print("Samples generated in /generated folder")