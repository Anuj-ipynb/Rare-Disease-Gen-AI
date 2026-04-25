import torch
import os
from torchvision.utils import save_image
from models.model_vae import CVAE

device = "cuda" if torch.cuda.is_available() else "cpu"

num_classes = 2
model = CVAE(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("checkpoints/cvae.pth", map_location=device))
model.eval()

os.makedirs("augmented_dataset/melanoma", exist_ok=True)

with torch.no_grad():
    for i in range(100):
        z = torch.randn(1, 32).to(device)
        label = torch.tensor([[1, 0]]).float().to(device)

        img = model.sample(z, label)
        save_image(img, f"augmented_dataset/melanoma/{i}.png")