import os
from medmnist import DermaMNIST
from torchvision import transforms
from PIL import Image
import numpy as np

# Create folders
os.makedirs("dataset/benign", exist_ok=True)
os.makedirs("dataset/melanoma", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((64, 64))
])

dataset = DermaMNIST(split="train", download=True)

print("Preparing dataset...")

for i, (img, label) in enumerate(dataset):

    # Ensure correct image format
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.array(img))

    img = transform(img)

    # FIXED LABEL HANDLING
    if int(label[0]) == 0:
        cls = "benign"
    else:
        cls = "melanoma"

    img.save(f"dataset/{cls}/{i}.png")

print("Dataset ready!")