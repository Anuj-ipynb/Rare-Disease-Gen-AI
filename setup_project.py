import os

# Folder structure
folders = [
    "models",
    "scripts",
    "app",
    "dataset",
    "augmented_dataset",
    "checkpoints"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Files with content
files = {
    "models/model_vae.py": """import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, num_classes=2, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3 + num_classes, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(32 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(32 * 16 * 16, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim + num_classes, 32 * 16 * 16)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, labels):
        z = torch.cat([z, labels], dim=1)
        h = self.fc_decode(z).view(-1, 32, 16, 16)
        return self.decoder(h)

    def forward(self, x, labels):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

    def sample(self, z, labels):
        return self.decode(z, labels)""",
    "models/classifier.py": """import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)""",

    "scripts/train_vae.py": """import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from models.model_vae import CVAE

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("dataset", transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

model = CVAE(num_classes=len(dataset.classes)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(15):
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        onehot = F.one_hot(y, len(dataset.classes)).float()

        onehot_img = onehot.unsqueeze(-1).unsqueeze(-1)
        onehot_img = onehot_img.expand(-1, -1, 64, 64)

        encoder_input = torch.cat([x, onehot_img], dim=1)

        recon, mu, logvar = model(encoder_input, onehot)

        loss = F.mse_loss(recon, x) + \
               -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} Loss {loss.item():.4f}")

torch.save(model.state_dict(), "checkpoints/cvae.pth")""",

    "scripts/generate.py": """import torch
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
        save_image(img, f"augmented_dataset/melanoma/{i}.png")""",
    "scripts/train_classifier.py": """import torch
from torchvision import datasets, transforms
from models.classifier import CNN

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("dataset", transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

model = CNN(num_classes=len(dataset.classes)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    print(f"Epoch {epoch} Accuracy {correct/total:.2f}")

torch.save(model.state_dict(), "checkpoints/classifier.pth")""",
    "app/app.py": """import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from models.classifier import CNN

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN(num_classes=2).to(device)

try:
    model.load_state_dict(torch.load("checkpoints/classifier.pth", map_location=device))
except:
    print("⚠️ Train classifier first!")

model.eval()

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

def predict(img):
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)
        pred = prob.argmax().item()

    return f"Prediction: {pred} | Confidence: {prob[0][pred]:.2f}"

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Rare Disease AI",
    description="⚠️ This is NOT a medical diagnosis tool"
).launch()""",
    "requirements.txt": "torch\ntorchvision\ngrady\ngradio\npillow\npandas"
}

for path, content in files.items():
    with open(path, "w") as f:
        f.write(content)

print("✅ Project created successfully!")