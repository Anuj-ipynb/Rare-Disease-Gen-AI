import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from models.classifier import CNN
import os

# =========================
# Config
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3

USE_AUGMENTED = True  # set False for "before" run

# =========================
# Data
# =========================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

real_data = datasets.ImageFolder("dataset", transform=transform)

if USE_AUGMENTED and os.path.exists("augmented_dataset"):
    aug_data = datasets.ImageFolder("augmented_dataset", transform=transform)
    dataset = ConcatDataset([real_data, aug_data])
    print("Using augmented dataset")
else:
    dataset = real_data
    print("Using real dataset only")

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================
# Model
# =========================
model = CNN(num_classes=2).to(device)

# 🔥 Class weights (important)
# benign=0, melanoma=1 → melanoma gets higher weight
weights = torch.tensor([1.0, 2.5]).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# Training
# =========================
for epoch in range(EPOCHS):
    correct = 0
    total = 0
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / total
    avg_loss = total_loss / len(loader)

    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {acc:.3f}")

# =========================
# Save
# =========================
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/classifier.pth")

print("Classifier training complete.")