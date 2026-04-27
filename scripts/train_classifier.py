import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset
from models.classifier import CNN
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import numpy as np
from PIL import Image
import random

# =========================
# Config
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 10
LR = 5e-4

USE_GENERATED = True

# =========================
# Transforms
# =========================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# =========================
# Custom Dataset for generated images
# =========================
class GeneratedDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = []
        self.labels = []
        self.transform = transform

        for file in os.listdir(folder):
            if file.endswith(".png"):
                path = os.path.join(folder, file)

                if "benign" in file.lower():
                    label = 0
                elif "melanoma" in file.lower():
                    label = 1
                else:
                    continue

                self.paths.append(path)
                self.labels.append(label)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# =========================
# Balance Dataset Function
# =========================
import random

def balance_dataset(dataset):
    data = []

    for x, y in dataset:
        data.append((x, y))

    class0 = [d for d in data if d[1] == 0]
    class1 = [d for d in data if d[1] == 1]

    min_size = min(len(class0), len(class1))

    class0_sample = random.sample(class0, min_size)
    class1_sample = random.sample(class1, min_size)

    balanced = class0_sample + class1_sample
    random.shuffle(balanced)

    return balanced

# =========================
# Load REAL dataset
# =========================
real_data = datasets.ImageFolder("dataset", transform=transform)

# =========================
# Split ONLY real data
# =========================
train_size = int(0.8 * len(real_data))
val_size = len(real_data) - train_size

train_real, val_dataset = random_split(real_data, [train_size, val_size])

# =========================
# Combine + Balance
# =========================
if USE_GENERATED and os.path.exists("generated"):
    gen_data = GeneratedDataset("generated", transform=transform)
    combined = ConcatDataset([train_real, gen_data])
    print(f"Using generated data: {len(gen_data)} samples")
else:
    combined = train_real
    print("Using real data only")

print("Balancing dataset...")
balanced_data = balance_dataset(combined)

# =========================
# Loaders
# =========================
train_loader = DataLoader(balanced_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# =========================
# Model
# =========================
model = CNN(num_classes=2).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# Training
# =========================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # =========================
    # Validation
    # =========================
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            preds = out.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1}")
    print(f"Loss: {avg_loss:.4f} | Acc: {acc:.3f}")
    print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
    print("-"*40)

# =========================
# Save
# =========================
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/classifier.pth")

print("Classifier training complete.")