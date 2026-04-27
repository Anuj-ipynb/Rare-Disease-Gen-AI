import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
from models.classifier import CNN
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import numpy as np

# =========================
# Config
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
USE_AUGMENTED = True

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
    full_dataset = ConcatDataset([real_data, aug_data])
    print("Using augmented dataset")
else:
    full_dataset = real_data
    print("Using real dataset only")

# 🔥 Split dataset
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# =========================
# Model
# =========================
model = CNN(num_classes=2).to(device)

# 🔥 Compute class weights dynamically
targets = [y for _, y in real_data.samples]
class_counts = np.bincount(targets)
class_weights = 1. / class_counts
weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = torch.nn.CrossEntropyLoss(weight=weights)
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