import torch
from torchvision import datasets, transforms
from models.classifier import CNN
from utils.metrics import compute_metrics, confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("dataset", transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=16)

model = CNN(num_classes=2).to(device)
model.load_state_dict(torch.load("checkpoints/classifier.pth", map_location=device))
model.eval()

preds, labels = [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        p = out.argmax(1).cpu().tolist()

        preds.extend(p)
        labels.extend(y.tolist())

# Compute metrics
metrics = compute_metrics(preds, labels)
cm = confusion_matrix(preds, labels)

# Print nicely
print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy : {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall   : {metrics['recall']:.3f}")
print(f"F1 Score : {metrics['f1']:.3f}")

print("\nConfusion Matrix:")
print("         Pred 0   Pred 1")
print(f"Actual 0   {cm[0][0]}       {cm[0][1]}")
print(f"Actual 1   {cm[1][0]}       {cm[1][1]}")