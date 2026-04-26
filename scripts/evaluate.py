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

acc, f1 = compute_metrics(preds, labels)
cm = confusion_matrix(preds, labels)

print(f"Accuracy: {acc:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Confusion Matrix:")
print(cm)