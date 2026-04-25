import torch
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

torch.save(model.state_dict(), "checkpoints/classifier.pth")