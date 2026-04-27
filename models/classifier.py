import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_saliency(model, img):
    model.eval()
    img = img.clone().detach().requires_grad_(True)

    out = model(img)
    pred_class = out.argmax(dim=1)

    loss = out[0, pred_class]
    loss.backward()

    saliency = img.grad.abs().max(dim=1)[0]
    return saliency