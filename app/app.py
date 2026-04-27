import gradio as gr
import torch
from PIL import Image
import os
from torchvision import transforms, datasets
import numpy as np

from models.classifier import CNN
from models.model_vae import CVAE
from utils.metrics import compute_metrics, confusion_matrix

# =========================
# Setup
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 64
LATENT_DIM = 8   # ✅ FIXED (must match training)
NUM_CLASSES = 2

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

label_map = {0: "Benign", 1: "Melanoma"}

# =========================
# Load Models
# =========================
classifier = CNN(num_classes=NUM_CLASSES).to(device)
cvae = CVAE(num_classes=NUM_CLASSES, latent_dim=LATENT_DIM).to(device)

def load_models():
    cls_loaded = False
    gen_loaded = False

    if os.path.exists("checkpoints/classifier.pth"):
        classifier.load_state_dict(torch.load("checkpoints/classifier.pth", map_location=device))
        classifier.eval()
        cls_loaded = True

    if os.path.exists("checkpoints/cvae.pth"):
        cvae.load_state_dict(torch.load("checkpoints/cvae.pth", map_location=device))
        cvae.eval()
        gen_loaded = True

    return cls_loaded, gen_loaded

cls_loaded, gen_loaded = load_models()

# =========================
# Saliency Map (FIXED)
# =========================
def get_saliency(img_tensor):
    classifier.eval()
    img_tensor = img_tensor.clone().detach().requires_grad_(True)

    out = classifier(img_tensor)
    pred_class = out.argmax(dim=1)

    loss = out[0, pred_class]
    loss.backward()

    saliency = img_tensor.grad.abs().max(dim=1)[0]
    return saliency.squeeze().cpu().numpy()

# =========================
# Prediction
# =========================
def predict(image):
    if not cls_loaded:
        return None, "Classifier not trained.", None

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = classifier(img)
        prob = torch.softmax(out, dim=1)
        pred = prob.argmax().item()
        conf = prob[0][pred].item()

    saliency = get_saliency(img)

    return (
        {label_map[pred]: float(conf)},
        f"{label_map[pred]} ({conf:.2f})",
        saliency
    )

# =========================
# Generate Synthetic Images
# =========================
def generate_samples(class_idx, num_samples):
    if not gen_loaded:
        return None, "Generator not trained yet."

    class_idx = int(class_idx)
    num_samples = int(num_samples)

    label = torch.zeros(1, NUM_CLASSES).to(device)
    label[0, class_idx] = 1

    images = []

    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(1, LATENT_DIM).to(device)
            img = cvae.sample(z, label)

            img = torch.clamp(img, 0, 1)  # ✅ FIX
            img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
            images.append(img_np)

    return images, "Generated successfully."

# =========================
# Metrics Evaluation
# =========================
def evaluate_model():
    if not cls_loaded:
        return "Train classifier first.", ""

    dataset = datasets.ImageFolder("dataset", transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)

    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = classifier(x)
            p = out.argmax(1).cpu().tolist()

            preds.extend(p)
            labels.extend(y.tolist())

    # ✅ FIXED ORDER
    metrics = compute_metrics(labels, preds)
    cm = confusion_matrix(labels, preds)

    result_text = f"""
Accuracy : {metrics['accuracy']:.3f}
Precision: {metrics['precision']:.3f}
Recall   : {metrics['recall']:.3f}
F1 Score : {metrics['f1']:.3f}
"""

    cm_text = f"""
Confusion Matrix:

          Pred 0   Pred 1
Actual 0   {cm[0][0]}       {cm[0][1]}
Actual 1   {cm[1][0]}       {cm[1][1]}
"""

    return result_text, cm_text

# =========================
# UI
# =========================
with gr.Blocks() as app:

    gr.Markdown("# 🏥 Rare Disease AI Assistant")
    gr.Markdown("⚠️ This is NOT a medical diagnosis tool")

    # -------- Prediction Tab --------
    with gr.Tab("Prediction"):
        input_img = gr.Image(type="pil", label="Upload Image")

        conf_bar = gr.Label(label="Confidence")
        pred_text = gr.Textbox(label="Prediction")
        saliency_map = gr.Image(label="Saliency Map")

        pred_btn = gr.Button("Predict + Explain")

        pred_btn.click(
            predict,
            inputs=input_img,
            outputs=[conf_bar, pred_text, saliency_map]
        )

    # -------- Generation Tab --------
    with gr.Tab("Generate Synthetic Data"):
        class_input = gr.Radio(
            choices=[("Benign", 0), ("Melanoma", 1)],
            label="Select Class",
            value=1
        )

        num_input = gr.Slider(1, 10, value=5, step=1, label="Number of Samples")

        gen_btn = gr.Button("Generate")

        gen_gallery = gr.Gallery(label="Generated Images")
        gen_status = gr.Textbox(label="Status")

        gen_btn.click(
            generate_samples,
            inputs=[class_input, num_input],
            outputs=[gen_gallery, gen_status]
        )

    # -------- Metrics Tab --------
    with gr.Tab("Model Metrics"):
        metrics_output = gr.Textbox(label="Metrics", lines=6)
        cm_output = gr.Textbox(label="Confusion Matrix", lines=6)

        eval_btn = gr.Button("Evaluate Model")

        eval_btn.click(
            evaluate_model,
            inputs=[],
            outputs=[metrics_output, cm_output]
        )

# =========================
# Launch
# =========================
app.launch()