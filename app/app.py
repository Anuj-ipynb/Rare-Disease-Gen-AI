import gradio as gr
import torch
from PIL import Image
import os
from torchvision import transforms, utils
import numpy as np

from models.classifier import CNN
from models.model_vae import CVAE

# =========================
# Setup
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 64
LATENT_DIM = 32
NUM_CLASSES = 2

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# =========================
# Load Models Safely
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
# Prediction Function
# =========================
def predict(image):
    if not cls_loaded:
        return "Classifier not trained yet."

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = classifier(img)
        prob = torch.softmax(out, dim=1)
        pred = prob.argmax().item()
        conf = prob[0][pred].item()

    label_map = {0: "benign", 1: "melanoma"}
    return f"Prediction: {label_map.get(pred, pred)} | Confidence: {conf:.2f}"

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

            img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
            images.append(img_np)

    return images, "Generated successfully."

# =========================
# UI
# =========================
with gr.Blocks() as app:

    gr.Markdown("# 🏥 Rare Disease AI Assistant")

    # -------- Prediction Tab --------
    with gr.Tab("Prediction"):
        input_img = gr.Image(type="pil", label="Upload Image")
        pred_output = gr.Textbox(label="Prediction")

        pred_btn = gr.Button("Predict")
        pred_btn.click(predict, inputs=input_img, outputs=pred_output)

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

# Launch
app.launch()