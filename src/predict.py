import torch
from src.model import build_model
from torchvision import transforms, models
from PIL import Image
import numpy as np
import argparse
import warnings
import json
import os

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

FEEDBACK_FILE = "feedback_corrections.json"


# ----------------------------------------------------
# Load or Create Feedback File
# ----------------------------------------------------
def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return []


def save_feedback(data):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ----------------------------------------------------
# Load Model
# ----------------------------------------------------
def load_model(checkpoint_path):
    model = build_model(backbone="resnet18", num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    clean_state = {k.replace("model.", "").replace("net.", ""): v
                   for k, v in checkpoint.items()}

    model.load_state_dict(clean_state, strict=False)
    model.to(device)
    model.eval()
    return model


# ----------------------------------------------------
# Embedding extractor using pretrained ResNet18
# ----------------------------------------------------
embed_model = models.resnet18(pretrained=True)
embed_model.fc = torch.nn.Identity()
embed_model.to(device)
embed_model.eval()

embed_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def get_embedding(img):
    img = embed_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = embed_model(img).cpu().numpy()[0]
    return emb / np.linalg.norm(emb)


# ----------------------------------------------------
# Transform pipeline
# ----------------------------------------------------
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ----------------------------------------------------
# Multi-crop inference
# ----------------------------------------------------
def generate_crops(img, crop_size=224):
    w, h = img.size
    if w < crop_size or h < crop_size:
        img = img.resize((max(w, crop_size), max(h, crop_size)))

    crops = [
        img.crop((0, 0, crop_size, crop_size)),
        img.crop((w - crop_size, 0, w, crop_size)),
        img.crop((0, h - crop_size, crop_size, h)),
        img.crop((w - crop_size, h - crop_size, w, h))
    ]

    cx = (w - crop_size) // 2
    cy = (h - crop_size) // 2
    crops.append(img.crop((cx, cy, cx + crop_size, cy + crop_size)))

    return crops


# ----------------------------------------------------
# Apply feedback correction
# ----------------------------------------------------
def apply_feedback_corrections(probs, embedding, feedback):
    for fb in feedback:
        fb_emb = np.array(fb["embedding"])
        similarity = np.dot(embedding, fb_emb)  # cosine similarity

        if similarity > 0.80:  # Only apply if very similar
            correction = np.array(fb["correction"])
            probs = probs + correction

    probs = np.maximum(probs, 1e-8)
    probs /= probs.sum()
    return probs


# ----------------------------------------------------
# Main Predict Function
# ----------------------------------------------------
def predict_image(model, image_path):
    img = Image.open(image_path).convert("RGB")

    crops = generate_crops(img)
    preds = []

    for crop in crops:
        tensor_img = base_transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor_img)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            preds.append(probs)

    preds = np.stack(preds)
    mean_probs = preds.mean(axis=0)
    smooth_probs = mean_probs ** 1.2
    smooth_probs /= smooth_probs.sum()

    # get embedding for feedback system
    embedding = get_embedding(img)

    # apply stored feedback corrections
    feedback = load_feedback()
    final_probs = apply_feedback_corrections(smooth_probs, embedding, feedback)

    pred_class = int(np.argmax(final_probs))
    return pred_class, final_probs, embedding


# ----------------------------------------------------
# CLI
# ----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="outputs/best.ckpt")
    args = parser.parse_args()

    model = load_model(args.checkpoint)

    pred, probs, embedding = predict_image(model, args.image)

    labels = ["REAL", "FAKE"]
    print("\n🧠 Prediction:", labels[pred])
    print(f"🔢 REAL={probs[0]:.4f} | FAKE={probs[1]:.4f}")

    # -------------------------------
    # USER FEEDBACK SECTION
    # -------------------------------
    ans = input("\n❓ Is my prediction correct? (yes/no): ").strip().lower()

    if ans == "no":
        correct_label = input("👉 What is the correct label? (REAL/FAKE): ").strip().upper()
        correct_idx = 0 if correct_label == "REAL" else 1

        correction = np.zeros(2)
        correction[correct_idx] += 0.25
        correction[1 - correct_idx] -= 0.25

        feedback = load_feedback()
        feedback.append({
            "embedding": embedding.tolist(),
            "correction": correction.tolist()
        })

        save_feedback(feedback)

        print("\n✅ Feedback saved! I will learn from this mistake.\n")

    else:
        print("👍 Great! No adjustment needed.\n")
