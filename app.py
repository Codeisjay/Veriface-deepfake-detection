import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ExifTags
from src.model import build_model
import json
import os
import imagehash
import numpy as np
import cv2
import hashlib
from io import BytesIO

# ===========================================================
# CONFIG
# ===========================================================
CHECKPOINT_PATH = "outputs/best.ckpt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["real", "fake"]
FEEDBACK_FILE = "feedback_memory.json"

# ===========================================================
# FEEDBACK MEMORY
# ===========================================================
def load_feedback_file():
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w") as f:
            json.dump({}, f)
        return {}
    with open(FEEDBACK_FILE, "r") as f:
        try:
            return json.load(f)
        except:
            return {}

def write_feedback_file(mem):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(mem, f, indent=4)

if "FEEDBACK_MEMORY" not in st.session_state:
    st.session_state.FEEDBACK_MEMORY = load_feedback_file()

# ===========================================================
# HASHING
# ===========================================================
def normalize_image_for_hash(img: Image.Image, size=(256, 256)):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = img._getexif()
        if exif:
            v = exif.get(orientation, None)
            if v == 3:
                img = img.rotate(180, expand=True)
            elif v == 6:
                img = img.rotate(270, expand=True)
            elif v == 8:
                img = img.rotate(90, expand=True)
    except:
        pass

    img = img.convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    bio = BytesIO()
    img.save(bio, format="PNG")
    raw = bio.getvalue()
    return img, raw

def get_image_hash(image: Image.Image):
    norm_img, raw = normalize_image_for_hash(image, size=(256, 256))
    p_hash = imagehash.phash(norm_img)
    md5 = hashlib.md5(raw).hexdigest()[:12]
    return f"{str(p_hash)}-{md5}"

# ===========================================================
# FORENSICS
# ===========================================================
def frequency_artifact_score(image):
    gray = np.array(image.convert("L"))
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    return float(np.mean(magnitude > np.percentile(magnitude, 99.5)))

def sharpness_score(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())

def noise_level(image):
    im = np.array(image.convert("L"))
    return float(np.std(im - cv2.medianBlur(im, 5)))

# ===========================================================
# LOAD MODEL
# ===========================================================
@st.cache_resource
def load_model():
    model = build_model(backbone="resnet18", num_classes=len(CLASSES))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    clean_state = {k.replace("model.", "").replace("module.", "").replace("net.", ""): v
                   for k, v in state_dict.items()}

    model.load_state_dict(clean_state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# ===========================================================
# PREPROCESSING
# ===========================================================
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

# ===========================================================
# PREDICTION
# ===========================================================
def predict_image(model, image):
    img_hash = get_image_hash(image)

    # Memory override
    if img_hash in st.session_state.FEEDBACK_MEMORY:
        return st.session_state.FEEDBACK_MEMORY[img_hash]["label"], 1.0, img_hash

    tensor = preprocess_image(image).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    pred_label = CLASSES[pred.item()]
    base_conf = float(conf.item())

    freq = frequency_artifact_score(image)
    sharp = sharpness_score(image)
    noise = noise_level(image)

    fake_score = 0
    if freq > 0.012: fake_score += 1
    if sharp < 120: fake_score += 1
    if noise < 3.0: fake_score += 1

    if fake_score >= 2:
        return "fake", max(base_conf, 0.90), img_hash

    if pred_label == "real" and base_conf < 0.60:
        return "fake", 0.75, img_hash

    return pred_label, base_conf, img_hash

# ===========================================================
# SAVE FEEDBACK
# ===========================================================
def save_feedback(img_hash, correct_label):
    st.session_state.FEEDBACK_MEMORY[img_hash] = {"label": correct_label}
    write_feedback_file(st.session_state.FEEDBACK_MEMORY)

# ===========================================================
# UI DESIGN
# ===========================================================
st.set_page_config(page_title="DeepFake Detector", layout="wide")

# --- TOP BANNER ---
st.markdown("""
<style>
.title {
    font-size: 48px; font-weight: 900;
    text-align: center;
    background: linear-gradient(to right, #00c6ff, #0072ff);
    -webkit-background-clip: text;
    color: transparent;
}
.sub {
    text-align:center;
    font-size:20px;
    color:#d7d7d7;
    margin-top:-10px;
}
.upload-box {
    border-radius: 15px;
    padding: 25px;
    background: #111827;
    border: 1px solid #1f2937;
}
.result-card {
    padding: 25px;
    border-radius: 20px;
    background: #0f172a;
    border: 1px solid #1e293b;
}
</style>
<p class="title">🕵️ DeepFake Detector</p>
<p class="sub">AI-powered Forensic Image Analysis with Learning Feedback</p>
""", unsafe_allow_html=True)

# --- MAIN LAYOUT ---
left, right = st.columns([1, 1])

with left:
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)

        with st.spinner("🧠 Analyzing image..."):
            model = load_model()
            pred, conf, img_hash = predict_image(model, image)

        st.subheader("🔍 Result")

        if pred == "real":
            st.success(f"Prediction: **REAL**\nConfidence: **{conf*100:.2f}%**")
        else:
            st.error(f"Prediction: **FAKE**\nConfidence: **{conf*100:.2f}%**")

        st.markdown("### ❓ Was this correct?")
        with st.form(key=f"fb_{img_hash}"):
            correction = st.radio("Correct label:", ["real", "fake"])
            submitted = st.form_submit_button("Save Feedback")
            if submitted:
                save_feedback(img_hash, correction)
                st.success("✅ Feedback saved! Future predictions updated.")

    else:
        st.info("Upload an image to begin analysis.")

    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Powered by ResNet18 • Forensic Signals • Self-Learning via Memory")
