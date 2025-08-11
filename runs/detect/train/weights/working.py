import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import imagehash

# ---------- CONFIGURATION ----------
MODEL_PATH = "runs/detect/train/weights/best.pt"  # change to your trained YOLO path
OUTPUT_DIR = "extracted_signatures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- LOAD YOLO MODEL ----------
model = YOLO(MODEL_PATH)

# ---------- FUNCTIONS ----------
def detect_and_crop_signatures(image_path, output_dir):
    results = model(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cropped_paths = []

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = cv2.imread(image_path)[y1:y2, x1:x2]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        output_path = os.path.join(output_dir, f"authorise_signature_{base_name}_{i+1}.png")
        cv2.imwrite(output_path, gray_crop)
        cropped_paths.append(output_path)

    return cropped_paths

def compare_images_ssim(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, (300, 150))
    img2 = cv2.resize(img2, (300, 150))
    score, _ = ssim(img1, img2, full=True)
    return score

def hash_difference(img1_path, img2_path):
    img1 = Image.open(img1_path).convert("L").resize((300, 150))
    img2 = Image.open(img2_path).convert("L").resize((300, 150))
    hash1 = imagehash.average_hash(img1)
    hash2 = imagehash.average_hash(img2)
    return (hash1 - hash2)

# ---------- STREAMLIT APP ----------
st.set_page_config(page_title="Signature Authentication", layout="centered")
st.title("🖋 Signature Authentication with YOLO")

uploaded_file = st.file_uploader("Upload Document Image", type=["jpg", "jpeg", "png"])
authorized_file = st.file_uploader("Upload Authorized Signature", type=["jpg", "jpeg", "png"])

if uploaded_file and authorized_file:
    with open("temp_doc.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    with open("auth_signature.png", "wb") as f:
        f.write(authorized_file.getbuffer())

    st.image("temp_doc.png", caption="Uploaded Document", use_column_width=True)
    st.image("auth_signature.png", caption="Authorized Signature", use_column_width=True)

    st.subheader("🔍 Detecting and Cropping Signatures...")
    cropped_paths = detect_and_crop_signatures("temp_doc.png", OUTPUT_DIR)

    if cropped_paths:
        for i, path in enumerate(cropped_paths):
            st.image(path, caption=f"Cropped Signature {i+1}")
            ssim_score = compare_images_ssim(path, "auth_signature.png")
            hash_diff = hash_difference(path, "auth_signature.png")
            
            st.write(f"**SSIM Similarity:** {ssim_score:.4f}")
            st.write(f"**Hash Difference:** {hash_diff}")
            
            if ssim_score > 0.8 and hash_diff < 5:
                st.success("✅ Signature matches the authorized signature.")
            else:
                st.error("❌ Signature does not match.")
    else:
        st.warning("No signatures detected.")
