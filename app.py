import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np
from skimage.metrics import structural_similarity as ssim
import imagehash
from PIL import Image

# =====================
# Helper Functions
# =====================
def compare_images_ssim(imageA, imageB):
    # Convert to grayscale if needed
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Resize both images to the same dimensions
    height = min(grayA.shape[0], grayB.shape[0])
    width = min(grayA.shape[1], grayB.shape[1])
    grayA = cv2.resize(grayA, (width, height))
    grayB = cv2.resize(grayB, (width, height))

    # Compute SSIM
    score, _ = ssim(grayA, grayB, full=True)
    return score

def hash_difference(imageA, imageB):
    hashA = imagehash.average_hash(Image.fromarray(cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)))
    hashB = imagehash.average_hash(Image.fromarray(cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB)))
    return abs(hashA - hashB)

def crop_detected_signature(results, image):
    cropped_sigs = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_sig = image[y1:y2, x1:x2]
            cropped_sigs.append(cropped_sig)
    return cropped_sigs

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="YOLO Signature Authentication", layout="wide")
st.title("🖊 YOLO-based Signature Detection & Authentication")

# Load YOLO model
model_path = "runs/detect/train/weights/best.pt"  # Your trained YOLO model
model = YOLO(model_path)

# Upload files
uploaded_doc = st.file_uploader("Upload Document with Signature", type=["jpg", "jpeg", "png"])
uploaded_auth = st.file_uploader("Upload Authorised Signature", type=["jpg", "jpeg", "png"])

if uploaded_doc and uploaded_auth:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_doc:
        tmp_doc.write(uploaded_doc.read())
        tmp_doc_path = tmp_doc.name

    # Read authorised signature
    authorised_sig = np.array(Image.open(uploaded_auth).convert("RGB"))
    authorised_sig_cv = cv2.cvtColor(authorised_sig, cv2.COLOR_RGB2BGR)

    # Run YOLO detection
    results = model(tmp_doc_path)

    # Read document image
    doc_img = cv2.imread(tmp_doc_path)

    # Crop signatures from document
    cropped_sigs = crop_detected_signature(results, doc_img)

    if cropped_sigs:
        cropped_sig = cropped_sigs[0]  # Taking first detected signature

        # Compare SSIM & Hash
        ssim_score = compare_images_ssim(cropped_sig, authorised_sig_cv)
        hash_diff = hash_difference(cropped_sig, authorised_sig_cv)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("YOLO Detected Signature")
            annotated_img = results[0].plot()  # YOLO annotated output
            st.image(annotated_img, channels="BGR")
        with col2:
            st.subheader("Authorised Signature")
            st.image(authorised_sig, channels="RGB")
        with col3:
            st.subheader("Cropped Signature")
            st.image(cv2.cvtColor(cropped_sig, cv2.COLOR_BGR2RGB), channels="RGB")

        st.markdown("---")
        st.subheader("🔍 Authentication Results")
        st.write(f"**SSIM Similarity:** {ssim_score:.4f}")
        st.write(f"**Hash Difference:** {hash_diff}")

        if ssim_score > 0.80 and hash_diff < 20:
            st.success("✅ Signature Matched!")
        else:
            st.error("❌ Signature Mismatch!")
    else:
        st.error("No signature detected in the document.")

    os.remove(tmp_doc_path)
