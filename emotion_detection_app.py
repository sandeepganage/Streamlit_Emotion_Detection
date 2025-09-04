import io
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Face Emotion Classifier", page_icon="🙂", layout="centered")
st.title("🙂 Face Emotion Classifier — CPU-only, Streamlit-ready")

st.write("Use the camera or upload a photo. I’ll detect your face and predict an emotion (FER+ classes).")

# -------------------------------------------------------
# Config
# -------------------------------------------------------
FERPLUS_LABELS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt"
]

# Put your FER+ ONNX model in ./models/emotion-ferplus.onnx (recommended ~13MB).
# (Why local? Faster/safer than downloading at runtime on free tiers.)
MODEL_PATH = Path("models/emotion-ferplus-8.onnx")


# -------------------------------------------------------
# Utils
# -------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_onnx_session(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found at {model_path}. "
            f"Add it to your repo (models/emotion-ferplus.onnx)."
        )
    # Use CPU execution provider
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


@st.cache_resource(show_spinner=False)
def get_face_detector():
    # MediaPipe face detection (short-range model)
    return mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)


def read_image_to_rgb(image_file) -> np.ndarray:
    """Accepts a PIL Image or uploaded image file (bytes) and returns RGB np.array."""
    if isinstance(image_file, Image.Image):
        img = image_file
    else:
        img = Image.open(image_file)
    img = img.convert("RGB")
    return np.array(img)


def detect_faces_rgb(rgb: np.ndarray, detector) -> list:
    h, w, _ = rgb.shape
    results = detector.process(rgb)
    boxes = []
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x1 = max(int(bbox.xmin * w), 0)
            y1 = max(int(bbox.ymin * h), 0)
            x2 = min(int((bbox.xmin + bbox.width) * w), w - 1)
            y2 = min(int((bbox.ymin + bbox.height) * h), h - 1)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes


def preprocess_ferplus(face_rgb: np.ndarray) -> np.ndarray:
    """
    FER+ expects 64x64 grayscale, scaled to 0..255 or 0..1 depending on model.
    We'll produce shape (1, 1, 64, 64) float32 in range 0..255 for safety.
    """
    face_gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    face_resized = cv2.resize(face_gray, (64, 64), interpolation=cv2.INTER_AREA)
    x = face_resized.astype(np.float32)
    x = np.expand_dims(x, axis=(0, 1))  # (1, 1, 64, 64)
    return x


def softmax(logits: np.ndarray) -> np.ndarray:
    ex = np.exp(logits - np.max(logits))
    return ex / np.sum(ex)


def draw_boxes_and_labels(image_rgb: np.ndarray, boxes: list, labels: list, probs: list) -> np.ndarray:
    img = image_rgb.copy()
    for (x1, y1, x2, y2), label, p in zip(boxes, labels, probs):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 255), 2)
        text = f"{label} ({p:.2f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(y1 - 8, th + 4)
        cv2.rectangle(img, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 4), (0, 180, 255), -1)
        cv2.putText(img, text, (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return img


# -------------------------------------------------------
# Load models
# -------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Options")
    st.caption("Model: FER+ (ONNX) • Detector: MediaPipe")
    show_face_boxes = st.checkbox("Show face boxes", value=True)

try:
    ort_sess = load_onnx_session(MODEL_PATH)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

detector = get_face_detector()

# -------------------------------------------------------
# Inputs
# -------------------------------------------------------
st.subheader("1) Capture from camera or upload an image")

c1, c2 = st.columns(2)
with c1:
    captured = st.camera_input("Camera (click to take a photo)", key="cam")

with c2:
    uploaded = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])

# Select source image
img_rgb = None
src = None
if captured is not None:
    src = "camera"
    img_rgb = read_image_to_rgb(captured)
elif uploaded is not None:
    src = "upload"
    img_rgb = read_image_to_rgb(uploaded)

if img_rgb is None:
    st.info("No image yet. Use your webcam or upload a photo.")
    st.stop()

st.subheader("2) Detect face(s) and classify emotion")

# Face detection
boxes = detect_faces_rgb(img_rgb, detector)
if not boxes:
    st.warning("No face detected. Try a clearer, front-facing photo with good lighting.")
    st.image(img_rgb, caption="Input image", use_column_width=True)
    st.stop()

# For each face: crop -> preprocess -> ONNX -> softmax
input_name = ort_sess.get_inputs()[0].name
pred_labels, pred_probs = [], []
for (x1, y1, x2, y2) in boxes:
    face = img_rgb[y1:y2, x1:x2]
    x = preprocess_ferplus(face)
    outputs = ort_sess.run(None, {input_name: x})
    logits = outputs[0].reshape(-1)  # (8,)
    prob = softmax(logits)
    k = int(np.argmax(prob))
    pred_labels.append(FERPLUS_LABELS[k])
    pred_probs.append(float(prob[k]))

# Show results
if show_face_boxes:
    annotated = draw_boxes_and_labels(img_rgb, boxes, pred_labels, pred_probs)
    st.image(annotated, caption="Detections & emotions", use_column_width=True)
else:
    st.image(img_rgb, caption="Input image", use_column_width=True)
    for i, (b, lab, p) in enumerate(zip(boxes, pred_labels, pred_probs), start=1):
        st.write(f"Face {i}: **{lab}** (prob={p:.2f})")

# Nice summary
st.success("Done! Predictions computed on CPU with MediaPipe + ONNX Runtime.")
st.caption("Classes: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt.")
