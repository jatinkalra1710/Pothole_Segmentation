import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="Pothole Detection", layout="wide")

st.title("Real-Time Pothole Detection App")

# We now use the TorchScript export instead of the .pt checkpoint
MODEL_FILES = {
    "Custom pothole model": "PotholeSegmentation.torchscript",
    "YOLO11n (demo only)": "yolo11n.pt",      # optional if you have it
    "YOLOv8s (demo only)": "yolov8s.pt",      # optional if you have it
}

model_choice = st.selectbox("Choose Model:", list(MODEL_FILES.keys()))
weights_path = MODEL_FILES[model_choice]


@st.cache_resource
def load_model(path: str):
    """
    Load YOLO model once and cache it.
    For our custom model we use the TorchScript export, which avoids pickle.
    """
    return YOLO(path)


# Load selected model
model = load_model(weights_path)
st.success(f"Loaded model file: {weights_path}")

mode = st.radio("Choose Mode:", ["Image", "Video File", "Real-Time Webcam"])

# ====================== IMAGE MODE ======================
if mode == "Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        img_array = np.array(img)

        st.image(img_array, caption="Uploaded Image", use_container_width=True)

        results = model.predict(img_array)[0]
        annotated = results.plot()  # BGR
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        st.image(annotated_rgb, caption="Detection Output", use_container_width=True)

        with st.expander("Detection JSON"):
            st.json(results.tojson())

# ====================== VIDEO FILE MODE ======================
elif mode == "Video File":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(video_path)

        st.write("Processing video... (frames will appear below)")
        stframe = st.empty()

        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame)[0]
            annotated = results.plot()  # BGR
            stframe.image(annotated, channels="BGR", use_container_width=True)

        cap.release()
        st.success("Video processing complete!")

# ====================== REAL-TIME WEBCAM MODE ======================
elif mode == "Real-Time Webcam":
    st.markdown("Real-Time Webcam Pothole Detection")

    start = st.checkbox("Start Webcam")
    frame_window = st.image([])

    if start:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to access camera.")
                break

            results = model.predict(frame)[0]
            annotated = results.plot()  # BGR

            frame_window.image(annotated, channels="BGR", use_container_width=True)

            # Stop when checkbox is unchecked
            if not st.session_state.get("Start Webcam", True):
                break

        cap.release()
        st.success("Webcam stopped.")
