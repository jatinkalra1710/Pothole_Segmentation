import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import time

# --- FIX FOR TORCH SAFE LOADING OF OLD YOLO CHECKPOINTS ---
import torch.serialization as ts
from ultralytics.nn.tasks import DetectionModel

# Allow torch to unpickle Ultralytics DetectionModel stored in .pt checkpoint
# (safe to do because the checkpoint comes from your own training)
ts.add_safe_globals([DetectionModel])
# -----------------------------------------------------------

st.set_page_config(page_title="Pothole Detection", layout="wide")

st.title("Real-Time Pothole Detection App")
model_choice = st.selectbox(
    "Choose Model:",
    ["PotholeSegmentation.pt", "yolo11n.pt", "yolov8s.pt"]
)

@st.cache_resource
def load_model(name: str):
    return YOLO(name)

model = load_model(model_choice)

st.success(f"Loaded model: {model_choice}")
mode = st.radio("Choose Mode:", ["Image", "Video File", "Real-Time Webcam"])

if mode == "Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded)
        img_array = np.array(img)

        st.image(img_array, caption="Uploaded Image")

        results = model.predict(img_array)[0]
        annotated = results.plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        st.image(annotated, caption="Detection Output")

        with st.expander("Detection JSON"):
            st.json(results.to_json())

elif mode == "Video File":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(video_path)

        st.write("Processing video...")

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame)[0]
            annotated = results.plot()

            stframe.image(annotated, channels="BGR")

        cap.release()
        st.success("Video processing complete!")

elif mode == "Real-Time Webcam":
    st.markdown("Real-Time Webcam Pothole Detection")

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to access camera.")
                break

            results = model.predict(frame)[0]
            annotated = results.plot()

            FRAME_WINDOW.image(annotated, channels="BGR", use_container_width=True)

            # Keep reading the checkbox state
            run = st.session_state.get("Start Webcam", True)

        cap.release()
        st.success("Webcam stopped.")
