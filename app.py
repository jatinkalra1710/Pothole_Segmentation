import tempfile
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from ultralytics import YOLO

st.set_page_config(page_title="Pothole Detection", layout="wide")
st.title("Pothole Detection App")

MODEL_PATH = "PotholeSegmentation.pt"

st.sidebar.header("Model")
st.sidebar.write(f"Using: `{MODEL_PATH}`")

if not Path(MODEL_PATH).exists():
    st.error(
        f"Model file not found: {MODEL_PATH}. "
        "Make sure the .pt file is in the same folder as app.py "
        "or update MODEL_PATH to the correct relative path."
    )
    st.stop()


@st.cache_resource
def load_model(path: str):
    return YOLO(path)


model = load_model(MODEL_PATH)
st.success("Model loaded successfully.")


def annotate_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    results = model.predict(frame_bgr, verbose=False)
    return results[0].plot()


mode = st.radio("Choose mode:", ["Image", "Video File", "Real-Time Webcam"], horizontal=True)

# ---------------- IMAGE MODE ----------------
if mode == "Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        img_rgb = np.array(img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        st.subheader("Original")
        st.image(img_rgb, use_container_width=True)

        annotated_bgr = annotate_bgr(img_bgr)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        st.subheader("Detection Output")
        st.image(annotated_rgb, use_container_width=True)

# ---------------- VIDEO FILE MODE ----------------
elif mode == "Video File":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tfile:
            tfile.write(uploaded_video.read())
            input_video_path = tfile.name

        st.subheader("Original Video")
        st.video(input_video_path)

        output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_video_path = output_tmp.name
        output_tmp.close()

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            st.error("Could not open the uploaded video.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            st.error("Could not create the output video writer.")
            cap.release()
            st.stop()

        st.subheader("Processing video")
        frame_placeholder = st.empty()
        progress = st.progress(0)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = annotate_bgr(frame)
            writer.write(annotated)

            frame_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

            frame_idx += 1
            if total_frames > 0:
                progress.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        writer.release()
        progress.progress(1.0)

        st.subheader("Annotated Video")
        st.video(output_video_path)
        st.success("Video processing complete.")

# ---------------- REAL-TIME WEBRTC MODE ----------------
elif mode == "Real-Time Webcam":
    st.subheader("Real-Time Webcam Detection")
    st.caption("Allow camera access in your browser when prompted.")

    class PotholeProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img_bgr = frame.to_ndarray(format="bgr24")
            annotated_bgr = annotate_bgr(img_bgr)
            return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

    webrtc_streamer(
        key="pothole-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PotholeProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )
