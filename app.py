import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="Pothole Detection", layout="wide")
st.title("🚗 Real-Time Pothole Detection")

# Model selection
model_choice = st.selectbox(
    "Choose Model:",
    ["PotholeSegmentation.pt", "yolo11n.pt", "yolov8s.pt"]
)

@st.cache_resource
def load_model(name):
    return YOLO(name)

model = load_model(model_choice)
st.success(f"✅ Loaded: {model_choice}")

# Mode selection (remove webcam for cloud deployment)
mode = st.radio("Choose Mode:", ["Image", "Video File"])

if mode == "Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        img_array = np.array(img)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_array, caption="Original", use_container_width=True)
        
        with st.spinner("Detecting potholes..."):
            results = model.predict(img_array)[0]
            annotated = results.plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.image(annotated, caption="Detection", use_container_width=True)
        
        with st.expander("📊 Detection Details"):
            st.json(results.to_json())

elif mode == "Video File":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        st.video(video_path)
        
        if st.button("🔍 Process Video"):
            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()
            progress = st.progress(0)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model.predict(frame)[0]
                annotated = results.plot()
                stframe.image(annotated, channels="BGR", use_container_width=True)
                
                frame_count += 1
                progress.progress(frame_count / total_frames)
            
            cap.release()
            st.success("✅ Processing complete!")

st.markdown("---")
st.markdown("💡 **Note:** For webcam detection, run this app locally with `streamlit run app.py`")