import tempfile
import cv2
import streamlit as st
from PIL import Image
from transformers import pipeline

st.title("Deepfake Detector")


@st.cache_resource
def load_detective():
    return pipeline(
        "image-classification",
        model="dima806/deepfake_vs_real_image_detection",
    )


detective = load_detective()


def fake_score(picture):
    results = detective(picture)
    scores = {r["label"]: r["score"] for r in results}
    return scores["Fake"]


def show_verdict(score):
    st.write("Fake score:", round(score * 100, 1), "%")
    if score > 0.5:
        st.error("Verdict: FAKE")
    else:
        st.success("Verdict: REAL")


choice = st.radio("What do you want to check?", ["Image", "Video"])

if choice == "Image":
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file:
        picture = Image.open(file).convert("RGB")
        st.image(picture, width=300)
        show_verdict(fake_score(picture))

else:
    file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if file:
        # OpenCV needs a real file on disk, so we save a temporary copy
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp.write(file.read())
        temp.close()

        with st.spinner("The detective is watching the video..."):
            video = cv2.VideoCapture(temp.name)
            step = max(int(video.get(cv2.CAP_PROP_FPS)), 1)
            scores = []
            frame_number = 0

            while len(scores) < 30:
                ok, frame = video.read()
                if not ok:
                    break
                if frame_number % step == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    scores.append(fake_score(Image.fromarray(rgb)))
                frame_number += 1
            video.release()

        if scores:
            show_verdict(sum(scores) / len(scores))
        else:
            st.warning("I couldn't read any frames from this video.")