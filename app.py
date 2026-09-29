import tempfile
import statistics
import pandas as pd
import cv2
import streamlit as st
from PIL import Image
from transformers import pipeline
from face_crop import crop_face

st.title("Deepfake Detector")


@st.cache_resource
def load_detective():
    return pipeline(
        "image-classification",
        model="Wvolf/ViT_Deepfake_Detection",
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
        face, found = crop_face(picture)
        st.image(face, width=300)
        if found:
            st.caption("Face found and cropped.")
        else:
            st.warning("No face found, so I checked the whole picture.")
        show_verdict(fake_score(face))

else:
    file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if file:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp.write(file.read())
        temp.close()

        with st.spinner("The detective is watching the video..."):
            video = cv2.VideoCapture(temp.name)
            step = max(int(video.get(cv2.CAP_PROP_FPS)), 1)
            scores = []
            faces_found = 0
            frame_number = 0

            while len(scores) < 30:
                ok, frame = video.read()
                if not ok:
                    break
                if frame_number % step == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face, found = crop_face(Image.fromarray(rgb))
                    faces_found += found
                    scores.append(fake_score(face))
                frame_number += 1
            video.release()

        if scores:
            st.caption(f"Face found in {faces_found} of {len(scores)} checked frames.")
            show_verdict(statistics.median(scores))
            chart_data = pd.DataFrame(
                {"Fake score (%)": [s * 100 for s in scores]},
                index=range(1, len(scores) + 1),
            )
            chart_data.index.name = "Second"
            st.subheader("Fake score for each second")
            st.line_chart(chart_data)
        else:
            st.warning("I couldn't read any frames from this video.")