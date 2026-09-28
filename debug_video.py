import cv2
from PIL import Image
from transformers import pipeline
from face_crop import crop_face

detective = pipeline(
    "image-classification",
    model="dima806/deepfake_vs_real_image_detection",
)


def fake_score(picture):
    results = detective(picture)
    scores = {r["label"]: r["score"] for r in results}
    return scores["Fake"]


video = cv2.VideoCapture("test.mp4")
step = max(int(video.get(cv2.CAP_PROP_FPS)), 1)
frame_number = 0
saved = 0

while saved < 5:
    ok, frame = video.read()
    if not ok:
        break
    if frame_number % step == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        whole = Image.fromarray(rgb)
        face, found = crop_face(whole)
        face.save(f"debug_{saved}.jpg")
        print(
            f"Frame {saved}: face found={found}, crop size={face.size}, "
            f"whole={round(fake_score(whole) * 100, 1)}% fake, "
            f"cropped={round(fake_score(face) * 100, 1)}% fake"
        )
        saved += 1
    frame_number += 1

video.release()