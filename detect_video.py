import cv2
from PIL import Image
from transformers import pipeline

# 1. Hire the detective (same one as before)
detective = pipeline(
    "image-classification",
    model="dima806/deepfake_vs_real_image_detection",
)

# 2. Open the video
video = cv2.VideoCapture("test.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
step = max(int(fps), 1)  # look at one frame per second

fake_scores = []
frame_number = 0

# 3. Flip through the pages one by one
while len(fake_scores) < 30:
    ok, frame = video.read()
    if not ok:
        break  # the video ended

    if frame_number % step == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        picture = Image.fromarray(rgb)
        results = detective(picture)
        scores = {r["label"]: r["score"] for r in results}
        fake_scores.append(scores["Fake"])
        print("Second", len(fake_scores), "-> Fake:", round(scores["Fake"] * 100, 1), "%")

    frame_number += 1

video.release()

# 4. Average all the answers
average = sum(fake_scores) / len(fake_scores)
print("---")
print("Average fake score:", round(average * 100, 1), "%")
print("Verdict:", "FAKE" if average > 0.5 else "REAL")