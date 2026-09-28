import cv2
import numpy as np
from PIL import Image

# 1. Load OpenCV's built-in face finder
face_finder = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def crop_face(picture):
    # 2. Make a black-and-white copy (the face finder works best with it)
    gray = cv2.cvtColor(np.array(picture), cv2.COLOR_RGB2GRAY)

    # 3. Look for faces
    faces = face_finder.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    # 4. No face? Give back the original picture
    if len(faces) == 0:
        return picture, False

    # 5. Pick the biggest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # 6. Add a little extra space around the face
    pad = int(0.2 * max(w, h))
    left = max(x - pad, 0)
    top = max(y - pad, 0)
    right = min(x + w + pad, picture.width)
    bottom = min(y + h + pad, picture.height)

    return picture.crop((left, top, right, bottom)), True