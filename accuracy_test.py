from pathlib import Path
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


def check_folder(folder, is_fake):
    whole_right = 0
    crop_right = 0
    total = 0
    mistakes = []

    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        picture = Image.open(path).convert("RGB")
        face, found = crop_face(picture)

        whole_says_fake = fake_score(picture) > 0.5
        crop_says_fake = fake_score(face) > 0.5

        total += 1
        whole_right += whole_says_fake == is_fake
        crop_right += crop_says_fake == is_fake

        if crop_says_fake != is_fake:
            mistakes.append(path.name)

    return total, whole_right, crop_right, mistakes


print("Marking the exam...")
for folder, is_fake in [("real", False), ("fake", True)]:
    total, whole_right, crop_right, mistakes = check_folder(folder, is_fake)
    print()
    print(f"Folder '{folder}': {total} pictures")
    print(f"  Whole picture: {whole_right} of {total} right")
    print(f"  Cropped face:  {crop_right} of {total} right")
    print(f"  Cropped mistakes: {mistakes}")