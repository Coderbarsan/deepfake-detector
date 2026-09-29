from pathlib import Path
from PIL import Image
from transformers import pipeline
from face_crop import crop_face

# The detectives we want to interview
MODELS = [
    "Wvolf/ViT_Deepfake_Detection",
    "prithivMLmods/deepfake-detector-model-v1",
    "umm-maybe/AI-image-detector",
    "date3k2/vit-real-fake-classification-v4",
]

# Words that mean "this one is fake"
FAKE_LABELS = {"fake", "deepfake", "artificial", "ai", "ai-generated", "generated"}

# 1. Load all the exam pictures once, and cut out the faces once
exam = []
for folder, is_fake in [("real", False), ("fake", True)]:
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue
        picture = Image.open(path).convert("RGB")
        face, found = crop_face(picture)
        exam.append((picture, face, is_fake))
print("Exam ready:", len(exam), "pictures")

# 2. Interview each detective
for model_name in MODELS:
    print()
    print("=== Detective:", model_name)
    try:
        detective = pipeline("image-classification", model=model_name)
    except Exception as error:
        print("Could not load this one:", str(error)[:150])
        continue

    labels = list(detective.model.config.id2label.values())
    print("His labels:", labels)
    if not any(label.lower() in FAKE_LABELS for label in labels):
        print("I can't tell which label means fake. Skipping him.")
        continue

    def fake_score(picture):
        results = detective(picture)
        return sum(r["score"] for r in results if r["label"].lower() in FAKE_LABELS)

    # 3. Mark the exam
    right = {"real": [0, 0, 0], "fake": [0, 0, 0]}  # [total, whole right, crop right]
    for picture, face, is_fake in exam:
        group = "fake" if is_fake else "real"
        right[group][0] += 1
        right[group][1] += (fake_score(picture) > 0.5) == is_fake
        right[group][2] += (fake_score(face) > 0.5) == is_fake

    for group in ["real", "fake"]:
        total, whole, crop = right[group]
        print(f"  {group}: whole {whole}/{total}, cropped {crop}/{total}")
    total = len(exam)
    print(f"  OVERALL: whole {right['real'][1] + right['fake'][1]}/{total}, "
          f"cropped {right['real'][2] + right['fake'][2]}/{total}")