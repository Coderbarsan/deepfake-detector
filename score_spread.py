from pathlib import Path
from PIL import Image
from transformers import pipeline

detective = pipeline("image-classification", model="Wvolf/ViT_Deepfake_Detection")


def fake_score(picture):
    results = detective(picture)
    scores = {r["label"]: r["score"] for r in results}
    return scores["Fake"]


all_scores = {"real": [], "fake": []}
for folder in ["real", "fake"]:
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue
        all_scores[folder].append(fake_score(Image.open(path).convert("RGB")))

for folder in ["real", "fake"]:
    print()
    print(f"'{folder}' folder, fake score (%) of each picture, lowest to highest:")
    print(sorted(round(s * 100) for s in all_scores[folder]))