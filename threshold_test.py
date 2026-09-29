from pathlib import Path
from PIL import Image
from transformers import pipeline

MODELS = [
    "dima806/deepfake_vs_real_image_detection",
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "Organika/sdxl-detector",
]
FAKE_LABELS = {"fake", "deepfake", "artificial", "ai", "ai-generated", "generated"}
CUTOFFS = [0.5, 0.7, 0.9, 0.95, 0.99]

# 1. Load the exam pictures (whole pictures, no cropping)
exam = []
for folder, is_fake in [("real", False), ("fake", True)]:
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue
        exam.append((Image.open(path).convert("RGB"), is_fake))

real_total = sum(1 for _, is_fake in exam if not is_fake)
fake_total = sum(1 for _, is_fake in exam if is_fake)
print("Exam ready:", real_total, "real,", fake_total, "fake")

# 2. Interview each detective
for model_name in MODELS:
    print()
    print("=== Detective:", model_name)
    detective = pipeline("image-classification", model=model_name)

    def fake_score(picture):
        results = detective(picture)
        return sum(r["score"] for r in results if r["label"].lower() in FAKE_LABELS)

    # Ask about every picture ONCE and keep the answers
    answers = [(fake_score(picture), is_fake) for picture, is_fake in exam]

    # 3. Mark the same answers with different cutoffs
    print("Cutoff | real right | fake right | overall")
    for cutoff in CUTOFFS:
        real_right = sum(1 for s, is_fake in answers if not is_fake and s <= cutoff)
        fake_right = sum(1 for s, is_fake in answers if is_fake and s > cutoff)
        overall = real_right + fake_right
        print(
            f"{int(cutoff * 100):>5}% | {real_right:>3}/{real_total} | "
            f"{fake_right:>3}/{fake_total} | {overall}/{len(exam)}"
        )