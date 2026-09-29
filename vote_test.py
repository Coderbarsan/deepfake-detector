from itertools import combinations
from pathlib import Path
from PIL import Image
from transformers import pipeline

MODELS = {
    "wvolf": "Wvolf/ViT_Deepfake_Detection",
    "dima": "dima806/deepfake_vs_real_image_detection",
    "date3k2": "date3k2/vit-real-fake-classification-v4",
}

# 1. Load the exam pictures
pictures = []
for folder, is_fake in [("real", False), ("fake", True)]:
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            pictures.append((Image.open(path).convert("RGB"), is_fake))

# 2. Ask each detective about every picture and keep his notebook
scores = {}
for name, model_name in MODELS.items():
    detective = pipeline("image-classification", model=model_name)
    column = []
    for picture, _ in pictures:
        results = detective(picture)
        column.append({r["label"]: r["score"] for r in results}["Fake"])
    scores[name] = column
    print("finished", name)

answers = [is_fake for _, is_fake in pictures]
total = len(pictures)


def report(label, verdicts):
    right = sum(v == a for v, a in zip(verdicts, answers))
    print(f"{label}: {right}/{total}")


# 3. Each detective alone, then in teams
names = list(MODELS)
print()
for name in names:
    report(name + " alone", [s > 0.5 for s in scores[name]])
for a, b in combinations(names, 2):
    avg = [(x + y) / 2 for x, y in zip(scores[a], scores[b])]
    report(f"average of {a} + {b}", [s > 0.5 for s in avg])
avg3 = [sum(vals) / 3 for vals in zip(*scores.values())]
report("average of all three", [s > 0.5 for s in avg3])
votes = [sum(s > 0.5 for s in vals) >= 2 for vals in zip(*scores.values())]
report("majority vote of all three", votes)

# 4. What if we say "not sure" when they disagree?
agree_right = 0
agree_total = 0
for vals, answer in zip(zip(*scores.values()), answers):
    verdicts = [s > 0.5 for s in vals]
    if all(verdicts) or not any(verdicts):
        agree_total += 1
        agree_right += verdicts[0] == answer
print()
print(f"When all three agree: {agree_right} right out of {agree_total} pictures")
print(f"Left as 'not sure': {total - agree_total} pictures")