from transformers import pipeline
from PIL import Image

# 1. Borrow a trained detective from the brain library
detective = pipeline(
    "image-classification",
    model="dima806/deepfake_vs_real_image_detection",
)

# 2. Open our picture
picture = Image.open("fake.jpg").convert("RGB")

# 3. Ask the detective what he thinks
results = detective(picture)

# 4. Print his answer
for r in results:
    print(r["label"], round(r["score"] * 100, 1), "%")