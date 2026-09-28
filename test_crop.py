from PIL import Image
from face_crop import crop_face

picture = Image.open("test.jpg").convert("RGB")
face, found = crop_face(picture)

print("Face found:", found)
face.save("cropped.jpg")