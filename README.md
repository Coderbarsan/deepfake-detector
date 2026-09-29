# Deepfake Detector

A simple app that checks whether an image or video is real or AI-generated.

## How it works
- Images: the face is found and cut out (with a wide border), then a pre-trained AI model gives a "Real" or "Fake" score.
- Videos: one frame per second is checked (up to 30 frames), and the median score is used.
- The interface is built with Streamlit.
- Videos also get a chart of the fake score for each second, so you can see when something looks fake.

## How to run
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `.\venv\Scripts\Activate.ps1`
3. Install the tools: `pip install -r requirements.txt`
4. Start the app: `streamlit run app.py`

## Accuracy test
I tested seven pre-trained models on 54 pictures: 27 real faces and 27 AI-generated faces. Score is the number of pictures judged correctly (whole picture, "fake" if the model is more than 50% sure).

| Model | Real faces right | Fake faces right | Overall |
|---|---|---|---|
| Wvolf/ViT_Deepfake_Detection (used in the app) | 20 / 27 | 17 / 27 | 37 / 54 (69%) |
| dima806/deepfake_vs_real_image_detection | 22 / 27 | 14 / 27 | 36 / 54 (67%) |
| date3k2/vit-real-fake-classification-v4 | 15 / 27 | 16 / 27 | 31 / 54 (57%) |
| prithivMLmods/deepfake-detector-model-v1 | 24 / 27 | 3 / 27 | 27 / 54 (50%) |
| Organika/sdxl-detector | 9 / 27 | 15 / 27 | 24 / 54 (44%) |
| prithivMLmods/Deep-Fake-Detector-v2-Model | 3 / 27 | 18 / 27 | 21 / 54 (39%) |
| umm-maybe/AI-image-detector | 20 / 27 | 1 / 27 | 21 / 54 (39%) |

What I learned:
- No model was good. The best ones catch only about half to two thirds of the AI-generated faces and wrongly flag about 1 in 4 real faces. The gap between the best two is one picture, which is within noise.
- Some models say "real" almost every time (they catch 1 to 3 fakes out of 27), and others say "fake" too often.
- Cropping the face did not improve accuracy on this test. A tight crop made a real video look fake, and a wider border fixed it.
- Raising the "fake" cutoff (50% to 99%) did not help. For the models tested it either missed more fakes or did not separate real from fake at all.
- One odd video frame can throw off an average, so the app uses the median.

## Limitations
- This is a probability, not proof. The model can be wrong.
- The test is small (54 pictures) and all the fake faces came from one website, so the numbers are rough.
- It can miss new kinds of fakes it wasn't trained on.
- Compressed or edited media can confuse it.
- It only looks at single frames, not at movement over time.