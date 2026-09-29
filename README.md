# Deepfake Detector

A simple app that checks whether an image or video is real or AI-generated.

## How it works
- Images: the face is found and cut out (with a wide border), then a pre-trained AI model gives a "Real" or "Fake" score.
- Videos: one frame per second is checked (up to 30 frames), and the median score is used.
- The interface is built with Streamlit.

## How to run
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `.\venv\Scripts\Activate.ps1`
3. Install the tools: `pip install -r requirements.txt`
4. Start the app: `streamlit run app.py`

## Accuracy test
I tested three pre-trained models on 54 pictures: 27 real faces and 27 AI-generated faces. Score is the number of pictures judged correctly.

| Model | Real faces right | Fake faces right | Overall |
|---|---|---|---|
| dima806/deepfake_vs_real_image_detection (used in the app) | 22 / 27 | 14 / 27 | 36 / 54 (67%) |
| prithivMLmods/Deep-Fake-Detector-v2-Model | 3 / 27 | 18 / 27 | 21 / 54 (39%) |
| Organika/sdxl-detector | 9 / 27 | 15 / 27 | 24 / 54 (44%) |

What I learned:
- The model I use is good at recognising real faces but catches only about half of the AI-generated ones, which is close to guessing.
- The other two models were over-suspicious and called most real faces fake.
- Cropping the face did not improve accuracy on this test. A tight crop made a real video look fake, and a wider border fixed it.
- Averaging video frames can be thrown off by one odd frame, so the app uses the median.

## Limitations
- This is a probability, not proof. The model can be wrong.
- The test is small (54 pictures) and all the fake faces came from one website, so the numbers are rough.
- It can miss new kinds of fakes it wasn't trained on.
- Compressed or edited media can confuse it.
- It only looks at single frames, not at movement over time.