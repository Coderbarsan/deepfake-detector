# Deepfake Detector

A simple app that checks whether an image or video is real or AI-generated.

## How it works
- Images: a pre-trained AI model gives a "Real" or "Fake" score.
- Videos: one frame per second is checked (up to 30 frames), and the scores are averaged.
- The interface is built with Streamlit.

## How to run
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `.\venv\Scripts\Activate.ps1`
3. Install the tools: `pip install -r requirements.txt`
4. Start the app: `streamlit run app.py`

## Limitations
- This is a probability, not proof. The model can be wrong.
- It can miss new kinds of fakes it wasn't trained on.
- Compressed or edited media can confuse it.
- It only looks at single frames, not at movement over time.