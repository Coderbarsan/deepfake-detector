import os
import statistics
import tempfile

import cv2
import streamlit as st
from PIL import Image
from transformers import pipeline

from face_crop import crop_face

st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️", layout="centered")

STYLE = """<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Figtree:wght@400;500;600&display=swap');
:root{--paper:#EEF2F6;--panel:#FFFFFF;--hero:#14213D;--ink:#14213D;--muted:#5B6B7F;--line:#C9D3DE;--dash:#94A8BE;--track:#D5DDE5;--cobalt:#2F5BEA;--real:#0F766E;--fake:#B42340;}
.stApp{background:var(--paper);color:var(--ink);}
header[data-testid="stHeader"]{background:transparent;}
.stApp,.stApp p,.stApp label,.stApp li,.stApp button{font-family:'Figtree',sans-serif;}
.block-container{max-width:980px;padding-top:2rem;padding-bottom:3rem;}
footer,[data-testid="stToolbar"]{display:none;}
[data-testid="stToggle"] p,[data-testid="stCheckbox"] p{color:var(--ink);}
[data-testid="stCaptionContainer"]{color:var(--muted);}
[data-testid="stSpinner"]{color:var(--ink);}
.stApp h3{color:var(--ink);}
.stButton{width:100%;}
.stButton button{width:100%;border-radius:999px;background:var(--panel);border:1px solid var(--line);padding:.45rem 1rem;}
.stButton button:hover{border-color:var(--cobalt);}
.stButton button p{color:var(--ink);font-weight:600;}
.hero{display:grid;grid-template-columns:1.3fr .7fr;gap:2rem;align-items:center;background:var(--hero);border-radius:24px;padding:2.8rem 2.6rem;}
.headline{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:3.3rem;line-height:1.02;letter-spacing:-.025em;color:#fff;margin:0 0 1rem;}
.hero p{color:#C5D0DE;font-size:1.1rem;line-height:1.55;max-width:34rem;margin:0;}
.scope{width:100%;max-width:230px;justify-self:center;}
.scope .frame{fill:none;stroke:#7EA0FF;stroke-width:3;stroke-linecap:round;}
.scope .head{fill:none;stroke:#3E5786;stroke-width:1.5;}
.scope .dot{fill:#9DB8FF;}
.scope .sweep{stroke:#9DB8FF;stroke-width:2;animation:scan 2.4s ease-in-out 1 both;}
@keyframes scan{from{transform:translateY(20px);opacity:0;}15%{opacity:1;}85%{opacity:1;}to{transform:translateY(180px);opacity:0;}}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:1.5rem;margin:1.8rem 0 1.4rem;}
.steps div{border-top:2px solid var(--ink);padding-top:.75rem;color:var(--muted);font-size:.95rem;line-height:1.45;}
.steps b{display:block;font-family:'Bricolage Grotesque',sans-serif;color:var(--ink);font-size:1.1rem;margin-bottom:.2rem;}
[role="tab"],[role="tab"] *{color:var(--muted) !important;font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.1rem;}
[role="tab"][aria-selected="true"],[role="tab"][aria-selected="true"] *{color:var(--ink) !important;}
[role="tab"]:focus:not(:focus-visible){outline:none;}
.uphead{display:flex;align-items:center;gap:.9rem;margin:.7rem 0 1rem;}
.uphead .ico{display:grid;place-items:center;width:54px;height:54px;border-radius:14px;background:var(--panel);border:1px solid var(--line);color:var(--cobalt);flex:none;}
.uphead svg{width:28px;height:28px;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;}
.uphead b{display:block;font-family:'Bricolage Grotesque',sans-serif;font-size:1.15rem;color:var(--ink);}
.uphead .sub{color:var(--muted);font-size:.9rem;}
[data-testid="stFileUploaderDropzone"]{background:var(--panel);border:1.5px dashed var(--dash);border-radius:16px;padding:2rem 1.5rem;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--cobalt);}
[data-testid="stFileUploaderDropzone"] span,[data-testid="stFileUploaderDropzone"] small,[data-testid="stFileUploaderDropzone"] div{color:var(--muted);}
[data-testid="stFileUploaderDropzone"] button{background:var(--paper);border:1px solid var(--line);border-radius:10px;}
[data-testid="stFileUploaderDropzone"] button *{color:var(--ink);}
[data-testid="stFileUploaderFile"],[data-testid="stFileUploaderFileName"]{color:var(--ink);}
[data-testid="stImage"] img{border-radius:12px;border:1px solid var(--line);}
.verdict{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1.4rem 1.6rem 1.5rem;text-align:center;}
.gauge{width:100%;max-width:300px;}
.gauge .track,.gauge .fill{fill:none;stroke-width:14;}
.gauge .track{stroke:var(--track);}
.gauge .fill{stroke-linecap:round;stroke-dasharray:var(--v) 100;animation:sweep .9s ease-out;}
.gauge .mid{stroke:var(--ink);stroke-width:2;}
.gauge .gnum{font:700 27px 'Bricolage Grotesque',sans-serif;fill:var(--ink);}
.gauge .gend{font:500 9px 'Figtree',sans-serif;fill:var(--muted);}
@keyframes sweep{from{stroke-dasharray:0 100;}}
.real .fill{stroke:var(--real);}
.fake .fill{stroke:var(--fake);}
.vtitle{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.9rem;margin:.1rem 0 .25rem;}
.real .vtitle{color:var(--real);}
.fake .vtitle{color:var(--fake);}
.verdict p{color:var(--muted);margin:0;font-size:.95rem;}
.chart{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:1rem 1rem .6rem;margin-top:1.2rem;}
.chart .ctitle{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.2rem;color:var(--ink);margin:0 0 .4rem;}
.chart svg{width:100%;height:auto;display:block;}
.chart .grid{stroke:var(--line);stroke-width:1;}
.chart .thr{stroke:var(--muted);stroke-width:1;stroke-dasharray:4 4;}
.chart .line{fill:none;stroke:var(--cobalt);stroke-width:2.5;stroke-linejoin:round;}
.chart .pt{fill:var(--cobalt);}
.chart .lbl{font:500 11px 'Figtree',sans-serif;fill:var(--muted);}
.page-end{margin-top:2.5rem;color:var(--muted);font-size:.9rem;text-align:center;}
.stApp a,.stApp a:visited{color:var(--cobalt) !important;text-decoration:underline;}
@media (max-width:720px){.hero{grid-template-columns:1fr;padding:2rem 1.5rem;}.headline{font-size:2.4rem;}.steps{grid-template-columns:1fr;}}
@media (prefers-reduced-motion:reduce){.scope .sweep,.gauge .fill{animation:none;}}
</style>"""

DARK = """<style>
:root{--paper:#0E1726;--panel:#16223A;--hero:#1B2B4A;--ink:#E6ECF5;--muted:#9AA9BE;--line:#2A3A55;--dash:#3B4E6D;--track:#2A3A55;--cobalt:#7EA0FF;--real:#2DD4BF;--fake:#FB7185;}
</style>"""

HERO = """<div class="hero">
<div>
<div class="headline" role="heading" aria-level="1">Is this face real?</div>
<p>Upload a photo or a short video. The detector checks the face and scores how likely it is to be AI-made.</p>
</div>
<svg class="scope" viewBox="0 0 200 200" aria-hidden="true">
<path class="frame" d="M22 52V22H52M148 22H178V52M178 148V178H148M52 178H22V148"/>
<ellipse class="head" cx="100" cy="100" rx="46" ry="60"/>
<circle class="dot" cx="82" cy="88" r="3.5"/>
<circle class="dot" cx="118" cy="88" r="3.5"/>
<circle class="dot" cx="100" cy="112" r="3"/>
<circle class="dot" cx="84" cy="134" r="3"/>
<circle class="dot" cx="100" cy="140" r="3"/>
<circle class="dot" cx="116" cy="134" r="3"/>
<circle class="dot" cx="58" cy="100" r="3"/>
<circle class="dot" cx="142" cy="100" r="3"/>
<line class="sweep" x1="22" y1="0" x2="178" y2="0"/>
</svg>
</div>
<div class="steps">
<div><b>1. Upload</b>A face photo, or a video (the first 30 seconds are checked).</div>
<div><b>2. Find the face</b>The face is cut out with some space around it.</div>
<div><b>3. Get a score</b>An AI model rates it. For video, you get the middle score of all frames.</div>
</div>"""

IMAGE_ICON = '<rect x="3" y="4" width="18" height="16" rx="3"/><circle cx="9" cy="10" r="1.6"/><path d="M4 18l5-5 4 4 3-3 4 4"/>'
VIDEO_ICON = '<rect x="3" y="5" width="18" height="14" rx="3"/><path d="M10 9.5v5l4.5-2.5z"/>'


def upload_head(icon, title, sub):
    return (
        '<div class="uphead">'
        f'<span class="ico"><svg viewBox="0 0 24 24" aria-hidden="true">{icon}</svg></span>'
        f'<div><b>{title}</b><span class="sub">{sub}</span></div>'
        '</div>'
    )


# ---------- Dark / light button ----------
if "dark" not in st.session_state:
    st.session_state.dark = False


def flip_theme():
    st.session_state.dark = not st.session_state.dark


_, switch_col = st.columns([5, 1.6])
with switch_col:
    st.button(
        "☀️ Light mode" if st.session_state.dark else "🌙 Dark mode",
        on_click=flip_theme,
        key="theme_button",
    )
dark = st.session_state.dark

st.markdown(STYLE, unsafe_allow_html=True)
if dark:
    st.markdown(DARK, unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the detector (the first run downloads about 340 MB)...")
def load_detective():
    return pipeline(
        "image-classification",
        model="Wvolf/ViT_Deepfake_Detection",
    )


detective = load_detective()


def fake_score(picture):
    results = detective(picture)
    scores = {r["label"]: r["score"] for r in results}
    return scores["Fake"]


def verdict_html(score, note):
    percent = round(score * 100, 1)
    is_fake = score > 0.5
    css = "fake" if is_fake else "real"
    title = "Likely fake" if is_fake else "Likely real"
    return (
        f'<div class="verdict {css}">'
        f'<svg class="gauge" viewBox="0 0 200 122" role="img" aria-label="Fake score {percent} percent">'
        '<path class="track" d="M 20 100 A 80 80 0 0 1 180 100" pathLength="100"/>'
        f'<path class="fill" d="M 20 100 A 80 80 0 0 1 180 100" pathLength="100" style="--v:{percent}"/>'
        '<line class="mid" x1="100" y1="10" x2="100" y2="30"/>'
        f'<text class="gnum" x="100" y="96" text-anchor="middle">{percent}%</text>'
        '<text class="gend" x="20" y="118" text-anchor="middle">0%</text>'
        '<text class="gend" x="180" y="118" text-anchor="middle">100%</text>'
        '</svg>'
        f'<div class="vtitle">{title}</div>'
        f'<p>{note}</p>'
        '</div>'
    )


def chart_html(scores):
    w, h = 600, 210
    left, right, top, bottom = 38, 12, 12, 28
    plot_w = w - left - right
    plot_h = h - top - bottom
    n = len(scores)

    def x(i):
        return left + (plot_w / 2 if n == 1 else plot_w * i / (n - 1))

    def y(s):
        return top + plot_h * (1 - s)

    points = " ".join(f"{x(i):.1f},{y(s):.1f}" for i, s in enumerate(scores))
    dots = "".join(
        f'<circle class="pt" cx="{x(i):.1f}" cy="{y(s):.1f}" r="3.5">'
        f'<title>Second {i + 1}: {s * 100:.1f}%</title></circle>'
        for i, s in enumerate(scores)
    )
    grid = "".join(
        f'<line class="{"thr" if v == 0.5 else "grid"}" x1="{left}" y1="{y(v):.1f}" x2="{w - right}" y2="{y(v):.1f}"/>'
        f'<text class="lbl" x="{left - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{int(v * 100)}%</text>'
        for v in (0, 0.5, 1)
    )
    return (
        '<div class="chart">'
        '<div class="ctitle">Fake score for each second</div>'
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Fake score for each second of the video">'
        f'{grid}'
        f'<polyline class="line" points="{points}"/>'
        f'{dots}'
        f'<text class="lbl" x="{left}" y="{h - 8}" text-anchor="middle">1s</text>'
        f'<text class="lbl" x="{w - right}" y="{h - 8}" text-anchor="middle">{n}s</text>'
        '</svg></div>'
    )


tab_photo, tab_video = st.tabs(["Photo", "Video"])

with tab_photo:
    st.markdown(
        upload_head(IMAGE_ICON, "Upload a photo", "JPG or PNG, with one clear face"),
        unsafe_allow_html=True,
    )
    image_file = st.file_uploader(
        "Upload a photo", type=["jpg", "jpeg", "png"], key="photo", label_visibility="collapsed"
    )
    if image_file:
        picture = Image.open(image_file).convert("RGB")
        face, found = crop_face(picture)
        left, right = st.columns([1, 1.1], gap="large")
        with left:
            st.image(face, width=320)
            if found:
                st.caption("Face found and cropped.")
            else:
                st.warning("No face found, so the whole picture was checked.")
        with right:
            st.markdown(
                verdict_html(fake_score(face), "The mark at the top is the 50% line. Above it, the detector calls a face fake."),
                unsafe_allow_html=True,
            )

with tab_video:
    st.markdown(
        upload_head(VIDEO_ICON, "Upload a video", "MP4, MOV or AVI. The first 30 seconds are checked."),
        unsafe_allow_html=True,
    )
    video_file = st.file_uploader(
        "Upload a video", type=["mp4", "mov", "avi"], key="video", label_visibility="collapsed"
    )
    if video_file:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp.write(video_file.read())
        temp.close()

        with st.spinner("The detective is watching the video..."):
            video = cv2.VideoCapture(temp.name)
            step = max(int(video.get(cv2.CAP_PROP_FPS)), 1)
            scores = []
            faces_found = 0
            frame_number = 0

            while len(scores) < 30:
                ok, frame = video.read()
                if not ok:
                    break
                if frame_number % step == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face, found = crop_face(Image.fromarray(rgb))
                    faces_found += found
                    scores.append(fake_score(face))
                frame_number += 1
            video.release()

        os.remove(temp.name)

        if scores:
            note = f"Middle score of {len(scores)} frames. A face was found in {faces_found} of them."
            st.markdown(verdict_html(statistics.median(scores), note), unsafe_allow_html=True)
            st.markdown(chart_html(scores), unsafe_allow_html=True)
        else:
            st.warning("No frames could be read from this video. Try an MP4 file.")

st.markdown(
    '<div class="page-end">Built by Barsan. '
    '<a href="https://github.com/Coderbarsan/deepfake-detector">Source code on GitHub</a></div>',
    unsafe_allow_html=True,
)