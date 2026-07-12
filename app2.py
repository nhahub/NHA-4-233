import streamlit as st
from PIL import Image
import time
import numpy as np
import json
import base64
from io import BytesIO
from urllib.parse import quote
import streamlit.components.v1 as components
import joblib 
import sklearn.linear_model
from featureExtractor import FeatureExtractor
from keras.applications.inception_v3 import preprocess_input
from streamlit_cropper import st_cropper
import tensorflow as tf

# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="AI Pharaoh",
    page_icon="🏺",
    layout="wide"
)

# =========================
# Custom CSS
# =========================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.hero {
    padding: 30px;
    border-radius: 15px;
    background: linear-gradient(135deg,#1E1E1E,#2B2B2B);
    text-align:center;
    margin-bottom:20px;
}

.hero-title {
    font-size:48px;
    font-weight:bold;
    color:#D4AF37;
}

.hero-subtitle {
    font-size:20px;
    color:white;
}

.info-card {
    background-color:#1E1E1E;
    padding:20px;
    border-radius:12px;
    border:1px solid #D4AF37;
}

.result-card {
    background-color:#20252B;
    padding:20px;
    border-radius:12px;
    border-left:6px solid #D4AF37;
}

.small-title {
    color:#D4AF37;
    font-weight:bold;
    font-size:18px;
}

</style>
""", unsafe_allow_html=True)


# =========================
# Load Model
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("artifact_classifier.keras")

model = load_model()


# =========================
# Load Hieroglyph Model
# =========================

@st.cache_resource
def load_hieroglyph_pipeline():
    # required so joblib can unpickle the older sklearn logistic module path
    import sys 
    sys.modules['sklearn.linear_model.logistic'] = sklearn.linear_model._logistic

    extractor = FeatureExtractor()
    clf = joblib.load("real_svm_tuned.pkl")
    classes = clf.best_estimator_.classes_ if hasattr(clf, "best_estimator_") else clf.classes_
    return extractor, clf, classes

hiero_extractor, hiero_clf, hiero_classes = load_hieroglyph_pipeline()


@st.cache_data
def load_gardiner_dictionary():
    with open("gardiner_signs.json", "r", encoding="utf-8") as f:
        return json.load(f)

gardiner_dict = load_gardiner_dictionary()


def explain_gardiner_code(code: str) -> dict:
    """
    Given a Gardiner code (e.g. 'G43', 'D21', 'Aa11'), return the best
    available explanation:
      1. Specific sign meaning, if we have it in our reference dictionary.
      2. Otherwise, fall back to the general category meaning (e.g. 'G' -> Birds).
    """
    signs = gardiner_dict.get("signs", {})
    categories = gardiner_dict.get("categories", {})

    if code in signs:
        entry = signs[code]
        return {
            "level": "specific",
            "meaning": entry.get("description", "Unknown"),
            "phonetic": entry.get("phonetic", "-"),
            "notes": entry.get("notes", "")
        }

    # extract the leading letter(s) of the code, e.g. "G43" -> "G", "Aa11" -> "Aa"
    letters = "".join(ch for ch in code if ch.isalpha())
    category_name = categories.get(letters, None)

    if category_name:
        return {
            "level": "category",
            "meaning": f"{category_name} (specific meaning not yet in our reference list)",
            "phonetic": "-",
            "notes": f"Category {letters}: {category_name}"
        }

    return {
        "level": "unknown",
        "meaning": "Meaning not available",
        "phonetic": "-",
        "notes": ""
    }


def predict_hieroglyph(pil_image, top_k=5):
    # Matches the original training pipeline (imageLoader.py):
    # load_img(..., target_size=(299,299)) loads in RGB by default,
    # then img_to_array + preprocess_input. No grayscale conversion.
    img = pil_image.convert("RGB").resize((299, 299))
    img_array = preprocess_input(np.array(img).astype(np.float32))
    img_array = np.expand_dims(img_array, 0)

    features = hiero_extractor.get_features(img_array)
    prob = np.array(hiero_clf.predict_proba(features))[0]

    top_i = np.argsort(-prob)[:top_k]
    top_names = hiero_classes[top_i]
    top_scores = prob[top_i]

    return list(zip(top_names, top_scores))


# =========================
# Load Labels
# =========================

with open("labels.json", "r") as f:
    class_names = json.load(f)


# =========================
# Load Artifacts Info (Historical Data)
# =========================

@st.cache_data
def load_artifacts_info():
    with open("artifacts.json", "r", encoding="utf-8") as f:
        return json.load(f)

artifacts_info = load_artifacts_info()


def get_artifact_info(name: str):
    """
    Look up artifact info by name from the JSON file.
    Falls back to a case-insensitive match if the exact
    label doesn't match a key in the JSON exactly.
    """
    if name in artifacts_info:
        return artifacts_info[name]

    # fallback: case-insensitive match
    for key in artifacts_info:
        if key.lower() == name.lower():
            return artifacts_info[key]

    # nothing found
    return None


# =========================
# VR Simulation Builder
# =========================

def build_vr_html(pil_image, title: str, subtitle: str) -> str:
    """
    Builds a self-contained HTML/CSS/JS VR headset simulation:
    - Image split into two "lens" viewports (stereo split-screen look)
    - Slight parallax offset between the two eyes
    - Rotate Left / Rotate Right / Zoom / Reset controls run entirely
      in JS (no Streamlit rerun -> smooth, real-time feel)
    - Floating title/subtitle text hovering above the scene
    """

    buffered = BytesIO()
    pil_image.convert("RGB").save(buffered, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    img_data_url = f"data:image/jpeg;base64,{img_b64}"

    safe_title = title.replace('"', "'")
    safe_subtitle = subtitle.replace('"', "'")

    html = """
<div id="vr-stage">

  <div class="vr-vignette"></div>

  <div class="floating-text">
    <div class="floating-card">
      <div class="floating-title">🏺 __TITLE__</div>
      <div class="floating-subtitle">__SUBTITLE__</div>
    </div>
  </div>

  <div class="lens-row">
    <div class="lens">
      <div class="lens-inner" id="lensLeft"></div>
    </div>
    <div class="lens-bridge"></div>
    <div class="lens">
      <div class="lens-inner" id="lensRight"></div>
    </div>
  </div>

  <div class="vr-controls">
    <button onclick="vrRotate(-1)">⬅ Rotate Left</button>
    <button onclick="vrZoom(0.15)">🔍 Zoom In</button>
    <button onclick="vrZoom(-0.15)">🔎 Zoom Out</button>
    <button onclick="vrRotate(1)">➡ Rotate Right</button>
    <button onclick="vrReset()">🔄 Reset</button>
  </div>

</div>

<style>
#vr-stage {
    position: relative;
    width: 100%;
    height: 480px;
    background: radial-gradient(circle at center, #1b1b1b 0%, #000000 100%);
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid #D4AF37;
    font-family: -apple-system, Segoe UI, Roboto, sans-serif;
}

.lens-row {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 18px;
    padding: 40px 30px 90px 30px;
}

.lens {
    flex: 1;
    height: 100%;
    max-width: 46%;
    border-radius: 50% / 40%;
    overflow: hidden;
    position: relative;
    box-shadow: inset 0 0 70px rgba(0,0,0,0.85), 0 0 0 3px rgba(212,175,55,0.5);
}

.lens-inner {
    position: absolute;
    top: -15%;
    left: -15%;
    width: 130%;
    height: 130%;
    background-image: url("__IMG_URL__");
    background-repeat: no-repeat;
    background-size: cover;
    background-position: center;
    transform: translateX(0%) scale(1.1);
    transform-origin: center center;
    transition: transform 0.35s ease-out;
    filter: saturate(1.05) contrast(1.05);
}

.lens-bridge {
    width: 10px;
    height: 70%;
    background: #000;
    border-radius: 6px;
    box-shadow: 0 0 12px rgba(0,0,0,0.8);
}

.vr-vignette {
    position: absolute;
    inset: 0;
    pointer-events: none;
    background: radial-gradient(circle, rgba(0,0,0,0) 55%, rgba(0,0,0,0.75) 100%);
    z-index: 3;
}

.floating-text {
    position: absolute;
    top: 22px;
    left: 0;
    right: 0;
    display: flex;
    justify-content: center;
    z-index: 4;
    perspective: 600px;
}

.floating-card {
    background: rgba(20,20,20,0.55);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(212,175,55,0.6);
    border-radius: 14px;
    padding: 10px 24px;
    text-align: center;
    animation: floatUpDown 3.2s ease-in-out infinite;
    box-shadow: 0 0 25px rgba(212,175,55,0.25);
}

.floating-title {
    color: #D4AF37;
    font-weight: bold;
    font-size: 20px;
}

.floating-subtitle {
    color: #eee;
    font-size: 13px;
    max-width: 520px;
    margin-top: 4px;
}

@keyframes floatUpDown {
    0%   { transform: translateY(0px); }
    50%  { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}

.vr-controls {
    position: absolute;
    bottom: 16px;
    left: 0;
    right: 0;
    display: flex;
    justify-content: center;
    gap: 10px;
    z-index: 5;
    flex-wrap: wrap;
}

.vr-controls button {
    background: #20252B;
    color: #D4AF37;
    border: 1px solid #D4AF37;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 13px;
    cursor: pointer;
    transition: background 0.2s ease;
}

.vr-controls button:hover {
    background: #D4AF37;
    color: #20252B;
}
</style>

<script>
(function() {
    let pan = 0;      // horizontal look direction, in % of lens width
    let zoom = 1.1;   // scale factor, independent of pan

    const lensLeft = document.getElementById("lensLeft");
    const lensRight = document.getElementById("lensRight");

    function render() {
        const leftPan = pan;
        const rightPan = pan + 2; // slight parallax between the two eyes
        lensLeft.style.transform = `translateX(${leftPan}%) scale(${zoom})`;
        lensRight.style.transform = `translateX(${rightPan}%) scale(${zoom})`;
    }

    window.vrRotate = function(direction) {
        pan = Math.max(-30, Math.min(30, pan + direction * 8));
        render();
    };

    window.vrZoom = function(delta) {
        zoom = Math.max(1.0, Math.min(2.2, zoom + delta));
        render();
    };

    window.vrReset = function() {
        pan = 0;
        zoom = 1.1;
        render();
    };

    // subtle interactive look-around on mouse movement
    const stage = document.getElementById("vr-stage");
    stage.addEventListener("mousemove", function(e) {
        const rect = stage.getBoundingClientRect();
        const ratio = (e.clientX - rect.left) / rect.width; // 0 -> 1
        pan = (ratio - 0.5) * 50;
        render();
    });

    render();
})();
</script>
"""

    html = (
        html.replace("__TITLE__", safe_title)
        .replace("__SUBTITLE__", safe_subtitle)
        .replace("__IMG_URL__", img_data_url)
    )

    return html


def build_audio_button_html(text: str) -> str:
    """
    Small self-contained button that uses the browser's built-in
    Text-to-Speech (SpeechSynthesis API) to read the artifact's
    historical overview aloud. No server-side audio generation needed.
    """
    safe_text = text.replace('"', "'").replace("\n", " ")

    html = """
<div id="audio-guide-wrap">
  <button id="audioBtn" onclick="toggleSpeak()">🔊 Audio Guide</button>
</div>

<style>
#audio-guide-wrap { display:flex; justify-content:center; }
#audioBtn {
    background:#20252B;
    color:#D4AF37;
    border:1px solid #D4AF37;
    border-radius:8px;
    padding:8px 16px;
    font-size:14px;
    cursor:pointer;
    width:100%;
}
#audioBtn:hover { background:#D4AF37; color:#20252B; }
</style>

<script>
const speechText = "__TEXT__";
let utterance = null;

function toggleSpeak() {
    const btn = document.getElementById("audioBtn");
    if (speechSynthesis.speaking) {
        speechSynthesis.cancel();
        btn.innerText = "🔊 Audio Guide";
        return;
    }
    utterance = new SpeechSynthesisUtterance(speechText);
    utterance.lang = "en-US";
    utterance.rate = 0.95;
    utterance.onend = function() { btn.innerText = "🔊 Audio Guide"; };
    speechSynthesis.speak(utterance);
    btn.innerText = "⏸ Stop Audio";
}
</script>
"""
    return html.replace("__TEXT__", safe_text)


# =========================
# Hero Section
# =========================

st.markdown("""
<div class="hero">

<div class="hero-title">
🏺 AI PHARAOH
</div>

<div class="hero-subtitle">
Discover Egypt Through Artificial Intelligence
</div>

<br>

Smart Tourist Guide • Monument Recognition • Hieroglyph Translation

</div>
""", unsafe_allow_html=True)

# =========================
# Tabs
# =========================

tab1, tab2 = st.tabs(
    [
        "🏛️ Monument Recognition",
        "📜 Hieroglyph Translator"
    ]
)

# ==================================================
# MONUMENT TAB
# ==================================================

with tab1:

    st.header("🏛️ Egyptian Monument Recognition")

    st.write(
        "Upload an image of an Egyptian monument or artifact."
    )

    uploaded_file = st.file_uploader(
        "Choose Image",
        type=["jpg", "jpeg", "png"],
        key="monument"
    )

    if uploaded_file:

        col1, col2 = st.columns([1,1])

        with col1:

            image = Image.open(uploaded_file)
            # تجهيز الصورة للموديل

            img = image.resize((224,224))

            img = np.array(img).astype(np.float32)

            img = tf.keras.applications.efficientnet.preprocess_input(img)

            img = np.expand_dims(img, axis=0)

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with col2:

            st.subheader("🔍 AI Analysis")

            progress = st.progress(0)

            status = st.empty()

            steps = [
                "Detecting Features...",
                "Matching Artifact...",
                "Loading Historical Data...",
                "Finalizing Results..."
            ]

            for i, step in enumerate(steps):

                status.info(step)

                progress.progress((i+1)*25)

                time.sleep(0.5)

            prediction = model.predict(img, verbose=0)

            predicted_index = np.argmax(prediction)

            artifact_name = class_names[predicted_index]

            confidence = np.max(prediction) * 100

            status.success("Analysis Completed")

            st.markdown(
                f"""
                <div class="result-card">

                <h2>🏺 {artifact_name}</h2>

                <h3>Confidence: {confidence:.1f}%</h3>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # =====================================
        # Pull historical data from JSON
        # =====================================

        info = get_artifact_info(artifact_name)

        if info:

            st.markdown(
                f"""
                <div class="info-card">

                <div class="small-title">
                📖 Historical Overview
                </div>

                {info.get("Historical overview", "N/A")}

                <br><br>

                <div class="small-title">
                👑 Dynasty
                </div>

                {info.get("Dynasty", "N/A")}

                <br><br>

                <div class="small-title">
                📍 Location
                </div>

                {info.get("Location", "N/A")}

                <br><br>

                <div class="small-title">
                📅 Period
                </div>

                {info.get("Period", "N/A")}

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.warning(
                f"No historical data found for '{artifact_name}' in egyptian_artifacts.json"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        btn1, btn2, btn3 = st.columns(3)

        with btn1:
            audio_text = info.get("Historical overview", artifact_name) if info else artifact_name
            components.html(build_audio_button_html(audio_text), height=45)

        with btn2:
            map_query = quote(f'{artifact_name} {info.get("Location", "") if info else ""}')
            st.link_button(
                "📍 Show on Map",
                f"https://www.google.com/maps/search/{map_query}",
                use_container_width=True
            )

        with btn3:
            show_vr = st.button("🥽 Explore in VR")

        # ==================================
        # VR Simulation
        # ==================================

        if show_vr:

            st.divider()

            st.subheader("🥽 VR Experience")

            st.caption(
                "Move your mouse over the scene or use the controls to look around."
            )

            vr_subtitle = ""
            if info:
                overview = info.get("Historical overview", "")
                vr_subtitle = (
                    overview[:140] + "..." if len(overview) > 140 else overview
                )

            vr_html = build_vr_html(
                pil_image=image,
                title=artifact_name,
                subtitle=vr_subtitle
            )

            components.html(vr_html, height=500, scrolling=False)

# ==================================================
# HIEROGLYPH TAB
# ==================================================


with tab2:

    st.header("📜 Hieroglyph Translator")

    uploaded_symbol = st.file_uploader(
        "Upload Hieroglyph Image",
        type=["jpg","jpeg","png"],
        key="symbol"
    )

    if uploaded_symbol:

        raw_image = Image.open(uploaded_symbol).convert("RGB")

        st.write(
            "🖱️ Drag the box below over **one single symbol** for the most accurate result."
        )

        col_crop, col_preview = st.columns([2, 1])

        with col_crop:
            hiero_image = st_cropper(
                raw_image,
                realtime_update=True,
                box_color="#D4AF37",
                aspect_ratio=None,
                return_type="image",
                key="hiero_cropper"
            )

        with col_preview:
            st.caption("Selected symbol:")
            st.image(hiero_image, width=180)

        with st.spinner("Extracting deep features and classifying..."):
            results = predict_hieroglyph(hiero_image, top_k=5)

        st.success(f"𓂀 Top match: {results[0][0]}  ({results[0][1]*100:.1f}%)")

        top_info = explain_gardiner_code(results[0][0])
        st.markdown(f"**Meaning:** {top_info['meaning']}")
        if top_info["phonetic"] != "-":
            st.markdown(f"**Phonetic value:** {top_info['phonetic']}")

        st.markdown("### 🧠 Top 5 Predictions")

        medals = ["🥇", "🥈", "🥉", "🔹", "🔹"]

        for (code, score), medal in zip(results, medals):
            info = explain_gardiner_code(code)
            st.markdown(f"**{medal} {code}** — {info['meaning']} ({score*100:.1f}%)")
            st.progress(min(float(score), 1.0))