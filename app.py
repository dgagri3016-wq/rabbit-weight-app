import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
import os
import requests

# =========================
# CONFIGURATION
# =========================
MODEL_URL = "https://github.com/dgagri3016-wq/rabbit-weight-app/releases/download/keras_model/rabbit_weight_model.keras"
MODEL_PATH = "rabbit_weight_model.keras"
SCALER_PATH = "weight_scaler.pkl"

st.set_page_config(
    page_title="Rabbit Weight Predictor",
    page_icon="🐇",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #f4f6fb;
}

/* Main Container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Hero Section */
.hero-card {
    background: linear-gradient(135deg, #4F8DFD, #6BA8FF);
    color: white;
    padding: 30px;
    border-radius: 24px;
    margin-bottom: 25px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
}

.hero-title {
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 5px;
}

.hero-subtitle {
    font-size: 17px;
    opacity: 0.9;
}

/* Dashboard Cards */
.dashboard-card {
    background: white;
    padding: 25px;
    border-radius: 24px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

/* Result Card */
.result-card {
    background: white;
    padding: 35px;
    border-radius: 24px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.06);
    text-align: center;
}

.result-title {
    color: #6b7280;
    font-size: 18px;
    margin-bottom: 10px;
}

.result-value {
    font-size: 56px;
    font-weight: 700;
    color: #4F8DFD;
}

.result-status {
    color: #16a34a;
    font-weight: 600;
    margin-top: 10px;
}

/* Model Info */
.info-box {
    background: #f8fafc;
    border-radius: 16px;
    padding: 15px;
    margin-top: 20px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 14px;
    border: none;
    background-color: #4F8DFD;
    color: white;
    font-weight: 600;
    font-size: 16px;
}

.stButton > button:hover {
    background-color: #3b78f0;
}

/* Radio Buttons */
div[role="radiogroup"] {
    flex-direction: row;
    gap: 20px;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    border-radius: 16px;
}

/* Hide Streamlit Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🐇 Rabbit Weight Predictor</div>
    <div class="hero-subtitle">
        AI-powered rabbit weight estimation using deep learning
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL & SCALER
# =========================
@st.cache_resource
def load_assets():

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            response = requests.get(MODEL_URL)

            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    f.write(response.content)
            else:
                st.error("Failed to download the model.")
                st.stop()

    try:
        model = tf.keras.models.load_model(MODEL_PATH)

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        st.stop()

    return model, scaler


model, scaler = load_assets()

# =========================
# MAIN DASHBOARD
# =========================
left_col, right_col = st.columns([1, 1])

image_data = None
image = None

# =========================
# INPUT SECTION
# =========================
with left_col:

    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)

    st.subheader("📷 Rabbit Image")

    input_mode = st.radio(
        "Choose Input Method",
        ["Upload Image", "Take Picture"]
    )

    if input_mode == "Upload Image":
        image_data = st.file_uploader(
            "Upload a rabbit image",
            type=["jpg", "jpeg", "png"]
        )

    else:
        image_data = st.camera_input(
            "Take a picture"
        )

    if image_data is not None:

        image = Image.open(image_data)

        st.image(
            image,
            caption="Rabbit Image Preview",
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# RESULT SECTION
# =========================
with right_col:

    if image is None:

        st.markdown("""
        <div class="result-card">
            <div class="result-title">
                Predicted Weight
            </div>

            <div class="result-value">
                -- kg
            </div>

            Upload an image to begin prediction.
        </div>
        """, unsafe_allow_html=True)

    else:

        if st.button("🔍 Predict Weight"):

            with st.spinner("Analyzing image..."):

                try:

                    # =========================
                    # PREPROCESSING
                    # =========================
                    img = image.convert("RGB")
                    img = img.resize((192, 264))

                    img_array = np.array(
                        img,
                        dtype="float32"
                    )

                    img_array = (
                        img_array / 127.5
                    ) - 1.0

                    img_array = np.expand_dims(
                        img_array,
                        axis=0
                    )

                    # =========================
                    # PREDICTION
                    # =========================
                    pred_scaled = model.predict(
                        img_array,
                        verbose=0
                    )

                    pred_weight = scaler.inverse_transform(
                        pred_scaled
                    )

                    final_weight = pred_weight[0][0]

                    st.markdown(f"""
                    <div class="result-card">

                        <div class="result-title">
                            Predicted Weight
                        </div>

                        <div class="result-value">
                            {final_weight:.2f} kg
                        </div>

                        <div class="result-status">
                            ✓ Prediction Complete
                        </div>

                        <div class="info-box">
                            🤖 Model: Rabbit CNN<br>
                            📐 Input Size: 192 × 264<br>
                            ⚙️ Framework: TensorFlow / Keras
                        </div>

                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:

                    st.error(
                        f"Prediction failed: {e}"
                    )

# =========================
# FOOTER
# =========================
st.markdown("""
<br>

<div class="dashboard-card" style="text-align:center;">

<h4>About This App</h4>

This application estimates rabbit weight from images using a
deep learning regression model trained with TensorFlow/Keras.

Built with ❤️ using Streamlit.

</div>
""", unsafe_allow_html=True)
```
