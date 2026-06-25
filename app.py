import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
import os
import requests

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Rabbit Weight Predictor", page_icon="🐇", layout="centered")

# --- CUSTOM CSS FOR BEAUTIFUL UI ---
st.markdown("""
<style>
    /* Hide Streamlit default headers and footers */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Card for the Prediction Result */
    .result-card {
        background-color: #f0fdf4;
        border-left: 6px solid #22c55e;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        margin-top: 15px;
        animation: fadeIn 0.5s ease-in-out;
    }
    .result-label {
        font-size: 14px;
        color: #15803d;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 5px;
        font-weight: 600;
    }
    .result-value {
        font-size: 42px;
        font-weight: 800;
        color: #166534;
        margin: 0;
    }
    
    /* Simple fade-in animation for the result */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# --- APP HEADER ---
st.markdown("<h1 style='text-align: center; color: #ff7f50;'>🐇 Rabbit Weight Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 18px; margin-bottom: 30px;'>AI-powered weight estimation using deep learning</p>", unsafe_allow_html=True)

# --- LOAD MODEL & SCALER ---
MODEL_URL = "https://github.com/dgagri3016-wq/rabbit-weight-app/releases/download/keras_model/rabbit_weight_model.keras"
MODEL_PATH = "rabbit_weight_model.keras"
SCALER_PATH = "weight_scaler.pkl"

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model (This happens once)..."):
            response = requests.get(MODEL_URL)
            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    f.write(response.content)
            else:
                st.error("Failed to download the model.")
                st.stop()
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        st.stop()

model, scaler = load_assets()

# --- INPUT SECTION ---
st.write("### 📸 Select Image Source")
# Using horizontal=True looks much cleaner
input_mode = st.radio("Input Method:", ("Upload Image", "Take Picture"), horizontal=True, label_visibility="collapsed")
image_data = None

if input_mode == "Upload Image":
    image_data = st.file_uploader("Drag and drop or click to select", type=["jpg", "jpeg", "png"])
elif input_mode == "Take Picture":
    image_data = st.camera_input("Take a picture of the rabbit")

st.divider()

# --- PREDICTION & DISPLAY SECTION ---
if image_data is not None:
    # Open the image
    image = Image.open(image_data)
    
    # Create two columns! Image on the left, Button & Results on the right
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.image(image, caption="Ready for Analysis", use_container_width=True)

    with col2:
        st.write("### 🔮 Analysis")
        st.write("Click below to let the AI estimate the weight of this rabbit.")
        
        # Make the button span the full width of its column
        if st.button("Predict Weight", type="primary", use_container_width=True):
            with st.spinner("Analyzing image features..."):
                try:
                    # --- Corrected Preprocessing ---
                    img = image.convert('RGB')
                    img = img.resize((192, 264))
                    
                    img_array = np.array(img, dtype="float32")
                    img_array = (img_array / 127.5) - 1.0  # Scale to [-1, 1]
                    img_array = np.expand_dims(img_array, axis=0)

                    # --- Prediction ---
                    pred_scaled = model.predict(img_array)
                    pred_weight = scaler.inverse_transform(pred_scaled)
                    final_weight = pred_weight[0][0]
                    
                    # --- Beautiful HTML Result Box ---
                    st.markdown(f"""
                        <div class="result-card">
                            <p class="result-label">Estimated Weight</p>
                            <p class="result-value">{final_weight:.2f} kg</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.balloons() # Adds a fun little celebration effect on success!
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    # Gentle prompt when no image is selected yet
    st.info("👆 Please upload an image or take a picture to begin.")
