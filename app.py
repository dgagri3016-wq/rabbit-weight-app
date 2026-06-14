import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
import os
import requests

# --- CONFIGURATION ---
# Replace this URL with the direct download link from your GitHub Releases
MODEL_URL = "https://github.com/dgagri3016-wq/rabbit-weight-app/releases/download/keras_model/rabbit_weight_model.keras"
MODEL_PATH = "rabbit_weight_model.keras"
SCALER_PATH = "weight_scaler.pkl"

st.set_page_config(page_title="Rabbit Weight Predictor", page_icon="🐇")

st.title("🐇 Rabbit Weight Prediction App")
st.write("Upload an image or take a picture of a rabbit, and the deep learning model will predict its weight!")

# --- LOAD MODEL & SCALER ---
@st.cache_resource
def load_assets():
    # 1. Download the model from GitHub Releases if it doesn't exist locally
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from GitHub Releases (This happens once)..."):
            response = requests.get(MODEL_URL)
            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    f.write(response.content)
            else:
                st.error("Failed to download the model. Please check the MODEL_URL.")
                st.stop()
    
    # 2. Load the Keras Model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
        
    # 3. Load the Scikit-Learn Scaler
    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        st.stop()
        
    return model, scaler

model, scaler = load_assets()

# --- UI INTERFACE ---
# Create a radio button to select the input method
input_mode = st.radio("Select Image Input Method:", ("Upload Image", "Take Picture"))

image_data = None

# Show the appropriate input widget based on the radio button selection
if input_mode == "Upload Image":
    image_data = st.file_uploader("Choose a rabbit image...", type=["jpg", "jpeg", "png"])
elif input_mode == "Take Picture":
    image_data = st.camera_input("Take a picture of the rabbit")

# If the user has provided an image (either via upload or camera)
if image_data is not None:
    # Open the image so it's ready for prediction
    image = Image.open(image_data)
    
    # ONLY show the extra image if they uploaded a file
    # (Because the camera widget already shows a preview automatically)
    if input_mode == "Upload Image":
        st.image(image, caption="Uploaded Image", use_container_width=True)

  if st.button("Predict Weight", type="primary"):
        with st.spinner("Analyzing image..."):
            try:
                # --- PREPROCESSING ---
                # Open the image and force it into standard RGB format
                img = Image.open(image_data).convert('RGB')
                
                # Resize to (width=192, height=264)
                img = img.resize((192, 264))
                
                # Convert to numpy array and normalize to [0, 1]
                img_array = np.array(img) / 255.0
                
                # Add batch dimension -> (1, 264, 192, 3)
                img_array = np.expand_dims(img_array, axis=0)

                # --- PREDICTION ---
                # 1. Get the scaled prediction from the model
                pred_scaled = model.predict(img_array)
                
                # 2. Use the scaler to convert it back to actual weight (kg)
                pred_weight = scaler.inverse_transform(pred_scaled)
                
                # 3. Extract the exact float value from the nested array (e.g., [[2.5]]) -> 2.5
                final_weight = pred_weight[0][0]

                # 4. Display the result to the user!
                st.success(f"### Predicted Weight: {final_weight:.2f} kg") 
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
