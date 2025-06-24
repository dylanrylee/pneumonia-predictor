import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from utils.gradcam import get_gradcam_overlay
from utils.preprocessing import preprocess_image

# Load model
model = tf.keras.models.load_model("model/pneumonia_model.h5")

# App title
st.title("🩺 Pneumonia Detection from Chest X-Rays")
st.markdown("Upload a chest X-ray and get a prediction with model confidence and Grad-CAM heatmap.")

# Upload image
uploaded_file = st.file_uploader("Choose a Chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Predict button
    if st.button("Predict"):
        # Preprocess and predict
        preprocessed = preprocess_image(image)
        prediction = model.predict(preprocessed)[0][0]
        label = "Pneumonia" if prediction > 0.5 else "Normal"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        # Display prediction
        st.subheader(f"🧠 Prediction: **{label}** ({confidence * 100:.2f}%)")

        # Try generating Grad-CAM
        try:
            gradcam_image = get_gradcam_overlay(model, preprocessed)

            # Show images side-by-side
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original X-ray", use_container_width=True)
            with col2:
                st.image(gradcam_image, caption="Grad-CAM Overlay", use_container_width=True)

        except ValueError as e:
            st.warning(f"⚠️ Grad-CAM could not be generated: {e}")
