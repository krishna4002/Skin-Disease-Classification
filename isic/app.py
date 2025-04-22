import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.title("🩺 Skin Disease Classification")

# Upload model file
model_file = st.file_uploader("📦 Upload the trained model (.h5)", type=["h5"])

# Proceed if model is uploaded
if model_file is not None:
    # Save uploaded model file to disk
    with open("temp_model.h5", "wb") as f:
        f.write(model_file.read())

    try:
        with st.spinner("Loading model..."):
            model = tf.keras.models.load_model("temp_model.h5")
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

    class_labels = [
        "Actinic keratosis", "Basal cell carcinoma", "Benign keratosis",
        "Dermatofibroma", "Melanocytic nevus", "Melanoma",
        "Squamous cell carcinoma", "Vascular lesion"
    ]

    # Upload image for classification
    uploaded_file = st.file_uploader("📤 Upload a skin image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

        # Preprocess image
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Predict
        prediction = model.predict(image_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"✅ **Prediction:** {predicted_class}")
        st.info(f"📊 **Confidence:** {confidence:.2f}%")
