import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("isic_model.h5")

model = load_model()

# Define class labels (update this based on your model's classes)
class_labels = ["Actinic keratosis", "Basal cell carcinoma", "Benign keratosis", "Dermatofibroma", "Melanocytic nevus", "Melanoma", "Squamous cell carcinoma", "Vascular lesion"]

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to model's expected input size
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Skin Disease Classification")
st.write("Upload an image and the model will predict the class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and make prediction
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    # Get predicted class and confidence
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")







