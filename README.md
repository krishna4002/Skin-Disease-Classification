## Skin Disease Classification Web App

This is a **Streamlit** web application that uses a trained deep learning model to classify various types of skin lesions from images. The model is trained to recognize the following categories:

- Actinic keratosis  
- Basal cell carcinoma  
- Benign keratosis  
- Dermatofibroma  
- Melanocytic nevus  
- Melanoma  
- Squamous cell carcinoma  
- Vascular lesion  

---

### 🧠 Model Details

- **Framework**: TensorFlow / Keras  
- **Input Size**: 224x224 pixels  
- **File**: `isic_model.h5` (trained model)  
- **Preprocessing**:
  - Resizing image to 224x224
  - Normalizing pixel values (dividing by 255)
  - Expanding dimensions to include batch size

---

### 🚀 Getting Started

#### Prerequisites

Make sure the following Python packages are installed:

```bash
pip install streamlit tensorflow pillow numpy
```

#### Running the App

```bash
streamlit run app.py
```

#### File Structure

```
project/
│
├── app.py                 # Streamlit web app
├── isic_model.h5          # Trained classification model
└── README.md              # You're reading it!
```

---

### 📸 How to Use

1. Run the app.
2. Upload a skin lesion image (`.jpg`, `.jpeg`, or `.png`).
3. The model will predict the class and show the confidence level.

---

### ⚠️ Notes

- Update the model path if needed in `app.py` under the `load_model()` function.
- Ensure that the uploaded image is clear and high-resolution for best results.
