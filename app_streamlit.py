import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -------------------------------------------------------
# App title and description
# -------------------------------------------------------
st.title("🐱🐶 Cats vs Dogs Classifier")
st.write(
    "This simple web app uses a Convolutional Neural Network (CNN) trained on the "
    "[Cats vs Dogs dataset](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs) "
    "to classify uploaded images."
)

# -------------------------------------------------------
# Load the trained model (cached for performance)
# -------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "src/cat_dog_cnn_model.h5"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please make sure 'src/cat_dog_cnn_model.h5' exists.")
        st.stop()
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# -------------------------------------------------------
# Class names for cats_vs_dogs dataset
# -------------------------------------------------------
CLASS_NAMES = ["cat", "dog"]

# -------------------------------------------------------
# Image uploader
# -------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------------------------------------------------------
    # Preprocess image (must match training preprocessing)
    # -------------------------------------------------------
    IMG_SIZE = 128
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, 0)  # add batch dimension

    # -------------------------------------------------------
    # Predict
    # -------------------------------------------------------
    prediction = model.predict(img_array)[0][0]  # single sigmoid output
    predicted_label = 1 if prediction >= 0.5 else 0
    confidence = prediction if predicted_label == 1 else 1 - prediction

    # -------------------------------------------------------
    # Display result
    # -------------------------------------------------------
    st.markdown(f"### ✅ Predicted class: **{CLASS_NAMES[predicted_label]}**")
    st.write(f"📊 Confidence: {confidence*100:.2f}%")
