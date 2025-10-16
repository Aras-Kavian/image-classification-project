import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Constants
IMG_SIZE = 128

# Load model once (cached for performance)
@st.cache_resource
def load_cnn_model():
    return load_model("src/cat_dog_cnn_model.h5")

model = load_cnn_model()

st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image and the model will predict whether it's a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"

    st.markdown(f"### 🧠 Prediction: **{label}** (score={prediction:.4f})")