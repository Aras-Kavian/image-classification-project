import streamlit as st
import tensorflow as tf
from PIL import Image
import os

# -------------------------------------------------------
# App title and description
# -------------------------------------------------------
st.set_page_config(page_title="Cats vs Dogs Classifier", page_icon="🐱🐶", layout="centered")
st.title("🐱🐶 Cats vs Dogs Classifier")
st.write(
    "This web app uses a Convolutional Neural Network (CNN) trained on the "
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
# Class names and emojis
# -------------------------------------------------------
CLASS_NAMES = ["cat", "dog"]
CLASS_EMOJI = {
    "cat": "🐱",
    "dog": "🐶"
}
CLASS_COLORS = {
    "cat": "#FF6F61",  # reddish for cat
    "dog": "#4CAF50"   # greenish for dog
}

# -------------------------------------------------------
# Image uploader
# -------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------------------------------------------------------
    # Preprocess image
    # -------------------------------------------------------
    IMG_SIZE = 128
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, 0)  # add batch dimension

    # -------------------------------------------------------
    # Predict
    # -------------------------------------------------------
    prediction = model.predict(img_array)[0][0]  # single sigmoid output
    cat_confidence = 1 - prediction
    dog_confidence = prediction
    predicted_label = 1 if prediction >= 0.5 else 0
    predicted_class_name = CLASS_NAMES[predicted_label]
    predicted_class_with_emoji = f"{predicted_class_name} {CLASS_EMOJI[predicted_class_name]}"

    # -------------------------------------------------------
    # Display predicted class
    # -------------------------------------------------------
    st.markdown(f"### ✅ Predicted class: **{predicted_class_with_emoji}**")

    # -------------------------------------------------------
    # Display confidence bars with colors and percentage
    # -------------------------------------------------------
    st.write("📊 Confidence Scores:")

    for class_name, confidence, color in zip(
        CLASS_NAMES, [cat_confidence, dog_confidence], [CLASS_COLORS["cat"], CLASS_COLORS["dog"]]
    ):
        percent = confidence * 100
        st.markdown(
            f"""
            <div style='display:flex; align-items:center; margin-bottom:10px;'>
                <div style='font-size:24px; width:50px;'>{CLASS_EMOJI[class_name]}</div>
                <div style='flex:1; background-color:#e0e0e0; border-radius:5px; margin-left:10px;'>
                    <div style='width:{percent}%; background-color:{color}; padding:5px 0; border-radius:5px; text-align:center; color:white; font-weight:bold;'>
                        {percent:.2f}%
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
