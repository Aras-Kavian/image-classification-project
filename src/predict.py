#!/usr/bin/env python3
"""
Predict image class (cat or dog) using trained CNN model
Usage:
python src/predict.py --image path/to/image.jpg
"""
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

IMG_SIZE = 128

def load_and_preprocess_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the image")
    args = parser.parse_args()

    model = load_model("src/cat_dog_cnn_model.h5")
    img = load_and_preprocess_image(args.image)
    pred = model.predict(img)[0][0]

    label = "Dog 🐶" if pred > 0.5 else "Cat 🐱"
    print(f"Prediction: {label} (score={pred:.4f})")