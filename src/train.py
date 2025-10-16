#!/usr/bin/env python3
"""
Train a simple CNN model on Cats vs Dogs dataset using TensorFlow Datasets
Usage:
python src/train.py
"""
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import os

# ----------------------------
# 1️⃣ Load dataset
# ----------------------------
(ds_train, ds_val), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# ----------------------------
# 2️⃣ Preprocessing
# ----------------------------
IMG_SIZE = 128
BATCH_SIZE = 32

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0  # normalize
    return image, label

ds_train = ds_train.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_val = ds_val.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ----------------------------
# 3️⃣ Build CNN model
# ----------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# ----------------------------
# 4️⃣ Train model
# ----------------------------
EPOCHS = 5
history = model.fit(ds_train,
                    validation_data=ds_val,
                    epochs=EPOCHS)

# ----------------------------
# 5️⃣ Save model
# ----------------------------
os.makedirs("src", exist_ok=True)
model.save("src/cat_dog_cnn_model.h5")
print("✅ Model saved -> src/cat_dog_cnn_model.h5")