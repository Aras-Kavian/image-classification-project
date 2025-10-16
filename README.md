# 🐶🐱 Image Classification with CNN (Cats vs Dogs)

A simple **Convolutional Neural Network (CNN)** built with TensorFlow to classify images of **cats and dogs**.  
This project demonstrates a complete workflow: data loading with TensorFlow Datasets, preprocessing, model training, evaluation, saving the model, and deployment with Streamlit.

---

## 🚀 Demo

👉 [Live Streamlit App](https://ai1900-image-classification.streamlit.app/)  
- (Upload an image of a cat or dog and get the prediction in real-time.)

---

## ✨ Features

- Uses **TensorFlow Datasets (TFDS)** for easy data access  
- Lightweight CNN architecture suitable for fast training  
- Clear project structure with training scripts, notebook, and deployment code  
- **Streamlit UI** for interactive image classification

---

## 🧱 Project Structure

image-classification-project/
- data/
- └── image_samples/ cat1.jpg ... dog3.jpg
- notebook/
- └── image_classification_training.ipynb
- src/
- ├── app.py
- ├── train.py
- ├── predict.py
- └── vectorizer.pkl
- app_streamlit.py
- requirements.txt
- README.md

---

## 🛠️ Installation & Usage

### 1️⃣ Clone the Repository

- git clone https://github.com/Aras-Kavian/image-classification-project.git
- cd image-classification-project

### 2️⃣ Install Dependencies

pip install -r requirements.txt

### 3️⃣ Train the Model (optional)

If you want to retrain the model from scratch:

- python src/train.py

- The trained model will be saved at:

- src/cat_dog_cnn_model.h5

### 4️⃣ Run the Streamlit App

streamlit run src/app.py

- Then open your browser at http://localhost:8501.

⸻

## 📓 Notebook

For a full step-by-step tutorial of dataset loading, model training, and evaluation, check out the notebook:
- 👉 notebooks/training_notebook.ipynb

⸻

## 🧠 Technologies Used
	•	TensorFlow
	•	TensorFlow Datasets
	•	Streamlit
	•	Python 3.10+

⸻

## 🌍 Author & Links
#### 👤 Aras Kavyani / AI 1900
- 🔗 [GitHub](#www.github.com/Aras-Kavian)
- 🔗 [LinkedIn](#www.linkedin.com/in/aras-kavyani)
- 🔗 [LaborX Profile](#www.laborx.com/customers/users/id409982?ref=409982)
- 🔗 [CryptoTask Profile](#www.cryptotask.org/en/freelancers/aras-kavyan/46480)
- 🔗 [Twitter](#www.x.com/ai_1900?s=21)
