# Plant Disease Detection — CNN + Transfer Learning

A deep learning system that detects plant diseases from leaf images using Convolutional Neural Networks and Transfer Learning. Trained on the PlantVillage dataset and deployed with Streamlit.

## Tech Stack

- **Language:** Python 3
- **Deep Learning:** TensorFlow / Keras
- **Model:** Transfer Learning (EfficientNet / ResNet pretrained on ImageNet)
- **Dataset:** PlantVillage (38 disease classes across 14 crop species)
- **Deployment:** Streamlit + Hugging Face Spaces

## Features

- Upload a leaf image → get instant disease prediction
- Supports 38 plant disease classes
- Real-time inference using a pretrained CNN
- Deployed as a public web app

## How It Works

1. **Dataset** — PlantVillage dataset with 54,000+ images across 38 classes
2. **Model** — pretrained EfficientNet fine-tuned on PlantVillage
3. **Transfer Learning** — reused ImageNet weights, replaced final classification layer
4. **Training** — fine-tuned for 10 epochs with data augmentation
5. **Deployment** — Streamlit UI wraps the model for real-time image upload and prediction

## 🚀 How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/plant-disease-detection.git
cd plant-disease-detection

pip install tensorflow streamlit pillow numpy

streamlit run app.py
```

## 🌐 Live Demo

[Try it on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/plant-disease) *(link added after deployment)*

## What I Learned

- What Transfer Learning is and why it works — reusing features learned on ImageNet
- How CNN layers progressively learn from edges → shapes → complex patterns
- Why data augmentation helps prevent overfitting on image datasets
- How to wrap a deep learning model in a Streamlit web app for real-time inference
Copy
