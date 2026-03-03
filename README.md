# 🌾 Crop Disease Detection

A web-based tool to **detect crop diseases from plant leaf images** using deep learning image classification.  
This project helps in **automating plant disease diagnosis**, which can reduce time, cost, and human effort in identifying diseases in agricultural crops.

Live demo:
https://huggingface.co/spaces/sindhujaaa/Crop_Disease_Detection

---

## 🧠 Overview

Crop Disease Detection uses a **machine learning model** trained on images of leaves to classify whether a crop is healthy or diseased. When a user uploads an image of a leaf, the model predicts the disease (if any) based on patterns in leaf texture, shape, and color.

---

## 🚀 Features

✅ Detects diseases from leaf images  
✅ Flask web app interface  
✅ Deep learning model for classification  
✅ Easy to train or update with new data

---

## 🗂️ Project Structure

Crop-Disease-Detection/

├── app.py # Flask web app

├── requirements.txt # Project dependencies

├── crop\_weights.weights.h5 # Pre-trained model weights

├── templates/ # HTML templates

│ └── index.html

├── Dockerfile # Docker support (optional)

└── README.md # This documentation


---

## 🛠️ How It Works

1. User uploads leaf image through web UI  
2. Flask backend receives the image  
3. The pre-trained deep learning model predicts disease class  
4. Result is shown on the front end

---

## 📥 Installation

Before starting, ensure **Python 3.8+** is installed.

1. **Clone the repository**
```bash
git clone https://github.com/Sindhuja0206/Crop-Disease-Detection.git
```
2.Navigate to the project folder
```
cd Crop-Disease-Detection
```
3.Install required dependencies
```
pip install -r requirements.txt
```
4.Run the Flask app
```
python app.py
```
5.Open your browser and go to:
```
http://localhost:5000
```
---
## 📁 Templates (example structure)

Inside the templates/ folder:

├── index.html # Form for uploading leaf image

├── result.html # Page to display prediction result

You can customize the UI here.

---
## 🧠 Model Details

The project uses a deep learning classifier that has been trained to recognize crop diseases from leaf images.
You can update or replace the crop_weights.weights.h5 file with a better model if needed.
