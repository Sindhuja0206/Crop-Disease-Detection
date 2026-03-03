from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import base64
from io import BytesIO

# -------- CONFIG --------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
app = Flask(__name__)

# -------- DISEASE DATABASE --------
# This provides the "Descriptions" and "Treatments" for the UI
DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "title": "Pepper Bacterial Spot",
        "description": "Caused by Xanthomonas bacteria, this disease leads to leaf drop and fruit spots, significantly reducing yield.",
        "treatment": "Use copper-based fungicides. Avoid overhead irrigation and ensure seeds are pathogen-free."
    },
    "Pepper__bell___healthy": {
        "title": "Healthy Pepper Plant",
        "description": "The leaf appears vibrant and free of bacterial or fungal pathogens.",
        "treatment": "Maintain regular watering and organic fertilization to keep the immune system strong."
    },
    "Potato___Early_blight": {
        "title": "Potato Early Blight",
        "description": "A fungal disease (Alternaria solani) creating dark, 'target-like' spots with concentric rings on older leaves.",
        "treatment": "Apply fungicides containing chlorothalonil. Remove lower infected leaves and rotate crops annually."
    },
    "Potato___Late_blight": {
        "title": "Potato Late Blight",
        "description": "A devastating pathogen (Phytophthora infestans) that thrives in wet, cool conditions. Can kill plants in days.",
        "treatment": "Immediately remove and destroy infected plants. Use preventive copper sprays and improve air circulation."
    },
    "Potato___healthy": {
        "title": "Healthy Potato Plant",
        "description": "No signs of blight or nutrient deficiency detected.",
        "treatment": "Keep the soil hilled around the base and monitor for Colorado Potato Beetles."
    },
    "Tomato___Early_blight": {
        "title": "Tomato Early Blight",
        "description": "Common fungal infection causing brown spots. It usually starts at the bottom of the plant and moves up.",
        "treatment": "Prune the bottom 12 inches of leaves to prevent soil splash. Use mulch and organic fungicides."
    },
    "Tomato___Late_blight": {
        "title": "Tomato Late Blight",
        "description": "Appears as dark, water-soaked patches on leaves and stems. It spreads rapidly via wind and rain.",
        "treatment": "Remove infected tissue immediately. Do not compost infected plants. Apply a protectant fungicide."
    },
    "Tomato___healthy": {
        "title": "Healthy Tomato Plant",
        "description": "The foliage looks robust and shows high photosynthetic potential.",
        "treatment": "Ensure consistent watering to prevent blossom end rot as the fruit develops."
    }
}

class_names = list(DISEASE_INFO.keys())

# -------- BUILD MODEL --------
def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=None, 
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(8, activation="softmax")
    ])
    return model

# Load Weights
model = build_model()
try:
    model.load_weights("crop_weights.weights.h5")
    print("✅ Model weights loaded.")
except Exception as e:
    print(f"❌ Error loading weights: {e}")

# -------- PREPROCESS --------
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------- ROUTES --------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction_data = None
    error = None
    encoded_image = None

    if request.method == "POST":
        if "file" not in request.files:
            error = "No file part"
        else:
            file = request.files["file"]
            if file.filename == "":
                error = "No selected file"
            else:
                try:
                    # Process Image
                    img = Image.open(file).convert("RGB")
                    
                    # Convert image to base64 to show it back to the user
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    encoded_image = base64.b64encode(buffered.getvalue()).decode()

                    # Predict
                    processed = preprocess_image(img)
                    preds = model.predict(processed)
                    max_idx = np.argmax(preds)
                    conf = float(np.max(preds)) * 100
                    
                    # Get class info
                    class_key = class_names[max_idx]
                    prediction_data = {
                        "title": DISEASE_INFO[class_key]["title"],
                        "description": DISEASE_INFO[class_key]["description"],
                        "treatment": DISEASE_INFO[class_key]["treatment"],
                        "confidence": round(conf, 2),
                        "is_healthy": "healthy" in class_key.lower()
                    }

                except Exception as e:
                    error = f"Processing Error: {str(e)}"

    return render_template(
        "index.html", 
        prediction=prediction_data, 
        error=error,
        user_image=encoded_image
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
