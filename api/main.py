from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

# ------------------------------
# Load the latest model automatically
# ------------------------------
MODEL_DIR = "/Users/anamta/Desktop/potato diseases/models"

# Get all .keras models in the folder
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]

if not model_files:
    raise FileNotFoundError(f"No .keras models found in {MODEL_DIR}")

# Sort numerically (1.keras, 2.keras, etc.)
latest_model_file = sorted(model_files, key=lambda x: int(x.split(".")[0]))[-1]
MODEL_PATH = os.path.join(MODEL_DIR, latest_model_file)

print(f"Loading latest model: {MODEL_PATH}")
MODEL = tf.keras.models.load_model(MODEL_PATH)

# ------------------------------
# Constants
# ------------------------------
CLASS_NAMES = ["Early blight", "late blight", "healthy"]
IMG_SIZE = 256

# Preprocessing pipeline
resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE),
    tf.keras.layers.Rescaling(1. / 255)
])

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive!"}

def read_file_as_image(data) -> np.ndarray:
    """Process uploaded image to match training preprocessing exactly"""
    try:
        # Load image
        image = Image.open(BytesIO(data)).convert("RGB")
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        # Tensor conversion
        img_tensor = tf.convert_to_tensor(img_array)
        img_tensor = tf.expand_dims(img_tensor, 0)  # Add batch dimension
        processed_image = resize_and_rescale(img_tensor)
        return processed_image[0].numpy()
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict potato disease from uploaded image"""
    try:
        if not file.content_type.startswith('image/'):
            return {"error": "File must be an image"}

        contents = await file.read()
        image = read_file_as_image(contents)
        img_batch = np.expand_dims(image, 0)

        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'all_predictions': {
                CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))
            }
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)

