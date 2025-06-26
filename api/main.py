from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import cv2
import os
import sys
sys.path.append(os.path.join(os.path.abspath('..')))

from utils.binary_filter import binary_leaf_check
from utils.opencv_filter import is_leaf

# Load disease model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "tomato_main.keras")
disease_model = tf.keras.models.load_model(MODEL_PATH)




# List of class names (you should adjust this to match your model)
CLASS_NAMES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

CLASS_NAMESp = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

app = FastAPI()

def read_image(file: UploadFile) -> np.ndarray:
    contents = file.file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    return img_array.astype(np.float32)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Step 1: OpenCV leaf check
        file.file.seek(0)
        image_bytes = np.asarray(bytearray(await file.read()), dtype=np.uint8)
        cv_img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if not is_leaf(cv_img):
            return JSONResponse(content={"message": "Image rejected: not a leaf (OpenCV filter)"})

        # Step 2: Reload file & preprocess for DL models
        file.file.seek(0)
        img_array = read_image(file)

        # Step 3: Binary leaf check
        if not binary_leaf_check(img_array):
            return JSONResponse(content={"message": "Image rejected: not a leaf (Binary filter)"})

        # Step 4: Disease prediction
        img_array = np.expand_dims(img_array, axis=0)
        prediction = disease_model.predict(img_array)
        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return JSONResponse(content={
            "class": CLASS_NAMES[class_index],
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
