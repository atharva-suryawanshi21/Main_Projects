from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

CLASS_NAMES = ["early_blight", "late_blight", "healthy"]
MODEL = tf.keras.models.load_model("../saved_models/1")
app = FastAPI()


@app.get("/ping")
async def ping():
    return {"data": "hello there!!!!!!!!"}


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict_disease(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    # we need images in batches even if it is a single image
    image_batch = np.expand_dims(image, 0 )
    predictions = MODEL.predict(image_batch)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = np.max(predictions[0])

    return {
        'class':predicted_class,
        'confidence':float(confidence)
    }
