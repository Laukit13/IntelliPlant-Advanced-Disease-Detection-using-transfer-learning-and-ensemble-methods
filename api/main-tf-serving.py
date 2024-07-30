from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from main import MODEL

app = FastAPI()

MODEL= tf.keras.models.load_model('C:/Potato Project/saved_models/1')

endpoint = "http://localhost:3000/v1/models/potatoes_model:predict"

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "hello world"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file:UploadFile=File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)


    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    # nest_asyncio.apply()
    uvicorn.run(app,host="localhost",port=8000)