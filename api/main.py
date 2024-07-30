from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2

app = FastAPI()

origins = [
    "http://localhost/",
    "http://localhost:3000/",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL= tf.keras.models.load_model('C:/Potato Project/Veg_mod_2.h5')

CLASS_NAMES = ['Corn(maize): Cercospora_leaf_spot Gray_leafspot',
              'Corn(maize): Commonrust',
              'Corn(maize): Northern_LeafBlight',
              'Corn(maize): healthy',
              'Pepper_bell: Bacterial_spot',
              'Pepper_bell: healthy',
              'Potato: Early_blight',
              'Potato: Late_blight',
              'Potato: healthy',
              'Tomato: Bacterial_spot',
              'Tomato: Early_blight',
              'Tomato: Late_blight',
              'Tomato: Leaf_Mold',
              'Tomato: Septoria_leaf_spot',
              'Tomato: Spider_mites Two-spotted_spider_mite',
              'Tomato: Target_Spot',
              'Tomato: Tomato_Yellow_Leaf_Curl_Virus',
              'Tomato: Tomato_mosaic_virus',
              'Tomato: healthy']

Cure = ['👉Rotate crops to reduce disease pressure. 👉Use resistant varieties if available. 👉Apply fungicides containing azoxystrobin or pyraclostrobin as directed.',
        '👉Plant resistant varieties if available. 👉Remove and destroy infected plant debris. 👉Apply fungicides containing triazoles or strobilurins as recommended.',
        '👉Plant resistant varieties if available. 👉Practice crop rotation to reduce disease carryover. 👉Apply fungicides containing triazoles or strobilurins according to instructions.',
        '👉Implement good cultural practices such as proper watering and fertilization. 👉Monitor plants regularly for any signs of disease or pests. 👉Take preventive measures such as crop rotation and sanitation.',
        '👉Plant resistant varieties if available. 👉Remove and destroy infected plant material. 👉Apply copper-based bactericides or antibiotics as directed.',
        '👉Maintain good cultural practices such as proper watering and adequate nutrition. 👉Monitor plants regularly for any signs of disease or pests. 👉Take preventive measures such as crop rotation and sanitation.',
        '👉Remove and destroy infected plant debris. 👉Apply fungicides containing chlorothalonil or copper-based fungicides as recommended. 👉Practice crop rotation and avoid overhead watering.',
        '👉Plant resistant varieties if available. 👉Remove and destroy infected plants immediately. 👉Apply fungicides containing chlorothalonil or copper-based fungicides as directed.',
        '👉Implement good cultural practices such as proper watering and fertilization. 👉Take preventive measures such as crop rotation and sanitation.',
        '👉Plant resistant varieties if available. 👉Remove and destroy infected plant material. 👉Apply copper-based bactericides or antibiotics as directed.',
        '👉Remove and destroy infected plant debris. 👉Apply fungicides containing chlorothalonil or copper-based fungicides as recommended. 👉Practice crop rotation and avoid overhead watering.',
        '👉Plant resistant varieties if available. 👉Remove and destroy infected plants immediately. 👉Apply fungicides containing chlorothalonil or copper-based fungicides as directed.',
        '👉Provide good air circulation and avoid overcrowding plants. 👉Water the plants at the base, keeping the foliage dry. 👉Apply fungicides containing chlorothalonil or other recommended fungicides.',
        '👉Remove and destroy infected leaves and plant debris. 👉Water the plants at the base to minimize leaf wetness. 👉Apply fungicides containing chlorothalonil or copper-based fungicides as directed.',
        '👉Spray affected plants with a strong stream of water to dislodge mites. 👉Use predatory mites or insecticidal soaps to control spider mites. 👉Avoid excessive use of broad-spectrum insecticides that can harm beneficial insects.',
        '👉Remove and destroy infected plant debris. 👉Provide adequate spacing between plants for better airflow. 👉Apply fungicides containing chlorothalonil or copper-based fungicides as recommended.',
        '👉Plant resistant varieties if available. 👉Remove and destroy infected plants to prevent the spread of the virus.',
        '👉There is no cure for Tomato mosaic virus. Focus on prevention and management strategies.',
        '👉Implement good cultural practices such as proper watering, fertilization, and pruning. 👉Monitor plants regularly for any signs of disease or pests. 👉Take preventive measures such as crop rotation, sanitation, and using disease-resistant varieties.']


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
    image = cv2.resize(image, dsize = (124,124))
    img_batch = np.expand_dims(image, 0)


    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)
    predicted_cure = Cure[np.argmax(predictions)]


    return {
        "class": predicted_class,
        "confidence": float(confidence),
        "predictedCure" : predicted_cure
    }

if __name__ == "__main__":
    # nest_asyncio.apply()
    uvicorn.run(app,host="localhost",port=3000)