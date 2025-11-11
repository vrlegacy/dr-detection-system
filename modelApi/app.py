from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO: Replace this with your actual trained model file
model = tf.keras.models.load_model("model.h5")

def preprocess_image(image):
    img = image.resize((224, 224))      # adjust size to match your model
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    class_id = int(np.argmax(prediction))
    return {"result": class_id}
