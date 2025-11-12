from fastapi import FastAPI, UploadFile, File
from model import predict
import uvicorn

app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description="Upload a retina image to predict the diabetic retinopathy stage.",
    version="1.0.0"
)

# Class labels in the same order as your model training
CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]

@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    file_path = "temp.jpg"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Get prediction index
    result_index = predict(file_path)

    # Map index to readable class name
    predicted_label = CLASS_NAMES[result_index] if result_index < len(CLASS_NAMES) else "Unknown"

    return {
        "predicted_index": int(result_index),
        "predicted_label": predicted_label
    }

# Optional: run directly using `python app.py`
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=9000, reload=True)

###
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
