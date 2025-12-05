from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from model import predict

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "DR Detection API Running"}

@app.post("/predict")
async def detect(file: UploadFile = File(...)):
    # 1. Read the raw bytes from the uploaded file
    # We do NOT save to disk; we pass the bytes directly to memory.
    image_bytes = await file.read()

    # 2. Pass bytes to the predict function
    # model.py is designed to handle raw bytes or base64 strings
    result = predict(image_bytes)

    return result