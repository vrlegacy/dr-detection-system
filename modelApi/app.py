from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from model import predict

app = FastAPI(title="Diabetic Retinopathy Detection API (PyTorch)")

# ✅ Enable frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def get_prediction(file: UploadFile = File(...)):
    file_path = "temp.jpg"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    result = predict(file_path)
    return result

@app.get("/")
def root():
    return {"message": "DR Detection API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=9000, reload=True)
