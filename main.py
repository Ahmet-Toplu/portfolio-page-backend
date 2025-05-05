from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
# Ensure the model is downloaded
MODEL_PATH = Path("facenet_ec_0.7543.pth")
MODEL_URL = os.environ.get("MODEL_URL")

def download_model():
    if not MODEL_PATH.exists():
        print(f"Downloading model from {MODEL_URL}…")
        resp = requests.get(MODEL_URL, stream=True)
        resp.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(1024*1024):
                f.write(chunk)

# Call this before you load your model
download_model()

from facenet_classifier import load_model, predict_emotion

app = FastAPI()
# Load once, at startup
model = load_model("facenet_ec_0.7543.pth", device="cpu")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # only allow common image types
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(400, "Uploaded file is not an image")
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image")
    emotion = predict_emotion(img, model, device="cpu")
    return {"emotion": emotion}