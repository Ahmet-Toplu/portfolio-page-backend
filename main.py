from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

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