from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ikramulh10.github.io/eye-disease/"],  # Replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_URL = "https://www.dropbox.com/scl/fi/r4exghzf0knsr3tc22k3r/model.pth?rlkey=jkfc535ez7kljthdor1yjw5u4&st=35cx4i3q&dl=1"
MODEL_PATH = "models/model.pth"

def download_file_direct(url, destination):
    if os.path.exists(destination):
        print("Model file already exists, skipping download.")
        size = os.path.getsize(destination)
        print(f"Existing model file size: {size} bytes")
        return
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    size = os.path.getsize(destination)
    print(f"Model downloaded successfully. File size: {size} bytes")

class EfficientNetB3Pretrained(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.model = models.efficientnet_b3(pretrained=pretrained)
        self.edd_head = nn.Sequential(nn.Linear(1000, 8))
    def forward(self, x):
        x = self.model(x)
        return self.edd_head(x)

def get_model(model_name, device, load_model_path=None):
    if model_name != "EfficientNetB3Pretrained":
        raise Exception(f"Model not found: {model_name}")
    model = EfficientNetB3Pretrained(pretrained=False).to(device)
    if load_model_path:
        model.load_state_dict(torch.load(load_model_path, map_location=device))
        print(f"Model loaded from path: {load_model_path}")
    model.eval()
    return model

# Download and load model once on startup
download_file_direct(MODEL_URL, MODEL_PATH)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = get_model("EfficientNetB3Pretrained", DEVICE, MODEL_PATH)

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

CLASSES = {
    0: "Normal",
    1: "Diabetes",
    2: "Glaucoma",
    3: "Cataract",
    4: "Age-related Macular Degeneration",
    5: "Hypertensive Retinopathy",
    6: "Myopia",
    7: "Unclassified case"
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image type")
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    input_tensor = IMG_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = MODEL(input_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy().flatten()
        pred_index = output.argmax(dim=1).item()
    response = {
        "predicted_class": CLASSES[pred_index],
        "probabilities": {CLASSES[i]: float(probabilities[i]) for i in range(len(probabilities))}
    }
    return response
