import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# -------- Class Names ----------
CLASS_NAMES = [
    "No_DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative_DR"
]

# -------- Load Proper Model ----------
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, 5)

    state_dict = torch.load("dr_model1.pth", map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()
    return model

model = load_model()

# -------- Image Transform ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------- Image Validation + Prediction ----------
def is_retina_image(img: Image.Image):
    # Simple sanity check (you can improve later)
    w, h = img.size
    return w > 50 and h > 50

def predict(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except:
        return {
            "error": "Invalid image file"
        }

    # 📌 Validate Retina-like Image
    if not is_retina_image(img):
        return {
            "error": "Image is not a valid retina image"
        }

    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return {
        "predicted_index": int(pred.item()),
        "predicted_label": CLASS_NAMES[pred.item()],
        "confidence": round(float(conf.item() * 100), 2)
    }
