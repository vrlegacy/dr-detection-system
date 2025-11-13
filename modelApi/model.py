import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# ✅ Define your model class (must match the one used during training)
# If you used EfficientNet, MobileNet, or ResNet — adjust accordingly.
class SimpleRetinalDiseaseClassifier(nn.Module):
    def __init__(self, base_model):
        super(SimpleRetinalDiseaseClassifier, self).__init__()
        self.base_model = base_model
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, 5)

    def forward(self, x):
        return self.base_model(x)

# ✅ Load pretrained model backbone
from torchvision import models
base_model = models.efficientnet_b0(pretrained=False)
model = SimpleRetinalDiseaseClassifier(base_model)

# ✅ Load your .pth file
state_dict = torch.load("colormodel_clean.pth", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

# ✅ Class names
CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]

# ✅ Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_index = torch.argmax(probabilities).item()
        confidence = probabilities[0][predicted_index].item()

    predicted_label = CLASS_NAMES[predicted_index]
    severity_score = predicted_index + 1  # 1–5 scale

    return {
        "predicted_index": predicted_index,
        "predicted_label": predicted_label,
        "severity_score": severity_score,
        "confidence": round(confidence * 100, 2)
    }
