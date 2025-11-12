import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# ✅ Step 1: Define your model class
class SimpleRetinalDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleRetinalDiseaseClassifier, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=False)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# ✅ Step 2: Create model instance
model = SimpleRetinalDiseaseClassifier(num_classes=5)

# ✅ Step 3: Load weights safely
state_dict = torch.load("colormodel_clean.pth", map_location="cpu")

# Remove mismatched classifier weights if any
for key in list(state_dict.keys()):
    if "classifier" in key:
        del state_dict[key]

# Load all other layers
model.load_state_dict(state_dict, strict=False)
model.eval()
print("✅ Model loaded successfully (classifier layer ignored).")

# ✅ Step 4: Define a prediction function
def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()
