import torch
import torch.nn as nn
import torchvision.models as models

# ---- RECREATE THE ORIGINAL MODEL CLASS ----
class SimpleRetinalDiseaseClassifier(nn.Module):
    def __init__(self):
        super(SimpleRetinalDiseaseClassifier, self).__init__()
        base = models.efficientnet_b3(weights="IMAGENET1K_V1")
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 5)
        )
        self.model = base

    def forward(self, x):
        return self.model(x)

# ---- LOAD THE ORIGINAL FULL MODEL ----
model = torch.load("colormodel.pth", map_location="cpu", weights_only=False)

# ---- CONVERT TO STATE_DICT ----
torch.save(model.state_dict(), "colormodel_clean.pth")

print("✅ Converted successfully → colormodel_clean.pth created")
