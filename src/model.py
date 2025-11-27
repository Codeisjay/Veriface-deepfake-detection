import torch
import torch.nn as nn
import torchvision.models as models

def build_model(backbone="resnet18", num_classes=2, pretrained=False, dropout=0.2):
    if backbone == "resnet18":
        m = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
        in_features = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        return m
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
