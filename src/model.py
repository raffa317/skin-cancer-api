import torch.nn as nn
from torchvision import models

def get_model(num_classes=7, pretrained=True):
    # Using ResNet50
    # weights='DEFAULT' is equivalent to weights='IMAGENET1K_V2'
    model = models.resnet50(weights='DEFAULT' if pretrained else None)
    
    # Modify the final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model
