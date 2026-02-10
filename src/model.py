import torch.nn as nn

def get_model(num_classes=7, pretrained=True, arch='mobilenet_v3'):
    """
    Get a pre-trained model for skin lesion classification
    
    Args:
        num_classes: Number of output classes
        pretrained: Use ImageNet pre-trained weights
        arch: Architecture - 'mobilenet_v3', 'resnet50', 'efficientnet', 'densenet'
    """
    if arch == 'mobilenet_v3':
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        if pretrained:
            model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        else:
            model = mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    
    elif arch == 'resnet50':
        from torchvision.models import resnet50, ResNet50_Weights
        if pretrained:
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif arch == 'efficientnet':
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        if pretrained:
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        else:
            model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    elif arch == 'densenet':
        from torchvision.models import densenet121, DenseNet121_Weights
        if pretrained:
            model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        else:
            model = densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    return model
