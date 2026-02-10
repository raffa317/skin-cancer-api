import torch
import numpy as np
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

def get_heatmap(model, image_tensor, target_class=None, method="Grad-CAM"):
    """
    Generates heatmap (Grad-CAM or Score-CAM) for a given image and model.
    """
    # Define target layer for MobileNetV3
    # It is usually the last block in 'features'
    target_layers = [model.features[-1]]
    
    # Construct CAM object based on method
    if method == "Score-CAM":
        cam = ScoreCAM(model=model, target_layers=target_layers)
    else:
        cam = GradCAM(model=model, target_layers=target_layers)
    
    # Define targets
    if target_class is None:
        targets = None # Uses highest scoring class
    else:
        targets = [ClassifierOutputTarget(target_class)]
        
    # Generate CAM
    # image_tensor should be (1, C, H, W)
    # ScoreCAM is slower because it runs forward pass multiple times
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    
    # In this example grayscale_cam has shape (1, H, W)
    grayscale_cam = grayscale_cam[0, :]
    
    # Prepare image for visualization
    # Denormalize image tensor for visualization
    # Mean and std from training transform
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    
    return visualization

# Alias for backward compatibility if needed, but we will update app.py
get_gradcam = get_heatmap
