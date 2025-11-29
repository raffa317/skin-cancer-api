import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

def get_gradcam(model, image_tensor, target_class=None):
    """
    Generates Grad-CAM heatmap for a given image and model.
    
    Args:
        model: The PyTorch model.
        image_tensor: Preprocessed image tensor (1, C, H, W).
        target_class: Integer class index to visualize. If None, uses the highest scoring class.
        
    Returns:
        visualization: Numpy array of the image with heatmap overlay.
    """
    # Define target layer (last convolutional layer of ResNet50)
    # For ResNet, it is usually layer4[-1]
    target_layers = [model.layer4[-1]]
    
    # Construct GradCAM object
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Define targets
    if target_class is None:
        targets = None # Uses highest scoring class
    else:
        targets = [ClassifierOutputTarget(target_class)]
        
    # Generate CAM
    # image_tensor should be (1, C, H, W)
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
