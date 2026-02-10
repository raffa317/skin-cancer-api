import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from src.model import get_model
from src.gradcam import get_heatmap

def generate_explanation():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_path = "data/synthetic/mel/synthetic_big_0.png"
    model_path = "model.pth"
    output_path = "scorecam_heatmap.png"

    if not os.path.exists(img_path):
        print(f"Error: Image {img_path} not found.")
        return

    # 2. Load Model
    print("Loading model...")
    model = get_model(num_classes=11).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained weights.")
    else:
        print("Warning: model.pth not found, using random weights (heatmap might be nonsense).")
    model.eval()

    # 3. Load & Preprocess Image
    print("Preprocssing image...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    original_img = Image.open(img_path).convert('RGB')
    input_tensor = transform(original_img).unsqueeze(0).to(device) # (1, C, H, W)

    # 4. Generate Heatmap (Score-CAM)
    print("Generating Score-CAM heatmap...")
    # Note: Target class 4 is 'mel' in our class list: 
    # ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc', 'acne', 'eczema', 'normal', 'tinea']
    heatmap = get_heatmap(model, input_tensor, target_class=4, method="Score-CAM")
    
    # 5. Save Result
    # heatmap is an RGB numpy array (H, W, 3) in [0, 1] range? No, show_cam_on_image returns uint8 or float?
    # pytorch_grad_cam.utils.image.show_cam_on_image returns (H, W, 3) image in [0, 1] usually or [0, 255]
    # Let's check type. 
    # Actually cv2.imwrite expects [0, 255] BGR.
    # PIL expects RGB.
    
    # Let's convert to uint8 [0, 255]
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    Image.fromarray(heatmap_uint8).save(output_path)
    print(f"Saved heatmap to {output_path}")

if __name__ == "__main__":
    generate_explanation()
