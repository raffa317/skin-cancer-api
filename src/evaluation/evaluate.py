import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from src.dataset import SkinCancerDataset
from src.model import get_model
import os
from sklearn.metrics import classification_report
import numpy as np

def evaluate_model():
    # Setup
    DATA_DIR = "data"
    BATCH_SIZE = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Validation Transforms (No augmentation)
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load Dataset
    csv_file = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
    img_dir = os.path.join(DATA_DIR, "images")
    
    if not os.path.exists(csv_file) or not os.path.exists(img_dir):
        print("Data not found.")
        return

    full_dataset = SkinCancerDataset(
        csv_file=csv_file, 
        root_dir=img_dir, 
        phase2_dir=os.path.join(DATA_DIR, "pad_ufes"), 
        transform=transform_val
    )
    
    # Split (trying to replicate split logic, though random seed differs)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    # We use a fixed seed here to ensure consistent evaluation run-to-run for *this* script
    torch.manual_seed(42) 
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load Model (11 Classes)
    model = get_model(num_classes=11).to(device)
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth", map_location=device))
        print("Loaded trained model.")
    else:
        print("Model weights not found!")
        return
    
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Class Names (Updated for Phase 2)
    # Ensure this matches dataset.py exactly
    classes = [
        'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc', # 0-6
        'acne', 'eczema', 'normal', 'tinea'               # 7-10
    ]
    
    # 1. Full 11-Class Report
    print("\n" + "="*50)
    print("DETAILED CLASSIFICATION REPORT (11 Classes)")
    print("="*50)
    # Force labels=range(11) to match target_names
    print(classification_report(all_labels, all_preds, labels=range(len(classes)), target_names=classes, zero_division=0))
    
    # 2. Binary Report (Cancer vs Other)
    # Cancer indices: 0 (akiec), 1 (bcc), 4 (mel)
    # Other indices: 2 (bkl), 3 (df), 5 (nv), 6 (vasc)
    cancer_indices = [0, 1, 4]
    
    binary_preds = [0 if p in cancer_indices else 1 for p in all_preds]
    binary_labels = [0 if l in cancer_indices else 1 for l in all_labels]
    # 0 = Cancer, 1 = Other
    
    print("\n" + "="*50)
    print("BINARY CLASSIFICATION REPORT (Cancer vs Other)")
    print("="*50)
    print(classification_report(binary_labels, binary_preds, target_names=['cancer', 'other']))
    print("="*50)

if __name__ == "__main__":
    evaluate_model()
