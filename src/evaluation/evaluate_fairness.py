import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from src.dataset import SkinCancerDataset
from src.model import get_model
from src.classify_skin_tone import calculate_ita, get_skin_category
import os
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from tqdm import tqdm

def evaluate_fairness():
    # Setup
    DATA_DIR = "data"
    BATCH_SIZE = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Validation Transforms (Consistent with training)
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load Dataset (Including Synthetic for completeness, though we evaluate on Real)
    csv_file = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
    img_dir = os.path.join(DATA_DIR, "images")
    
    full_dataset = SkinCancerDataset(
        csv_file=csv_file, 
        root_dir=img_dir, 
        phase2_dir=os.path.join(DATA_DIR, "pad_ufes"), 
        synthetic_dir=os.path.join(DATA_DIR, "synthetic"),
        transform=transform_val
    )
    
    # Split (same as train.py)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    torch.manual_seed(42) 
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Load Model
    model = get_model(num_classes=11).to(device)
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth", map_location=device))
        print("✅ Loaded trained model.")
    else:
        print("❌ Model weights not found!")
        return
    
    model.eval()
    
    # Grouped results
    results = {
        'Light': {'preds': [], 'labels': [], 'paths': []},
        'Dark': {'preds': [], 'labels': [], 'paths': []}
    }
    
    print("📋 Classifying validation samples by skin tone and running inference...")
    
    # Note: val_dataset is a Subset, we need to access items carefully
    for i in tqdm(range(len(val_dataset))):
        # Get raw path from the underlying dataset
        idx = val_dataset.indices[i]
        img_path, label = full_dataset.samples[idx]
        
        # 1. Classify Skin Tone
        ita = calculate_ita(img_path)
        category = get_skin_category(ita)
        
        if category not in results:
            continue
            
        # 2. Inference
        with torch.no_grad():
            # Apply transforms manually for single image
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            img_t = transform_val(img).unsqueeze(0).to(device)
            output = model(img_t)
            _, pred = torch.max(output, 1)
            
            results[category]['preds'].append(pred.item())
            results[category]['labels'].append(label)
            results[category]['paths'].append(img_path)

    # Report
    classes = full_dataset.classes
    cancer_indices = [0, 1, 4] # akiec, bcc, mel
    
    print("\n" + "═"*60)
    print("📈 FAIRNESS-AWARE PERFORMANCE AUDIT")
    print("═"*60)
    
    for group in ['Light', 'Dark']:
        preds = results[group]['preds']
        labels = results[group]['labels']
        
        if len(preds) == 0:
            print(f"\n⚠️ No samples found for {group} skin group.")
            continue
            
        print(f"\n📊 GROUP: {group} Skin (N={len(preds)})")
        acc = accuracy_score(labels, preds)
        print(f"Overall Accuracy: {acc*100:.2f}%")
        
        # Binary Cancer Sensitivity (Critical for research)
        bin_preds = [1 if p in cancer_indices else 0 for p in preds]
        bin_labels = [1 if l in cancer_indices else 0 for l in labels]
        
        print("\nCancer Detection (Binary):")
        print(classification_report(bin_labels, bin_preds, target_names=['Benign', 'Malignant'], zero_division=0))
        print("-" * 30)

if __name__ == "__main__":
    evaluate_fairness()
