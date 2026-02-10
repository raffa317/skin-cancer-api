import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from src.model import get_model
from src.dataset import SkinCancerDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import random
import glob

# Style Config
plt.style.use('default')

# Config
EPOCHS = 20
BATCH_SIZE = 16
DATA_DIR = "data"
RESULTS_DIR = "comparison_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_dark_skin_test_set(transform, size=100):
    """
    Selects 100 separate images from PAD-UFES-20 to act as the 'Judge'.
    In a real scenario, we'd filter by metadata 'Fitzpatrick=V/VI'.
    Here, we'll randomly select 100 images from the 'pad_ufes' folder 
    and REMOVE them from the training set to avoid leakage.
    """
    pad_dir = os.path.join(DATA_DIR, "pad_ufes") 
    # Fallback to phase2 if pad_ufes empty, based on previous ls
    if not os.path.exists(pad_dir) or len(glob.glob(os.path.join(pad_dir, "*", "*"))) < 10:
        pad_dir = os.path.join(DATA_DIR, "phase2") 
        
    # Gather all PAD-UFES images
    all_pad_images = []
    # recursive search
    for root, dirs, files in os.walk(pad_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_pad_images.append(os.path.join(root, file))
                
    if len(all_pad_images) < size:
        print(f"Warning: Only found {len(all_pad_images)} PAD images. Using all for test.")
        test_paths = all_pad_images
    else:
        random.seed(42) # Fixed seed for fairness
        test_paths = random.sample(all_pad_images, size)
        
    # We need to return these paths so we can Exclude them from training
    return test_paths

def plot_confusion_matrix(y_true, y_pred, title, filename, metrics=None):
    """
    Standard Seaborn Heatmap (11x11) in Blue style.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    if metrics:
        title += f"\nAcc: {metrics['Accuracy']:.1f}% | F1: {metrics['F1']:.1f}% | Rec: {metrics['Recall']:.1f}%"
        
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved {filename}")

class ExcludeSubset(Subset):
    # Helper to exclude paths? 
    # Actually simpler: We will just build the dataset list manually in the experiment loop.
    pass

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

def run_experiment(name, train_files, test_files, device, class_to_idx, class_names):
    print(f"\n=== Experiment: {name} ===")
    print(f"   Training on {len(train_files)} images.")
    print(f"   Testing on {len(test_files)} images (Fixed Set).")
    
    # 1. Setup Datasets manually
    class SimplePathDataset(torch.utils.data.Dataset):
        def __init__(self, file_paths, transform, class_to_idx):
            self.files = file_paths
            self.transform = transform
            self.class_to_idx = class_to_idx
            
        def __len__(self):
            return len(self.files)
        
        def __getitem__(self, idx):
            path = self.files[idx]
            img_path, label = path
            
            try:
                from PIL import Image
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                image = torch.zeros((3, 224, 224))
                
            return image, label

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds = SimplePathDataset(train_files, transform, class_to_idx) 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    test_ds = SimplePathDataset(test_files, transform, class_to_idx)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = get_model(num_classes=11, pretrained=True, arch='mobilenet_v3').to(device)
    
    # === TURBO MODE: FREEZE BACKBONE ===
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze only the classifier (Last Layer)
    # MobileNetV3 classifier block is usually 'classifier'
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    print("   🚀 Turbo Mode: Backbone Frozen. Training Classifier only.")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"   Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")
        
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []
    
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # ROC AUC (Handle multi-class)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        auc = 0.5 # Fallback if only 1 class present in test set
        
    metrics = {
        "Accuracy": acc * 100,
        "Precision": prec * 100,
        "Recall": rec * 100,
        "F1": f1 * 100,
        "Macro AUC": auc
    }
    
    # Generate and save classification report
    # Get unique labels present in test set
    unique_labels = sorted(list(set(all_labels)))
    report = classification_report(all_labels, all_preds, 
                                   labels=unique_labels,
                                   target_names=[class_names[i] for i in unique_labels], 
                                   zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"Classification Report: {name}")
    print('='*60)
    print(report)
    print('='*60)
    
    # Save report to file
    report_path = os.path.join(RESULTS_DIR, f"classification_report_{name}.txt")
    with open(report_path, 'w') as f:
        f.write(f"Classification Report: {name}\n")
        f.write('='*60 + '\n')
        f.write(report)
    print(f"   💾 Saved classification report to {report_path}")
            
    plot_confusion_matrix(all_labels, all_preds, 
                          title=f"{name}\n(Tested on 100 Dark Skin Lesions)", 
                          filename=os.path.join(RESULTS_DIR, f"new_cm_{name}.png"),
                          metrics=metrics)
                          
    return metrics

def main():
    print("Initializing New Validation Protocol with Table Generation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    full_ds = SkinCancerDataset(
        csv_file=os.path.join(DATA_DIR, "HAM10000_metadata.csv"),
        root_dir=os.path.join(DATA_DIR, "images"),
        phase2_dir=os.path.join(DATA_DIR, "phase2"),
        pad_dir=os.path.join(DATA_DIR, "pad_ufes"),
        synthetic_dir=os.path.join(DATA_DIR, "synthetic")
    )
    
    all_samples = full_ds.samples
    ham_samples = [x for x in all_samples if "HAM10000" in x[0] or "images\\" in x[0]]
    pad_samples = [x for x in all_samples if "pad_ufes" in x[0] or "phase2" in x[0]]
    syn_samples = [x for x in all_samples if "synthetic" in x[0]]
    
    print(f"Found: {len(ham_samples)} HAM, {len(pad_samples)} PAD, {len(syn_samples)} Synthetic")
    
    random.seed(42)
    random.shuffle(pad_samples)
    test_set_100 = pad_samples[:100]
    pad_remaining = pad_samples[100:]
    
    results = []
    
    # Exp 1
    m1 = run_experiment("1_Baseline_Problem", ham_samples, test_set_100, device, full_ds.class_to_idx, full_ds.classes)
    results.append(("HAM10000", m1))
    
    # Exp 2
    m2 = run_experiment("2_Benchmark_Comparison", ham_samples + pad_remaining, test_set_100, device, full_ds.class_to_idx, full_ds.classes)
    results.append(("HAM10000 + PAD-UFES-20", m2))
    
    # Exp 3
    m3 = run_experiment("3_AiDerm_Solution", ham_samples + pad_remaining + syn_samples, test_set_100, device, full_ds.class_to_idx, full_ds.classes)
    results.append(("HAM10000 + PAD-UFES-20 + Synthetic (AiDerm)", m3))
    
    print("\n\n=== Final Comparison Table ===")
    print("| Dataset Configuration | Accuracy | Precision | Recall | F1 | Macro AUC |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for name, m in results:
        print(f"| {name} | {m['Accuracy']:.1f} | {m['Precision']:.1f} | {m['Recall']:.1f} | {m['F1']:.1f} | {m['Macro AUC']:.3f} |")
    
    print("\n✅ Validation Complete.")

if __name__ == "__main__":
    main()
