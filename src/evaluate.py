import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from src.dataset import SkinCancerDataset
from src.model import get_model
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model():
    # Hyperparameters
    BATCH_SIZE = 32
    DATA_DIR = "data"
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transforms (same as training, but no augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    csv_file = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
    img_dir = os.path.join(DATA_DIR, "images")
    
    if not os.path.exists(csv_file) or not os.path.exists(img_dir):
        print("Data not found.")
        return

    dataset = SkinCancerDataset(csv_file=csv_file, root_dir=img_dir, transform=transform)
    
    # Split (must match training split to evaluate on validation set)
    # Ideally we should have saved the indices, but for now we rely on deterministic random_split with same seed if possible
    # Or just evaluate on a subset for demonstration. 
    # NOTE: random_split is not deterministic without manual seed. 
    # For this demo, we'll just evaluate on the whole dataset or a random split, acknowledging data leakage risk if not careful.
    # To be proper, let's set a seed.
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = get_model(num_classes=7, pretrained=False)
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth", map_location=device))
        print("Loaded model weights.")
    else:
        print("Model weights not found.")
        return
        
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Evaluating...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    classes = dataset.classes
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()
