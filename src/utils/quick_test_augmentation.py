"""
Quick Test: Augmentation Only
Test heavy augmentation on existing data (no ISIC needed)
This can run NOW while ISIC downloads
"""
import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import get_model
from src.dataset import SkinCancerDataset
from src.augmentation import HeavyAugmentation
import random

# Config
EPOCHS = 10  # Quick test
BATCH_SIZE = 16
DATA_DIR = "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def quick_test():
    print("🔥 Quick Test: Heavy Augmentation Only")
    print(f"Device: {DEVICE}")
    
    # Heavy augmentation transform
    train_transform = HeavyAugmentation(train=True)
    val_transform = HeavyAugmentation(train=False)
    
    # Load dataset with augmentation
    dataset = SkinCancerDataset(
        csv_file=os.path.join(DATA_DIR, "HAM10000_metadata.csv"),
        root_dir=os.path.join(DATA_DIR, "images"),
        phase2_dir=os.path.join(DATA_DIR, "phase2"),
        pad_dir=os.path.join(DATA_DIR, "pad_ufes"),
        synthetic_dir=os.path.join(DATA_DIR, "synthetic"),
        transform=train_transform
    )
    
    print(f"Total samples: {len(dataset.samples)}")
    
    # Split for validation
    random.seed(42)
    indices = list(range(len(dataset.samples)))
    random.shuffle(indices)
    
    split_point = int(0.8 * len(indices))
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = get_model(num_classes=11, pretrained=True, arch='mobilenet_v3').to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)
        
        val_acc = 100 * correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)")
    
    print(f"\n✅ Quick Test Complete!")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print("\nThis is with HEAVY AUGMENTATION on your existing data.")
    print("When ISIC finishes downloading, we'll add that too for even better results!")

if __name__ == "__main__":
    quick_test()
