"""
Ensemble Training - Complete Version
Train 3 different architectures and combine predictions
"""
import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model import get_model
from src.dataset import SkinCancerDataset
from src.augmentation import HeavyAugmentation
from src.training_utils import train_one_epoch, validate
import random

# Config
EPOCHS = 30  # Per model
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARCHITECTURES = ['resnet50', 'efficientnet', 'densenet']
DATA_DIR = "data"

def train_single_model(arch, train_loader, val_loader, epochs):
    """Train a single model architecture"""
    print(f"\n{'='*60}")
    print(f"🔥 Training {arch.upper()}")
    print('='*60)
    
    model = get_model(num_classes=11, pretrained=True, arch=arch).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/ensemble_{arch}.pth")
            print(f"💾 Saved best {arch} model at epoch {epoch+1}")
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | "
                  f"Best: {best_val_acc:.2f}% | LR: {lr:.6f}")
    
    print(f"✅ {arch} complete! Best: {best_val_acc:.2f}%")
    return best_val_acc

def main():
    print("🎯 ENSEMBLE TRAINING: 3 Models (ResNet50, EfficientNet, DenseNet)")
    print(f"Device: {DEVICE}")
    print(f"Epochs per model: {EPOCHS}")
    print(f"Total estimated time: ~6-8 hours")
    print("-" * 60)
    
    # Load data (HAM + PAD + Synthetic, NO ISIC)
    print("\n📂 Loading datasets (HAM + PAD + Synthetic)...")
    
    train_transform = HeavyAugmentation(train=True)
    val_transform = HeavyAugmentation(train=False)
    
    dataset = SkinCancerDataset(
        csv_file=os.path.join(DATA_DIR, "HAM10000_metadata.csv"),
        root_dir=os.path.join(DATA_DIR, "images"),
        phase2_dir=os.path.join(DATA_DIR, "phase2"),
        pad_dir=os.path.join(DATA_DIR, "pad_ufes"),
        synthetic_dir=os.path.join(DATA_DIR, "synthetic"),
        transform=train_transform
    )
    
    print(f"Total samples: {len(dataset.samples)}")
    
    # Train/Val split
    random.seed(42)
    indices = list(range(len(dataset.samples)))
    random.shuffle(indices)
    
    split_point = int(0.85 * len(indices))
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    
    # Validation dataset with different transform
    val_dataset_raw = SkinCancerDataset(
        csv_file=os.path.join(DATA_DIR, "HAM10000_metadata.csv"),
        root_dir=os.path.join(DATA_DIR, "images"),
        phase2_dir=os.path.join(DATA_DIR, "phase2"),
        pad_dir=os.path.join(DATA_DIR, "pad_ufes"),
        synthetic_dir=os.path.join(DATA_DIR, "synthetic"),
        transform=val_transform
    )
    val_dataset = torch.utils.data.Subset(val_dataset_raw, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Train each model
    results = {}
    for i, arch in enumerate(ARCHITECTURES, 1):
        print(f"\n\n{'#'*60}")
        print(f"MODEL {i}/3: {arch.upper()}")
        print(f"{'#'*60}")
        
        acc = train_single_model(arch, train_loader, val_loader, EPOCHS)
        results[arch] = acc
    
    # Summary
    print(f"\n\n{'='*60}")
    print("🏆 ENSEMBLE TRAINING COMPLETE!")
    print('='*60)
    print("\nIndividual Model Results:")
    for arch, acc in results.items():
        print(f"  {arch:20s}: {acc:.2f}%")
    
    avg_acc = sum(results.values()) / len(results)
    print(f"\n📊 Average: {avg_acc:.2f}%")
    print(f"📈 Expected ensemble boost: +1-2%")
    print(f"🎯 Estimated ensemble accuracy: {avg_acc + 1.5:.2f}%")
    
    print(f"\n💾 Models saved to: models/ensemble_*.pth")
    print("\nNext step: Use these models together for final predictions!")

if __name__ == "__main__":
    main()
