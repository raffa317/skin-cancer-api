"""
Master Training Script
Combines everything: ISIC + PAD + HAM + Synthetic + Heavy Augmentation + ResNet50
"""
import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from src.model import get_model
from src.dataset import SkinCancerDataset
from src.isic_loader import ISICDataset
from src.augmentation import HeavyAugmentation
from src.training_utils import (
    calculate_class_weights,
    get_lr_scheduler,
    EarlyStopping,
    train_one_epoch,
    validate
)
import random

# Configuration
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
ARCH = 'resnet50'  # resnet50, mobilenet_v3, efficientnet, densenet
USE_CLASS_WEIGHTS = True
USE_LR_SCHEDULER = True
USE_EARLY_STOPPING = True
USE_MIXED_PRECISION = True

DATA_DIR = "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("🚀 MASTER TRAINING: Full Pipeline")
    print(f"Architecture: {ARCH}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print("-" * 60)
    
    # Transforms
    train_transform = HeavyAugmentation(train=True)
    val_transform = HeavyAugmentation(train=False)
    
    # Load all datasets
    print("\n📂 Loading datasets...")
    
    # HAM10000 + PAD + Synthetic
    ham_pad_synthetic = SkinCancerDataset(
        csv_file=os.path.join(DATA_DIR, "HAM10000_metadata.csv"),
        root_dir=os.path.join(DATA_DIR, "images"),
        phase2_dir=os.path.join(DATA_DIR, "phase2"),
        pad_dir=os.path.join(DATA_DIR, "pad_ufes"),
        synthetic_dir=os.path.join(DATA_DIR, "synthetic"),
        transform=train_transform
    )
    
    # ISIC 2019 (if available)
    isic_csv = os.path.join(DATA_DIR, "isic_2019", "ISIC_2019_Training_GroundTruth.csv")
    isic_dir = os.path.join(DATA_DIR, "isic_2019")
    
    if os.path.exists(isic_csv):
        print("✅ ISIC dataset found, loading...")
        isic_dataset = ISICDataset(
            csv_file=isic_csv,
            images_dir=isic_dir,
            transform=train_transform
        )
        full_dataset = ConcatDataset([ham_pad_synthetic, isic_dataset])
        print(f"📊 Total samples: {len(full_dataset)} (HAM+PAD+Synthetic+ISIC)")
    else:
        print("⚠️  ISIC dataset not found, using HAM+PAD+Synthetic only")
        full_dataset = ham_pad_synthetic
        print(f"📊 Total samples: {len(full_dataset)}")
    
    # Train/Val split
    random.seed(42)
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    
    split_point = int(0.85 * len(indices))
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,  pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    print(f"\n🧠 Loading {ARCH} model...")
    model = get_model(num_classes=11, pretrained=True, arch=ARCH).to(DEVICE)
    
    # Class weights
    if USE_CLASS_WEIGHTS:
        class_weights = calculate_class_weights(ham_pad_synthetic, num_classes=11).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # LR Scheduler
    if USE_LR_SCHEDULER:
        scheduler = get_lr_scheduler(optimizer, scheduler_type='cosine', epochs=EPOCHS)
    else:
        scheduler = None
    
    # Early stopping
    if USE_EARLY_STOPPING:
        early_stopping = EarlyStopping(patience=10)
    else:
        early_stopping = None
    
    # Training loop
    print("\n🏋️ Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            use_amp=USE_MIXED_PRECISION
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Update LR
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"models/best_{ARCH}_model.pth")
            print(f"💾 Saved best model at epoch {epoch+1}")
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"Best: {best_val_acc:.2f}% | LR: {current_lr:.6f}")
        
        # Early stopping
        if early_stopping:
            if early_stopping(val_acc):
                print(f"\n⛔ Early stopping triggered at epoch {epoch+1}")
                break
    
    print(f"\n✅ Training Complete!")
    print(f"🎯 Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), f"models/final_{ARCH}_model.pth")
    print(f"💾 Final model saved")

if __name__ == "__main__":
    # Create models directory
    os.makedirs("models", exist_ok=True)
    main()
