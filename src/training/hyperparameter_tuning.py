"""
Hyperparameter Tuning for 90% Target
Tests multiple configurations to find best settings
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
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"

# Configurations to test
CONFIGS = [
    # Config 1: Lower LR, more epochs
    {
        'name': 'low_lr_long',
        'lr': 0.0001,
        'epochs': 60,
        'batch_size': 32,
        'weight_decay': 1e-4
    },
    # Config 2: Even lower LR, very long
    {
        'name': 'very_low_lr',
        'lr': 0.00005,
        'epochs': 80,
        'batch_size': 32,
        'weight_decay': 1e-4
    },
    # Config 3: Larger batch, moderate LR
    {
        'name': 'large_batch',
        'lr': 0.0005,
        'epochs': 50,
        'batch_size': 64,
        'weight_decay': 1e-3
    },
]

def train_with_config(config, train_loader, val_loader, arch='efficientnet'):
    """Train model with specific configuration"""
    print(f"\n{'#'*60}")
    print(f"CONFIG: {config['name']}")
    print(f"LR: {config['lr']}, Epochs: {config['epochs']}, Batch: {config['batch_size']}")
    print('#'*60)
    
    model = get_model(num_classes=11, pretrained=True, arch=arch).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    # Cosine annealing with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['epochs'],
        eta_min=1e-7
    )
    
    best_val_acc = 0.0
    history = []
    
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("tuned_models", exist_ok=True)
            torch.save(model.state_dict(), f"tuned_models/{config['name']}_best.pth")
            print(f"💾 New best at epoch {epoch+1}: {best_val_acc:.2f}%")
        
        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{config['epochs']} - "
                  f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | "
                  f"Best: {best_val_acc:.2f}% | LR: {lr:.7f}")
        
        # Early stopping if we hit 90%
        if best_val_acc >= 90.0:
            print(f"\n🎉 TARGET ACHIEVED! Stopping early at {best_val_acc:.2f}%")
            break
    
    # Save history
    with open(f"tuned_models/{config['name']}_history.json", 'w') as f:
        json.dump({
            'config': config,
            'best_val_acc': best_val_acc,
            'history': history
        }, f, indent=2)
    
    print(f"\n✅ {config['name']} complete! Best: {best_val_acc:.2f}%")
    return best_val_acc

def main():
    print("🎯 HYPERPARAMETER TUNING FOR 90% TARGET")
    print(f"Device: {DEVICE}")
    print(f"Testing {len(CONFIGS)} configurations")
    print("-" * 60)
    
    # Load data (best setup: HAM + PAD + Synthetic)
    print("\n📂 Loading datasets...")
    train_transform = HeavyAugmentation(train=True)
    val_transform = HeavyAugmentation(train=False)
    
    dataset_train = SkinCancerDataset(
        csv_file=os.path.join(DATA_DIR, "HAM10000_metadata.csv"),
        root_dir=os.path.join(DATA_DIR, "images"),
        phase2_dir=os.path.join(DATA_DIR, "phase2"),
        pad_dir=os.path.join(DATA_DIR, "pad_ufes"),
        synthetic_dir=os.path.join(DATA_DIR, "synthetic"),
        transform=train_transform
    )
    
    dataset_val = SkinCancerDataset(
        csv_file=os.path.join(DATA_DIR, "HAM10000_metadata.csv"),
        root_dir=os.path.join(DATA_DIR, "images"),
        phase2_dir=os.path.join(DATA_DIR, "phase2"),
        pad_dir=os.path.join(DATA_DIR, "pad_ufes"),
        synthetic_dir=os.path.join(DATA_DIR, "synthetic"),
        transform=val_transform
    )
    
    # Split
    random.seed(42)
    indices = list(range(len(dataset_train.samples)))
    random.shuffle(indices)
    split_point = int(0.85 * len(indices))
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    
    train_dataset = torch.utils.data.Subset(dataset_train, train_indices)
    val_dataset = torch.utils.data.Subset(dataset_val, val_indices)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Test each configuration
    results = {}
    best_overall = 0.0
    best_config_name = ""
    
    for i, config in enumerate(CONFIGS, 1):
        print(f"\n\n{'='*60}")
        print(f"TESTING CONFIG {i}/{len(CONFIGS)}")
        print('='*60)
        
        # Create loaders with config's batch size
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=2, 
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True
        )
        
        acc = train_with_config(config, train_loader, val_loader)
        results[config['name']] = acc
        
        if acc > best_overall:
            best_overall = acc
            best_config_name = config['name']
        
        # If we hit 90%, stop testing
        if acc >= 90.0:
            print(f"\n🎯 90% TARGET REACHED! Stopping tuning.")
            break
    
    # Final summary
    print(f"\n\n{'='*60}")
    print("🏆 HYPERPARAMETER TUNING COMPLETE")
    print('='*60)
    print("\nResults:")
    for name, acc in results.items():
        marker = " ⭐" if name == best_config_name else ""
        print(f"  {name:20s}: {acc:.2f}%{marker}")
    
    print(f"\n🎯 Best Configuration: {best_config_name}")
    print(f"🎯 Best Accuracy: {best_overall:.2f}%")
    
    if best_overall >= 90.0:
        print(f"\n🎉🎉🎉 TARGET ACHIEVED! 90%+ REACHED! 🎉🎉🎉")
        print(f"Improvement: 88.53% → {best_overall:.2f}% = +{best_overall - 88.53:.2f}%")
    else:
        print(f"\n📊 Progress: 88.53% → {best_overall:.2f}% = +{best_overall - 88.53:.2f}%")
        print(f"   Still {90.0 - best_overall:.2f}% away from 90%")
    
    print(f"\n💾 Models saved to: tuned_models/")

if __name__ == "__main__":
    main()
