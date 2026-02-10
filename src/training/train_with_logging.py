"""
Train EfficientNet with Full Logging for Real Training Curves
This will save epoch-by-epoch accuracy and loss for plotting
"""
import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from src.model import get_model
from src.dataset import SkinCancerDataset
from src.augmentation import HeavyAugmentation
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

print("="*60)
print("TRAINING WITH FULL LOGGING FOR REAL CURVES")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Epochs: {EPOCHS}")
print(f"Architecture: EfficientNet-B0")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LEARNING_RATE}")
print("="*60)

# Prepare datasets
train_transform = HeavyAugmentation(train=True)
val_transform = HeavyAugmentation(train=False)

# Load full dataset
full_dataset = SkinCancerDataset(
    csv_file=os.path.join(DATA_DIR, "HAM10000_metadata.csv"),
    root_dir=os.path.join(DATA_DIR, "images"),
    phase2_dir=os.path.join(DATA_DIR, "phase2"),
    pad_dir=os.path.join(DATA_DIR, "pad_ufes"),
    synthetic_dir=os.path.join(DATA_DIR, "synthetic"),
    transform=train_transform
)

# Create train/val split (same as ensemble training)
import random
random.seed(42)
indices = list(range(len(full_dataset.samples)))
random.shuffle(indices)
split_point = int(0.85 * len(indices))
train_indices = indices[:split_point]
val_indices = indices[split_point:]

print(f"\n📊 Dataset Split:")
print(f"   Training samples: {len(train_indices)}")
print(f"   Validation samples: {len(val_indices)}")

# Create datasets
train_dataset = Subset(full_dataset, train_indices)
val_dataset_temp = SkinCancerDataset(
    csv_file=os.path.join(DATA_DIR, "HAM10000_metadata.csv"),
    root_dir=os.path.join(DATA_DIR, "images"),
    phase2_dir=os.path.join(DATA_DIR, "phase2"),
    pad_dir=os.path.join(DATA_DIR, "pad_ufes"),
    synthetic_dir=os.path.join(DATA_DIR, "synthetic"),
    transform=val_transform
)
val_dataset = Subset(val_dataset_temp, val_indices)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Model, loss, optimizer
model = get_model(num_classes=11, pretrained=True, arch='efficientnet')
model = model.to(DEVICE)

# Calculate class weights
train_labels = [full_dataset.samples[i][1] for i in train_indices]
class_counts = np.bincount(train_labels, minlength=11)
class_weights = 1.0 / (class_counts + 1)
class_weights = class_weights / class_weights.sum() * 11
class_weights = torch.FloatTensor(class_weights).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# History tracking
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'epochs': [],
    'learning_rate': []
}

print(f"\n🚀 Starting training...")
print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # === TRAINING ===
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS} [Train]')
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100.*train_correct/train_total:.2f}%'})
    
    train_loss = train_loss / train_total
    train_acc = 100. * train_correct / train_total
    
    # === VALIDATION ===
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch}/{EPOCHS} [Val]')
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100.*val_correct/val_total:.2f}%'})
    
    val_loss = val_loss / val_total
    val_acc = 100. * val_correct / val_total
    
    # Update learning rate
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save to history
    history['train_loss'].append(float(train_loss))
    history['train_acc'].append(float(train_acc))
    history['val_loss'].append(float(val_loss))
    history['val_acc'].append(float(val_acc))
    history['epochs'].append(epoch)
    history['learning_rate'].append(float(current_lr))
    
    # Print epoch summary
    print(f"\n📊 Epoch {epoch}/{EPOCHS} Summary:")
    print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
    print(f"   LR: {current_lr:.6f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'models/efficientnet_logged_best.pth')
        print(f"   ✅ New best model saved! Val Acc: {val_acc:.2f}%")
    
    # Save history every 5 epochs
    if epoch % 5 == 0:
        with open('training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        print(f"   💾 History saved (intermediate)")
    
    print("-" * 60)

# Save final history
with open('training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"Final Training Accuracy: {train_acc:.2f}%")
print(f"Final Validation Accuracy: {val_acc:.2f}%")
print(f"\n📁 Files saved:")
print(f"   - models/efficientnet_logged_best.pth (best model)")
print(f"   - training_history.json (full training history)")
print(f"\n⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)
