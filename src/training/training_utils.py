"""
Advanced Training Utilities
- Learning rate schedulers
- Class weights for imbalanced data
- Training loop with mixed precision
"""
import torch
import torch.nn as nn
import numpy as np
from collections import Counter

def calculate_class_weights(dataset, num_classes):
    """
    Calculate class weights for imbalanced dataset
    Helps model pay more attention to rare classes
    """
    # Count samples per class
    labels = [label for _, label in dataset.samples]
    class_counts = Counter(labels)
    
    # Calculate weights (inverse frequency)
    total = len(labels)
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)  # Avoid division by zero
        weight = total / (num_classes * count)
        weights.append(weight)
    
    weights = torch.FloatTensor(weights)
    print(f"📊 Class weights: {weights}")
    return weights

def get_lr_scheduler(optimizer, scheduler_type='cosine', epochs=50):
    """
    Get learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: 'cosine', 'step', or 'plateau'
        epochs: Total number of epochs
    """
    if scheduler_type == 'cosine':
        # Cosine annealing - smooth decay
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        print("✅ Using Cosine Annealing LR Scheduler")
    
    elif scheduler_type == 'step':
        # Step decay - drops at specific epochs
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=epochs//3, gamma=0.1
        )
        print("✅ Using Step LR Scheduler")
    
    elif scheduler_type == 'plateau':
        # Reduce on plateau - adaptive
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        print("✅ Using Reduce on Plateau LR Scheduler")
    
    else:
        scheduler = None
        print("ℹ️ No LR scheduler")
    
    return scheduler

class EarlyStopping:
    """
    Stop training when validation metric stops improving
    """
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_metric):
        score = val_metric
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"⚠️ Early stopping triggered after {self.counter} epochs")
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop

def train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp=True):
    """
    Train for one epoch with optional mixed precision
    
    Args:
        use_amp: Use automatic mixed precision (faster on GPU)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            # Mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Normal training
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    """
    Validate model
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy
