import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from src.dataset import SkinCancerDataset
from src.model import get_model
import os
import csv

def train_model():
    # Hyperparameters
    BATCH_SIZE = 24 # Reduced to fit alongside generation in VRAM
    LEARNING_RATE = 0.00001 
    EPOCHS = 5 # 5 high-quality epochs for fast turnaround
    DATA_DIR = "data"
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transforms - Aggressive Augmentation for Precision
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30), # Increased rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05), # Stronger jitter
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Shift
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    csv_file = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
    img_dir = os.path.join(DATA_DIR, "images")
    phase2_dir = os.path.join(DATA_DIR, "phase2") # Use specific Phase 2 folder
    synthetic_dir = os.path.join(DATA_DIR, "synthetic") # Massive Synthetic Infusion
    
    if not os.path.exists(csv_file) or not os.path.exists(img_dir):
        print("Data not found. Please run src/data_setup.py first.")
        return

    # Load full dataset (Phase 1 + Phase 2 + Phase 3 Synthetic)
    full_dataset = SkinCancerDataset(
        csv_file=csv_file, 
        root_dir=img_dir, 
        phase2_dir=phase2_dir,
        synthetic_dir=synthetic_dir,
        transform=transform_train
    )
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Validation Transform hack
    val_dataset.dataset.transform = transform_val 

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Dynamic Class Weighting
    print("Calculating class weights...")
    label_counts = {}
    for _, label in full_dataset.samples:
        label_counts[label] = label_counts.get(label, 0) + 1
        
    num_samples = len(full_dataset)
    num_classes = len(full_dataset.classes)
    
    # Weight = Total / (Num_Classes * Count)
    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 0)
        if count == 0:
            weights.append(1.0) # Avoid div by zero
        else:
            weights.append(num_samples / (num_classes * count))
            
    class_weights = torch.tensor(weights).float().to(device)
    print(f"Computed Class Weights: {class_weights}")

    # Model (11 Classes)
    model = get_model(num_classes=num_classes).to(device)
    
    # RESUME LOGIC: Load existing weights if they exist (so we don't start from zero)
    if os.path.exists("model.pth"):
        print("🔄 Loading saved model checkpoint to resume training...")
        try:
            model.load_state_dict(torch.load("model.pth", map_location=device))
            print("✅ Checkpoint loaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint (structure might be different): {e}")
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # Training Loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 0:
                 print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")
            
        avg_loss = running_loss/len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss}")
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Accuracy: {val_acc:.2f}%, Val Loss: {avg_val_loss:.4f}")
        
        # Scheduler step
        scheduler.step(avg_val_loss)

        # Log history to CSV
        log_file = "training_history.csv"
        file_exists = os.path.isfile(log_file)
        with open(log_file, mode='a', newline='') as f:
            headers = ['epoch', 'train_loss', 'val_loss', 'val_acc']
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
                'val_acc': val_acc
            })

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "model.pth")
            print(f"Saved new best model with accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_model()
