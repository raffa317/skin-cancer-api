import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from src.dataset import SkinCancerDataset
from src.model import get_model
import os

def train_model():
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001 # Lower LR for fine-tuning
    EPOCHS = 20
    DATA_DIR = "data"
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transforms - Enhanced Augmentation
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
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
    
    if not os.path.exists(csv_file) or not os.path.exists(img_dir):
        print("Data not found. Please run src/data_setup.py first.")
        return

    # Load full dataset
    full_dataset = SkinCancerDataset(csv_file=csv_file, root_dir=img_dir, transform=transform_train)
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply validation transform to validation set (hacky way since random_split shares transform)
    # Ideally we should split indices and create two datasets, but for now we'll stick to this or just use same transform
    # To do it properly:
    val_dataset.dataset.transform = transform_val 

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Class Weights for Imbalance
    # Counts: nv: 6705, mel: 1113, bkl: 1099, bcc: 514, akiec: 327, vasc: 142, df: 115
    # Total: 10015
    # Weights = Total / (Num_Classes * Count)
    # Approximate weights (normalized roughly)
    class_weights = torch.tensor([
        10015/(7*327),  # akiec
        10015/(7*514),  # bcc
        10015/(7*1099), # bkl
        10015/(7*115),  # df
        10015/(7*1113), # mel
        10015/(7*6705), # nv
        10015/(7*142)   # vasc
    ]).to(device)
    
    print(f"Class weights: {class_weights}")

    # Model
    model = get_model(num_classes=7).to(device)
    
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

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "model.pth")
            print(f"Saved new best model with accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_model()
