"""
Ensemble Prediction System
Combines predictions from 3 trained models for final accuracy
"""
import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from src.model import get_model
from src.dataset import SkinCancerDataset
from src.augmentation import HeavyAugmentation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"
BATCH_SIZE = 32

# Model configs
MODELS = {
    'resnet50': 'models/ensemble_resnet50.pth',
    'efficientnet': 'models/ensemble_efficientnet.pth',
    'densenet': 'models/ensemble_densenet.pth'
}

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc', 'acne', 'eczema', 'tinea', 'unknown']

def load_model(arch, model_path):
    """Load a trained model"""
    model = get_model(num_classes=11, pretrained=False, arch=arch).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_predictions(model, data_loader):
    """Get predictions from a single model"""
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)

def ensemble_predict(model_predictions, method='majority_vote'):
    """
    Combine predictions from multiple models
    
    Args:
        model_predictions: List of (preds, probs, labels) tuples
        method: 'majority_vote' or 'average_probs'
    """
    labels = model_predictions[0][2]  # Labels are same for all
    
    if method == 'majority_vote':
        # Stack all predictions
        all_preds = np.stack([p[0] for p in model_predictions])
        
        # Majority vote for each sample
        ensemble_preds = []
        for i in range(all_preds.shape[1]):
            votes = all_preds[:, i]
            # Most common prediction
            ensemble_preds.append(np.bincount(votes).argmax())
        
        return np.array(ensemble_preds), labels
    
    elif method == 'average_probs':
        # Average probabilities
        all_probs = np.stack([p[1] for p in model_predictions])
        avg_probs = np.mean(all_probs, axis=0)
        
        # Get predictions from averaged probabilities
        ensemble_preds = np.argmax(avg_probs, axis=1)
        
        return ensemble_preds, labels

def evaluate_and_save(preds, labels, name, class_names):
    """Evaluate predictions and save results"""
    # Get unique labels present
    unique_labels = sorted(list(set(labels)))
    present_class_names = [class_names[i] for i in unique_labels]
    
    # Accuracy
    accuracy = accuracy_score(labels, preds) * 100
    
    # Classification report
    report = classification_report(labels, preds,
                                   labels=unique_labels,
                                   target_names=present_class_names, 
                                   zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"{name}")
    print('='*60)
    print(f"Accuracy: {accuracy:.2f}%")
    print("\nClassification Report:")
    print(report)
    
    # Save report
    os.makedirs("ensemble_results", exist_ok=True)
    with open(f"ensemble_results/{name.lower().replace(' ', '_')}_report.txt", 'w') as f:
        f.write(f"{name}\n")
        f.write('='*60 + '\n')
        f.write(f"Accuracy: {accuracy:.2f}%\n\n")
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(12, 10))
    cm_labels = present_class_names if len(present_class_names) == cm.shape[0] else class_names
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=cm_labels, yticklabels=cm_labels)
    plt.title(f'{name}\nAccuracy: {accuracy:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"ensemble_results/{name.lower().replace(' ', '_')}_cm.png", dpi=300)
    plt.close()
    
    return accuracy

def main():
    print("🎯 ENSEMBLE PREDICTION SYSTEM")
    print(f"Device: {DEVICE}")
    print("-" * 60)
    
    # Load validation data
    print("\n📂 Loading validation data...")
    val_transform = HeavyAugmentation(train=False)
    
    dataset = SkinCancerDataset(
        csv_file=os.path.join(DATA_DIR, "HAM10000_metadata.csv"),
        root_dir=os.path.join(DATA_DIR, "images"),
        phase2_dir=os.path.join(DATA_DIR, "phase2"),
        pad_dir=os.path.join(DATA_DIR, "pad_ufes"),
        synthetic_dir=os.path.join(DATA_DIR, "synthetic"),
        transform=val_transform
    )
    
    # Use same validation split as training
    random.seed(42)
    indices = list(range(len(dataset.samples)))
    random.shuffle(indices)
    split_point = int(0.85 * len(indices))
    val_indices = indices[split_point:]
    
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Load models and get predictions
    print("\n🔮 Getting predictions from individual models...")
    model_predictions = []
    individual_accuracies = {}
    
    for arch, model_path in MODELS.items():
        print(f"\n  Loading {arch}...")
        model = load_model(arch, model_path)
        preds, probs, labels = get_predictions(model, val_loader)
        model_predictions.append((preds, probs, labels))
        
        # Individual accuracy
        acc = accuracy_score(labels, preds) * 100
        individual_accuracies[arch] = acc
        print(f"  {arch}: {acc:.2f}%")
    
    # Ensemble predictions
    print("\n\n🤝 Combining predictions...")
    
    print("\n1️⃣ Method: Majority Vote")
    ensemble_preds_vote, labels = ensemble_predict(model_predictions, method='majority_vote')
    acc_vote = evaluate_and_save(ensemble_preds_vote, labels, "Ensemble (Majority Vote)", CLASS_NAMES)
    
    print("\n2️⃣ Method: Average Probabilities")
    ensemble_preds_avg, labels = ensemble_predict(model_predictions, method='average_probs')
    acc_avg = evaluate_and_save(ensemble_preds_avg, labels, "Ensemble (Average Probs)", CLASS_NAMES)
    
    # Summary
    print("\n\n" + "="*60)
    print("🏆 FINAL RESULTS")
    print("="*60)
    print("\nIndividual Models:")
    for arch, acc in individual_accuracies.items():
        print(f"  {arch:20s}: {acc:.2f}%")
    
    print(f"\nEnsemble Methods:")
    print(f"  Majority Vote        : {acc_vote:.2f}%")
    print(f"  Average Probabilities: {acc_avg:.2f}%")
    
    best_method = "Majority Vote" if acc_vote > acc_avg else "Average Probabilities"
    best_acc = max(acc_vote, acc_avg)
    
    print(f"\n🎯 Best Result: {best_method} with {best_acc:.2f}%")
    
    if best_acc >= 90:
        print("\n🎉 TARGET ACHIEVED! You hit 90%+!")
    elif best_acc >= 88:
        print(f"\n✅ Excellent! Only {90 - best_acc:.2f}% away from 90%")
        print("   Consider hyperparameter tuning for final push.")
    else:
        print(f"\n📊 Good progress. {90 - best_acc:.2f}% away from 90%")
    
    print(f"\n💾 Results saved to: ensemble_results/")

if __name__ == "__main__":
    main()
