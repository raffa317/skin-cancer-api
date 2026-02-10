"""
Final Evaluation & Report Generation
Generate comprehensive results for paper
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model import get_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def evaluate_model(model_path, arch, test_loader, class_names, device):
    """Evaluate a single model"""
    model = get_model(num_classes=11, pretrained=False, arch=arch).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)

def ensemble_predict(model_paths, archs, test_loader, device):
    """Ensemble prediction from multiple models"""
    all_model_preds = []
    
    for model_path, arch in zip(model_paths, archs):
        preds, labels = evaluate_model(model_path, arch, test_loader, None, device)
        all_model_preds.append(preds)
    
    # Majority voting
    all_model_preds = np.array(all_model_preds)
    ensemble_preds = []
    
    for i in range(all_model_preds.shape[1]):
        votes = all_model_preds[:, i]
        # Most common prediction
        ensemble_preds.append(np.bincount(votes).argmax())
    
    return np.array(ensemble_preds), labels

def generate_final_report(preds, labels, class_names, output_dir="final_results"):
    """Generate comprehensive evaluation report"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Classification report
    report = classification_report(labels, preds, target_names=class_names)
    print("\n" + "="*60)
    print("FINAL CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    
    with open(f"{output_dir}/classification_report.txt", 'w') as f:
        f.write(report)
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Final Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
    plt.close()
    
    # Accuracy
    accuracy = (preds == labels).mean() * 100
    print(f"\n🎯 FINAL ACCURACY: {accuracy:.2f}%")
    
    return accuracy

# This will be used for final paper results
