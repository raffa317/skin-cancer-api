"""
Test-Time Augmentation (TTA) for Ensemble
Boosts accuracy by averaging predictions from multiple augmented versions
"""
import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from src.model import get_model
from src.dataset import SkinCancerDataset
from src.augmentation import HeavyAugmentation
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"

# TTA Transforms - 8 augmentations
TTA_TRANSFORMS = [
    transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  # Original
    transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  # H-Flip
    transforms.Compose([transforms.Resize((224, 224)), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  # V-Flip
    transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation((90, 90)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  # Rotate 90
    transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation((180, 180)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  # Rotate 180
    transforms.Compose([transforms.Resize((224, 224)), transforms.RandomRotation((270, 270)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  # Rotate 270
    transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  # Center crop
    transforms.Compose([transforms.Resize((224, 224)), transforms.ColorJitter(brightness=0.1), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  # Brightness
]

def load_models():
    """Load all ensemble models"""
    models = {}
    architectures = ['resnet50', 'efficientnet', 'densenet']
    
    for arch in architectures:
        model_path = f"models/ensemble_{arch}.pth"
        if os.path.exists(model_path):
            model = get_model(num_classes=11, pretrained=False, arch=arch)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            models[arch] = model
            print(f"  ✅ Loaded {arch}")
        else:
            print(f"  ❌ Not found: {model_path}")
    
    return models

def predict_with_tta(models, image_path, num_classes=11):
    """Get prediction using TTA across all models"""
    from PIL import Image
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    all_probs = []
    
    # For each model in ensemble
    for model in models.values():
        model_probs = []
        
        # For each TTA transform
        for transform in TTA_TRANSFORMS:
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                model_probs.append(probs.cpu().numpy())
        
        # Average across TTA transforms for this model
        avg_model_probs = np.mean(model_probs, axis=0)
        all_probs.append(avg_model_probs)
    
    # Average across all models
    final_probs = np.mean(all_probs, axis=0)
    prediction = np.argmax(final_probs, axis=1)[0]
    
    return prediction, final_probs[0]

def evaluate_with_tta():
    """Evaluate ensemble with TTA"""
    print("🎯 TEST-TIME AUGMENTATION EVALUATION")
    print(f"Device: {DEVICE}")
    print(f"TTA transforms: {len(TTA_TRANSFORMS)}")
    print("-" * 60)
    
    # Load models
    print("\n📂 Loading ensemble models...")
    models = load_models()
    
    if len(models) == 0:
        print("❌ No models found!")
        return
    
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
    
    # Get validation indices (same split as training)
    import random
    random.seed(42)
    indices = list(range(len(dataset.samples)))
    random.shuffle(indices)
    split_point = int(0.85 * len(indices))
    val_indices = indices[split_point:]
    
    print(f"Validation samples: {len(val_indices)}")
    
    # Evaluate with TTA
    print("\n🔮 Running TTA predictions...")
    all_preds = []
    all_labels = []
    
    for i, idx in enumerate(tqdm(val_indices)):
        img_path, label = dataset.samples[idx]
        pred, _ = predict_with_tta(models, img_path)
        all_preds.append(pred)
        all_labels.append(label)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    # Results
    print("\n" + "=" * 60)
    print("🏆 TTA RESULTS")
    print("=" * 60)
    print(f"\n✨ Ensemble + TTA Accuracy: {accuracy:.2f}%")
    print(f"   Previous (no TTA):      88.53%")
    print(f"   Improvement:            +{accuracy - 88.53:.2f}%")
    
    if accuracy >= 90.0:
        print(f"\n🎉🎉🎉 TARGET ACHIEVED! 90%+ REACHED! 🎉🎉🎉")
    else:
        print(f"\n📊 Gap to 90%: {90.0 - accuracy:.2f}%")
    
    # Save report
    class_names = list(dataset.class_to_idx.keys())
    unique_labels = sorted(set(all_labels))
    filtered_names = [class_names[i] for i in unique_labels]
    report = classification_report(all_labels, all_preds, labels=unique_labels, target_names=filtered_names, zero_division=0)
    
    os.makedirs("tta_results", exist_ok=True)
    with open("tta_results/tta_report.txt", "w") as f:
        f.write(f"TTA Accuracy: {accuracy:.2f}%\n\n")
        f.write(report)
    
    print(f"\n💾 Results saved to: tta_results/")
    
    return accuracy

if __name__ == "__main__":
    evaluate_with_tta()
