"""
Heavy Data Augmentation for Medical Images
Implements aggressive augmentation to multiply effective dataset size
"""
import torch
from torchvision import transforms
import random

class HeavyAugmentation:
    """
    Aggressive augmentation pipeline for medical images
    Multiplies effective dataset by 5-10x
    """
    def __init__(self, image_size=224, train=True):
        if train:
            self.transform = transforms.Compose([
                # Geometric augmentations
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),
                
                # Color augmentations (important for lesion detection)
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1
                ),
                
                # Convert to tensor
                transforms.ToTensor(),
                
                # Normalization
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                
                # Random erasing (simulates occlusion)
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
            ])
        else:
            # Validation: only resize and normalize
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
    
    def __call__(self, image):
        return self.transform(image)


class MixupAugmentation:
    """
    Mixup: Mix two images together
    Paper: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)
    Typically gives +1-2% accuracy boost
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, images, labels):
        """
        Mix a batch of images
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)
        
        Returns:
            mixed_images: Mixed images
            labels_a, labels_b, lam: For computing mixed loss
        """
        batch_size = images.size(0)
        
        # Sample lambda from beta distribution
        if self.alpha > 0:
            lam = random.betavariate(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Random permutation
        index = torch.randperm(batch_size).to(images.device)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index]
        
        labels_a = labels
        labels_b = labels[index]
        
        return mixed_images, labels_a, labels_b, lam


def get_augmentation_transform(train=True, heavy=True):
    """
    Get augmentation transform
    
    Args:
        train: If True, use training augmentation
        heavy: If True, use heavy augmentation (more aggressive)
    
    Returns:
        Transform function
    """
    if heavy:
        return HeavyAugmentation(train=train)
    else:
        # Light augmentation (original)
        if train:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
