"""
ISIC Dataset Loader
Loads ISIC 2019 dataset with proper class mapping
"""
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ISICDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None):
        """
        Args:
            csv_file: Path to ISIC_2019_Training_GroundTruth.csv
            images_dir: Directory with ISIC images
            transform: Optional transform
        """
        self.df = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform
        
        # ISIC 2019 has 8 diagnostic categories
        # Map to HAM10000 classes where possible
        self.class_mapping = {
            'MEL': 'mel',      # Melanoma
            'NV': 'nv',        # Melanocytic nevus
            'BCC': 'bcc',      # Basal cell carcinoma
            'AK': 'akiec',     # Actinic keratosis (maps to akiec)
            'BKL': 'bkl',      # Benign keratosis
            'DF': 'df',        # Dermatofibroma
            'VASC': 'vasc',    # Vascular lesion
            'SCC': 'bcc',      # Squamous cell carcinoma (map to bcc - similar)
        }
        
        # HAM10000 class indices
        self.class_to_idx = {
            'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3,
            'mel': 4, 'nv': 5, 'vasc': 6
        }
        
        # Build samples list
        self.samples = []
        for _, row in self.df.iterrows():
            image_name = row['image'] + '.jpg'
            image_path = os.path.join(self.images_dir, 'ISIC_2019_Training_Input', image_name)
            
            # Find which class is 1.0
            for col in self.class_mapping.keys():
                if col in row and row[col] == 1.0:
                    ham_class = self.class_mapping[col]
                    if ham_class in self.class_to_idx:
                        label = self.class_to_idx[ham_class]
                        self.samples.append((image_path, label))
                    break
        
        print(f"✅ Loaded {len(self.samples)} ISIC images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
