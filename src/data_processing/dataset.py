import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import glob

class SkinCancerDataset(Dataset):
    def __init__(self, csv_file, root_dir, phase2_dir=None, pad_dir=None, synthetic_dir=None, transform=None, use_augmentation=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations (Phase 1).
            root_dir (string): Directory with all the images (Phase 1).
            phase2_dir (string): Directory with Phase 2 data folders (Acne, Eczema, etc.).
            pad_dir (string): Directory with PAD-UFES-20 data (Dark skin cancer images).
            synthetic_dir (string): Directory with synthetic data folders (Phase 3).
            transform (callable, optional): Optional transform to be applied.
        """
        self.transform = transform
        self.use_augmentation = use_augmentation
        self.samples = [] # List of (image_path, label_index)
        
        # 1. Define All Classes (7 Original + New Ones)
        self.classes = [
            'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc', # 0-6 (HAM10000)
            'acne', 'eczema', 'normal', 'tinea'               # 7-10 (Phase 2)
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 2. Load Phase 1 Data (HAM10000) from CSV
        if os.path.exists(csv_file):
            metadata = pd.read_csv(csv_file)
            for _, row in metadata.iterrows():
                img_name = row['image_id'] + ".jpg"
                img_path = os.path.join(root_dir, img_name)
                label_str = row['dx']
                if label_str in self.class_to_idx:
                    self.samples.append((img_path, self.class_to_idx[label_str]))
        
        # 3. Load Phase 2 Data (from Folders)
        if phase2_dir and os.path.exists(phase2_dir):
            # Map folder names to class names
            # Folder: 'Acne' -> Class: 'acne'
            # Folder: 'Eczema' -> Class: 'eczema'
            # Folder: 'Normal' -> Class: 'normal'
            # Folder: 'Tinea Ringworm' -> Class: 'tinea'
            folder_map = {
                'Acne': 'acne',
                'Eczema': 'eczema',
                'Normal': 'normal',
                'Tinea Ringworm': 'tinea'
            }
            
            for folder_name, class_key in folder_map.items():
                folder_path = os.path.join(phase2_dir, folder_name)
                if os.path.exists(folder_path):
                    # Find all images
                    images = glob.glob(os.path.join(folder_path, "*.*"))
                    label_idx = self.class_to_idx[class_key]
                    
                    for img_path in images:
                        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((img_path, label_idx))

        # 4. Load PAD-UFES-20 Data (Dark Skin Cancer)
        if pad_dir and os.path.exists(pad_dir):
            # Map PAD-UFES folders to HAM10000 classes
            folder_map = {
                'ACK': 'akiec',  # Actinic Keratoses
                'BCC': 'bcc',    # Basal Cell Carcinoma
                'MEL': 'mel',    # Melanoma
                'NEV': 'nv',     # Melanocytic Nevi
                'SEK': 'bkl'     # Seborrheic Keratoses
                # SCC is skipped (not in HAM10000)
            }
            
            for folder_name, class_key in folder_map.items():
                folder_path = os.path.join(pad_dir, folder_name)
                if os.path.exists(folder_path):
                    images = glob.glob(os.path.join(folder_path, "*.png"))
                    label_idx = self.class_to_idx[class_key]
                    
                    for img_path in images:
                        self.samples.append((img_path, label_idx))

        # 5. Load Phase 3 Data (Synthetic Realism)
        if synthetic_dir and os.path.exists(synthetic_dir):
            # For now, we inject synthetic images into the 'mel' (melanoma) class
            mel_dir = os.path.join(synthetic_dir, "mel")
            if os.path.exists(mel_dir):
                images = glob.glob(os.path.join(mel_dir, "*.png"))
                label_idx = self.class_to_idx['mel']
                print(f"💉 Injecting {len(images)} synthetic melanoma samples...")
                for img_path in images:
                    self.samples.append((img_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # If image fails, return a blank black image to avoid crashing training
            # (Better than crashing 10 hours in)
            print(f"Warning: Corrupt image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)

        return image, label

