import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class SkinCancerDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Mapping classes to integers
        self.classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.metadata.iloc[idx, 1] + ".jpg")
        
        # Some images might be missing if download failed, handle gracefully?
        # For now assuming all exist.
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
             # Fallback or skip - for now let it error to debug
             raise FileNotFoundError(f"Image not found: {img_name}")
        
        label_str = self.metadata.iloc[idx, 2] # 'dx' column
        label = self.class_to_idx[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label
