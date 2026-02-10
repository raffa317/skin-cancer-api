"""
ISIC 2019 Dataset Downloader
Downloads melanoma and other skin lesion images from ISIC Archive
"""
import os
import requests
from pathlib import Path
import zipfile
from tqdm import tqdm

# ISIC 2019 Challenge Dataset
ISIC_2019_DOWNLOAD_URL = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
ISIC_2019_LABELS_URL = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"

DATA_DIR = "data/isic_2019"
os.makedirs(DATA_DIR, exist_ok=True)

def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def download_isic_2019():
    """Download ISIC 2019 dataset"""
    print("📦 Downloading ISIC 2019 Dataset...")
    print("⚠️  This is ~9GB and will take 10-30 minutes depending on your internet")
    print("⚠️  Make sure you have at least 20GB free disk space")
    
    # Download images
    images_zip = os.path.join(DATA_DIR, "ISIC_2019_Training_Input.zip")
    if not os.path.exists(images_zip):
        print("\n1/2: Downloading images (9GB)...")
        download_file(ISIC_2019_DOWNLOAD_URL, images_zip)
    else:
        print("✅ Images already downloaded")
    
    # Download labels
    labels_csv = os.path.join(DATA_DIR, "ISIC_2019_Training_GroundTruth.csv")
    if not os.path.exists(labels_csv):
        print("\n2/2: Downloading labels...")
        download_file(ISIC_2019_LABELS_URL, labels_csv)
    else:
        print("✅ Labels already downloaded")
    
    # Extract images
    images_dir = os.path.join(DATA_DIR, "images")
    if not os.path.exists(images_dir):
        print("\n📂 Extracting images (this will take 5-10 minutes)...")
        with zipfile.ZipFile(images_zip, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("✅ Extraction complete")
    else:
        print("✅ Images already extracted")
    
    print("\n✅ ISIC 2019 dataset ready!")
    print(f"📁 Location: {os.path.abspath(DATA_DIR)}")
    
if __name__ == "__main__":
    download_isic_2019()
