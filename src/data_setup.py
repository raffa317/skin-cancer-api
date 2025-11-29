import kagglehub
import os
import shutil
import pandas as pd

def download_data():
    print("Downloading HAM10000 dataset...")
    # This downloads to a cache directory
    path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    print(f"Dataset downloaded to: {path}")

    # Define target directory
    target_dir = os.path.join(os.getcwd(), "data")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # The dataset might have subfolders like 'ham10000_images_part_1' etc.
    # We want to move all images to a single 'images' folder in 'data/'
    images_dir = os.path.join(target_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    print(f"Organizing files into {target_dir}...")
    
    # Walk through the downloaded path
    for root, dirs, files in os.walk(path):
        for file in files:
            src_path = os.path.join(root, file)
            
            if file.endswith(".csv"):
                # Move csv to data root
                dst_path = os.path.join(target_dir, file)
                shutil.copy2(src_path, dst_path)
                print(f"Copied metadata: {file}")
            
            elif file.endswith(".jpg"):
                # Move images to images folder
                dst_path = os.path.join(images_dir, file)
                shutil.copy2(src_path, dst_path)
                # print(f"Copied image: {file}") # Too verbose

    print("Data setup complete.")
    
    # Verify
    metadata_path = os.path.join(target_dir, "HAM10000_metadata.csv")
    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        print(f"Metadata loaded. Shape: {df.shape}")
        print(f"Total images found: {len(os.listdir(images_dir))}")
    else:
        print("Error: Metadata file not found.")

if __name__ == "__main__":
    download_data()
