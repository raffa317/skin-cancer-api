import cv2
import numpy as np
import os
from pathlib import Path

def calculate_ita(image_path):
    """
    Calculate the Individual Typology Angle (ITA) to estimate skin tone.
    ITA = arctan((L* - 50) / b*) * (180 / pi)
    
    ITA > 55: Very Light
    41-55: Light
    28-41: Intermediate
    10-28: Tan
    -30-10: Brown
    < -30: Dark
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
        # Convert to Lab color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        
        # We assume the center of the image (the lesion) might be skewed, 
        # so we look at the healthy skin around the lesion.
        # However, for synthetic images, the whole image represents the skin type.
        # We'll take the median L and b values.
        L = lab[:, :, 0].astype(np.float32) * (100.0 / 255.0) # L* is 0-100
        b = lab[:, :, 2].astype(np.float32) - 128.0 # b* is roughly -128 to 127
        
        # Calculate ITA for each pixel
        ita_map = np.arctan2(L - 50, b) * (180.0 / np.pi)
        
        # Median ITA of the image
        median_ita = np.median(ita_map)
        return median_ita
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def get_skin_category(ita):
    if ita is None: return "Unknown"
    if ita > 28: return "Light" # Fitzpatrick I-III
    else: return "Dark" # Fitzpatrick IV-VI

def process_directory(directory):
    print(f"🔍 Classifying skin tones in: {directory}")
    results = {"Light": 0, "Dark": 0, "Unknown": 0}
    
    path = Path(directory)
    if not path.exists():
        print("❌ Directory not found.")
        return
        
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for img_file in path.glob(ext):
            ita = calculate_ita(img_file)
            category = get_skin_category(ita)
            results[category] += 1
        # print(f"  - {img_file.name}: ITA={ita:.2f} ({category})")
        
    print("\n📊 Skin Tone Distribution Results:")
    for cat, count in results.items():
        print(f"  - {cat}: {count}")
    return results

if __name__ == "__main__":
    # Test on synthetic data
    process_directory("data/synthetic")
