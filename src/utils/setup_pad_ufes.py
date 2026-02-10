import os

def setup_pad_ufes():
    base_dir = "data/pad_ufes"
    
    # PAD-UFES-20 Classes
    # We will map these to our model's classes or create new ones
    classes = [
        "BCC",    # Basal Cell Carcinoma
        "SCC",    # Squamous Cell Carcinoma (New!)
        "ACK",    # Actinic Keratosis (Maps to akiec)
        "NEV",    # Nevus (Maps to nv)
        "SEK",    # Seborrheic Keratosis (Maps to bkl)
        "MEL"     # Melanoma (Maps to mel)
    ]
    
    print(f"📁 Creating folder structure in {base_dir}...")
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    for cls in classes:
        path = os.path.join(base_dir, cls)
        try:
            os.makedirs(path, exist_ok=True)
            print(f"  - Created: {path}")
        except Exception as e:
            print(f"  - Error creating {path}: {e}")
            
    print("\n✅ Setup complete! ready for images.")
    print("Download the dataset and unzip it.")
    print("Then run 'src/sort_pad_data.py' (I will write this next) to verify.")

if __name__ == "__main__":
    setup_pad_ufes()
