import os

def setup_phase2_dirs():
    # Base directory for Phase 2 data
    base_dir = "data/phase2"
    
    # We will organize by class names
    classes = [
        "Acne", 
        "Eczema", 
        "Normal", 
        "Tinea Ringworm" # Adding a common fungal one too if user wants, or just stick to requested
    ]
    
    # Create the directories
    for class_name in classes:
        dir_path = os.path.join(base_dir, class_name)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
        
    print("\n✅ Phase 2 Folders Created!")
    print(f"Please drop your downloaded images into the respective folders in '{os.path.abspath(base_dir)}'")

if __name__ == "__main__":
    setup_phase2_dirs()
