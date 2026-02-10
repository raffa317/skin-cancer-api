import os
import shutil

def sort_data():
    project_root = "."
    target_base = "data/phase2"
    
    # We look for these keywords in folder names
    keywords = {
        "Acne": "Acne",
        "Eczema": "Eczema",
        "Tinea": "Tinea Ringworm",
        "Ringworm": "Tinea Ringworm"
    }
    
    # Folders to ignore
    ignore = ["data", "src", ".venv", ".git", ".gemini", "__pycache__"]
    
    print("🤖 Scanning for new data folders...")
    
    found_count = 0
    
    # Walk through project root
    for item in os.listdir(project_root):
        if item in ignore or not os.path.isdir(item):
            continue
            
        # This is a potential candidate folder (e.g. "archive", "train", "test")
        print(f"Checking folder: {item}...")
        
        for root, dirs, files in os.walk(item):
            folder_name = os.path.basename(root)
            
            # Check if this folder matches our keywords
            destination = None
            for key, target_folder in keywords.items():
                if key.lower() in folder_name.lower():
                    destination = os.path.join(target_base, target_folder)
                    break
            
            if destination:
                print(f"  Found matching folder: {folder_name} -> Moving to {destination}")
                os.makedirs(destination, exist_ok=True)
                
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        src_path = os.path.join(root, file)
                        dst_path = os.path.join(destination, file)
                        try:
                            shutil.move(src_path, dst_path)
                            found_count += 1
                        except Exception as e:
                            print(f"Error moving {file}: {e}")
                            
    if found_count > 0:
        print(f"\n✅ Success! Moved {found_count} images to Phase 2 folders.")
        print("You can now delete the original downloaded folder to save space.")
    else:
        print("\n❌ No matching images found yet.")
        print("Did you drag the folder into this directory?")
        print(f"Current Directory: {os.getcwd()}")

if __name__ == "__main__":
    sort_data()
