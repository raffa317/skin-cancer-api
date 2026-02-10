import os
import pandas as pd
import shutil
import glob

def sort_pad_ufes_data():
    project_root = "."
    target_base = "data/pad_ufes"
    
    # Class mapping
    class_map = {
        "BCC": "BCC",
        "SCC": "SCC",
        "ACK": "ACK",
        "NEV": "NEV",
        "SEK": "SEK",
        "MEL": "MEL"
    }
    
    # Source folders (The zip contains parts, and they are nested)
    source_folders = [
        "imgs_part_1/imgs_part_1",
        "imgs_part_2/imgs_part_2",
        "imgs_part_3/imgs_part_3"
    ]
    
    # PATH FIX: We know where it is now
    project_root = os.getcwd()
    metadata_full_path = os.path.join(project_root, "data", "pad_ufes", "metadata.csv")
    
    if not os.path.exists(metadata_full_path):
        print(f"❌ Critical Error: Metadata not found at {metadata_full_path}")
        return

    print(f"✅ Found Metadata: {metadata_full_path}")
    df = pd.read_csv(metadata_full_path)
    
    # Change working dir to data/pad_ufes for sorting operations
    os.chdir(os.path.join(project_root, "data", "pad_ufes"))
    
    # Check column name
    diag_col = 'diagnostic' if 'diagnostic' in df.columns else 'dx'
    id_col = 'img_id' if 'img_id' in df.columns else 'image_id'
    
    # Sort logic
    # We are already in data/pad_ufes thanks to os.chdir
    move_count = 0
    total = len(df)
    
    # Ensure source folders are correct relative to current dir
    valid_source_folders = [f for f in source_folders if os.path.exists(f)]
    print(f"📂 Found source folders: {valid_source_folders}")

    for index, row in df.iterrows():
        img_id = row[id_col]
        diag = row[diag_col]
        
        # Verify valid diagnosis
        if diag in class_map:
            target_class = class_map[diag]
            
            # Find the file in one of the source folders
            src_path = None
            for folder in valid_source_folders:
                 # Try typical extensions
                 potential_path = os.path.join(folder, img_id)
                 if os.path.exists(potential_path):
                     src_path = potential_path
                     break
                 elif os.path.exists(potential_path + ".png"):
                     src_path = potential_path + ".png"
                     break
                 elif os.path.exists(potential_path + ".jpg"):
                     src_path = potential_path + ".jpg"
                     break
            
            if src_path:
                dst_path = os.path.join(target_class, os.path.basename(src_path))
                if not os.path.exists(dst_path): # Don't overwrite if already there
                    try:
                        shutil.move(src_path, dst_path)
                        move_count += 1
                    except Exception as e:
                        print(f"Error moving {src_path}: {e}")
        
        if index % 100 == 0:
            print(f"Processed {index}/{total}...", end='\r')
            
    print(f"\n🎉 Finished! Sorted {move_count} images.")
                
    # Job done
        
if __name__ == "__main__":
    sort_pad_ufes_data()
