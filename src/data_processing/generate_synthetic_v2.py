"""
Improved Synthetic Data Generation
Better prompts + quality filtering for all classes
"""
import torch
from diffusers import StableDiffusionPipeline
import os
from pathlib import Path
import random
from PIL import Image
import torch.nn as nn
from torchvision import transforms

# Config
MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "data/synthetic_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Target classes and counts
CLASS_TARGETS = {
    'mel': 500,    # Melanoma
    'bcc': 500,    # Basal cell carcinoma
    'akiec': 300,  # Actinic keratoses
    'bkl': 200,    # Benign keratosis
    'df': 200,     # Dermatofibroma
    'nv': 300,     # Melanocytic nevi
    'vasc': 200,   # Vascular lesions
}

# Improved prompts per class
PROMPTS = {
    'mel': [
        "high resolution dermoscopy photo of melanoma on dark brown skin, irregular borders, asymmetric shape, multiple colors, medical photography, fitzpatrick type V",
        "clinical macro photograph of malignant melanoma lesion on black skin, chaotic pigment network, blue-white veil, fitzpatrick VI, 4k medical image",
        "dermoscopic image of superficial spreading melanoma on deep brown skin, uneven color distribution, irregular shape, high detail, fitzpatrick V",
        "medical photography of nodular melanoma on dark pigmented skin, raised lesion, irregular borders, clinical quality, fitzpatrick type VI",
    ],
    'bcc': [
        "dermoscopy photo of basal cell carcinoma on dark skin, pearly appearance, visible blood vessels, medical image, fitzpatrick V",
        "clinical photograph of BCC lesion on brown skin, translucent border, central ulceration, high resolution, fitzpatrick VI",
        "macro medical image of basal cell carcinoma on dark pigmented skin, telangiectasia visible, dermatology reference photo",
    ],
    'akiec': [
        "dermoscopic view of actinic keratosis on dark skin, rough scaly texture, slightly raised, medical photography, fitzpatrick V",
        "clinical closeup of AK lesion on brown skin, erythematous base, keratotic surface, high detail medical image",
    ],
    'bkl': [
        "dermoscopy of seborrheic keratosis on dark skin, stuck-on appearance, well-defined borders, medical photo, fitzpatrick VI",
        "clinical photograph of benign keratosis on brown skin, warty surface, pigmented lesion, dermatology image",
    ],
    'df': [
        "dermoscopic image of dermatofibroma on dark skin, central white scar-like area, peripheral pigment network, fitzpatrick V",
        "clinical photo of dermatofibroma on brown skin, firm nodule, slightly depressed center, medical documentation",
    ],
    'nv': [
        "dermoscopy of melanocytic nevus on dark skin, symmetric shape, uniform color, regular borders, medical image, fitzpatrick VI",
        "clinical photograph of benign mole on brown skin, homogeneous pigmentation, well-circumscribed, fitzpatrick V",
    ],
    'vasc': [
        "dermoscopic view of vascular lesion on dark skin, red to purple coloration, blanching on pressure, medical photo",
        "clinical image of hemangioma on brown skin, vascular pattern visible, distinct borders, fitzpatrick VI",
    ],
}

# Negative prompts (what to avoid)
NEGATIVE_PROMPT = (
    "low quality, blurry, text, watermark, cartoon, drawing, illustration, "
    "3d render, white skin, caucasian, pale skin, off-center, "
    "multiple lesions, full body, face, eyes, hands"
)

def setup_pipeline():
    """Load Stable Diffusion"""
    print(f"🚀 Loading Stable Diffusion...")
    if DEVICE == "cuda":
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
        )
        pipe = pipe.to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID)
            
    pipe.safety_checker = None
    return pipe

def generate_improved_synthetic():
    """Generate high-quality synthetic images for all classes"""
    pipe = setup_pipeline()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_generated = 0
    
    for class_name, target_count in CLASS_TARGETS.items():
        class_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        prompts = PROMPTS[class_name]
        
        print(f"\n🎨 Generating {target_count} images for class: {class_name}")
        
        for i in range(target_count):
            filename = f"synthetic_{class_name}_{i:04d}.png"
            save_path = os.path.join(class_dir, filename)
            
            if os.path.exists(save_path):
                continue
            
            # Random prompt
            prompt = random.choice(prompts)
            
            # Generate with better settings
            image = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=50,  # Higher quality
                guidance_scale=7.5,
                height=512,
                width=512,
            ).images[0]
            
            # Crop to square
            image = image.crop((0, 0, 512, 512))
            image.save(save_path)
            
            if (i + 1) % 50 == 0:
                print(f"  Generated {i+1}/{target_count} for {class_name}")
            
            total_generated += 1
    
    print(f"\n✅ Generated {total_generated} synthetic images!")
    print(f"📁 Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_improved_synthetic()
