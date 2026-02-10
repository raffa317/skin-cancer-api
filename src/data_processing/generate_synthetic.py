import torch
from diffusers import StableDiffusionPipeline
import os
from pathlib import Path

# Configuration
# Using a lightweight model variant if available, or standard 1.5
MODEL_ID = "runwayml/stable-diffusion-v1-5" 
OUTPUT_DIR = "data/synthetic"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Now uses GPU!

def setup_pipeline():
    print(f"🚀 Loading Stable Diffusion Pipeline ({MODEL_ID})...")
    
    # Load pipeline
    if DEVICE == "cuda":
        # fp16 is much faster on GPU and compatible with RTX 4050
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.float16,
            variant="fp16"
        )
        pipe = pipe.to("cuda")
        print("✅ GPU Acceleration Enabled (FP16)")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        print("ℹ️ Running on CPU (Float32)")
        
    # Disable safety checker for medical images (sometimes lesions trigger it)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    
    return pipe

import random

def generate_lesions(num_images=500):
    pipe = setup_pipeline()
    
    # Target class subfolder for ImageFolder compatibility
    class_dir = os.path.join(OUTPUT_DIR, "mel")
    Path(class_dir).mkdir(parents=True, exist_ok=True)
    
    # Diversified prompts for high-quality variation
    prompts = [
        "dermoscopy photo of melanoma skin lesion on dark brown skin, high quality, medical image, fitzpatrick skin type V",
        "malignant melanoma on black skin, dermoscopic view, irregular borders, high detail, fitzpatrick VI",
        "asymmetric skin lesion on dark skin, melanoma, medical photography, 4k, fitzpatrick V",
        "melanoma dermoscopy on deep brown skin, chaotic pigment network, fitzpatrick skin type VI",
        "photo of a cluster of pigment cells on dark skin, melanoma, clinical dermatology, fitzpatrick V",
        "macro photography of nodular melanoma on dark pigmented skin, medical grade, fitzpatrick scale VI",
        "superficial spreading melanoma on brown skin, irregular shape, uneven color, dermatical macro photo",
        "acral lentiginous melanoma on dark skin complex, clinical closeup, high resolution, fitzpatrick type V",
        "suspicious beauty mark turning into melanoma, dark skin tone, medical reference image, 8k details",
        "clinical zoom of pigmented lesion features on black skin, signs of malignancy, fitzpatrick VI, clear focus"
    ]
    
    print(f"🎨 Starting massive generation of {num_images} images...")
    
    for i in range(num_images):
        # Unique filename using index
        filename = f"synthetic_big_{i}.png"
        save_path = os.path.join(class_dir, filename)
        
        if os.path.exists(save_path):
            continue
            
        prompt = random.choice(prompts)
        print(f"  - Generating image {i+1}/{num_images} using prompt: '{prompt[:40]}...'")
        
        # Consistent settings for scientific validity
        image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
        image.save(save_path)

if __name__ == "__main__":
    generate_lesions(1000)
    print("\n🎉 Massive Generation Complete! Check data/synthetic folder.")
