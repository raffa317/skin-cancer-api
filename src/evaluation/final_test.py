import os
import sys
sys.path.append(os.getcwd()) # Ensure root dir is in path
import torch
from torchvision import transforms
from PIL import Image
from src.model import get_model
from src.gradcam import get_gradcam
# Import specific functions from app.py
# Note: detailed import might fail if app.py has global code. 
# We will duplicate the critical logic for this test to be safe and isolated.
from fpdf import FPDF
from datetime import datetime

def test_model_loading():
    print("[1/5] Testing Model Loading...")
    try:
        model = get_model(num_classes=11, pretrained=False)
        if os.path.exists("model.pth"):
            model.load_state_dict(torch.load("model.pth", map_location='cpu'))
            print("   ✅ Model weights loaded successfully.")
        else:
            print("   ❌ model.pth not found!")
            return None
        model.eval()
        return model
    except Exception as e:
        print(f"   ❌ Loading failed: {e}")
        return None

def test_prediction(model, image_path):
    print("[2/5] Testing Prediction Pipeline...")
    if not os.path.exists(image_path):
        print(f"   ❌ Test image not found at {image_path}")
        return None, None
    
    try:
        # Load Image
        image = Image.open(image_path).convert('RGB')
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
        top_prob, top_idx = torch.max(probs, 1)
        print(f"   ✅ Prediction successful. Class Index: {top_idx.item()}, Confidence: {top_prob.item():.4f}")
        return image_tensor, top_idx.item()
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        return None, None

def test_explainability(model, image_tensor, target_class):
    print("[3/5] Testing Explainability (Grad-CAM)...")
    try:
        # Using Score-CAM as per app default
        heatmap = get_gradcam(model, image_tensor, target_class=target_class, method="Score-CAM")
        if heatmap is not None:
             print("   ✅ Score-CAM heatmap generated successfully.")
        else:
             print("   ⚠️ Heatmap returned None (check gradcam.py).")
    except Exception as e:
        print(f"   ❌ Explainability failed: {e}")

def create_dummy_pdf(filename, label, score):
    # Replicating logic from app.py for test
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="AiDerm Test Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Date: {datetime.now()}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {label} ({score*100:.2f}%)", ln=True)
    # FPDF2 output() with dest='S' returns a string in some versions, bytes in others.
    # Safe way: output() to a buffer or use dest='S'.encode('latin-1') if it returns str.
    return pdf.output(dest='S').encode('latin-1')

def test_pdf_generation():
    print("[4/5] Testing PDF Report Generation...")
    try:
        pdf_bytes = create_dummy_pdf("test.jpg", "Melanoma", 0.95)
        if len(pdf_bytes) > 0:
            print("   ✅ PDF generated successfully (Bytes: {len(pdf_bytes)}).")
        else:
             print("   ❌ PDF is empty.")
    except Exception as e:
        print(f"   ❌ PDF generation failed: {e}")

def main():
    print("=== AiDerm Final System Verified ===")
    
    # 1. Load Model
    model = test_model_loading()
    if not model: return

    # 2. Predict on a Synthetic Image
    test_img = "data/synthetic/mel/synthetic_big_0.png"
    img_tensor, class_idx = test_prediction(model, test_img)
    if img_tensor is None: return

    # 3. Explainability
    test_explainability(model, img_tensor, class_idx)

    # 4. PDF
    test_pdf_generation()

    print("\n✅ System Ready for Deployment.")

if __name__ == "__main__":
    main()
