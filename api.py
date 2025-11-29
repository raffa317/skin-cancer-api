from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
from src.model import get_model
import io
import os

app = FastAPI(title="Skin Cancer Detection API")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=7, pretrained=False)

if os.path.exists("model.pth"):
    model.load_state_dict(torch.load("model.pth", map_location=device))
    print("Model loaded successfully.")
else:
    print("Warning: model.pth not found.")

model.to(device)
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classes = ['Actinic keratoses (akiec)', 'Basal cell carcinoma (bcc)', 
           'Benign keratosis-like lesions (bkl)', 'Dermatofibroma (df)', 
           'Melanoma (mel)', 'Melanocytic nevi (nv)', 'Vascular lesions (vasc)']

@app.get("/")
def read_root():
    return {"message": "Skin Cancer Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        # Get results
        probs = probabilities[0].cpu().numpy().tolist()
        
        # Top prediction
        top_idx = torch.argmax(probabilities).item()
        top_label = classes[top_idx]
        top_score = probs[top_idx]
        
        return JSONResponse(content={
            "prediction": top_label,
            "confidence": top_score,
            "all_probabilities": {classes[i]: probs[i] for i in range(len(classes))}
        })
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
