import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from src.model import get_model
from src.gradcam import get_gradcam
import os
import numpy as np

# Load model
@st.cache_resource
def load_model():
    model = get_model(num_classes=7, pretrained=False) # Pretrained=False because we load weights
    # Check if model weights exist
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    else:
        st.warning("Model weights not found. Please train the model first.")
    model.eval()
    return model

def predict(image, model):
    # Define transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0) # Add batch dimension
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    return probabilities[0], image_tensor

def main():
    st.title("Skin Cancer Detection AI")
    st.write("Upload an image or use the camera to classify a skin lesion.")
    
    # Input method selection
    input_method = st.radio("Select Input Method:", ("Upload Image", "Camera"))
    
    image = None
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            
    elif input_method == "Camera":
        camera_file = st.camera_input("Take a picture")
        if camera_file is not None:
            image = Image.open(camera_file).convert('RGB')
    
    if image is not None:
        st.image(image, caption='Input Image', use_column_width=True)
        
        st.write("Classifying...")
        model = load_model()
        probabilities, image_tensor = predict(image, model)
        
        # Classes
        classes = ['Actinic keratoses (akiec)', 'Basal cell carcinoma (bcc)', 
                   'Benign keratosis-like lesions (bkl)', 'Dermatofibroma (df)', 
                   'Melanoma (mel)', 'Melanocytic nevi (nv)', 'Vascular lesions (vasc)']
        
        # Display results
        st.subheader("Prediction Results:")
        
        # Create a dictionary for results
        results = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
        
        # Get top prediction
        top_class_idx = torch.argmax(probabilities).item()
        top_label = classes[top_class_idx]
        top_score = probabilities[top_class_idx].item()
        
        st.success(f"Prediction: **{top_label}**")
        st.write(f"Confidence: {top_score*100:.2f}%")
        st.progress(top_score)
        
        # Safety Warning
        dangerous_classes = ['Actinic keratoses (akiec)', 'Basal cell carcinoma (bcc)', 'Melanoma (mel)']
        if top_label in dangerous_classes:
            st.error("⚠️ **High Risk Alert**: This lesion is classified as potentially cancerous or pre-cancerous. Please consult a dermatologist immediately for a professional examination.")
        else:
            st.info("ℹ️ **Note**: Even for benign predictions, it is always good practice to monitor skin lesions for changes. Consult a doctor if you are concerned.")
        
        # Grad-CAM
        st.subheader("Explainability (Grad-CAM):")
        st.write("This heatmap shows where the model is looking to make its decision.")
        
        try:
            cam_image = get_gradcam(model, image_tensor, target_class=top_class_idx)
            st.image(cam_image, caption='Grad-CAM Heatmap', use_column_width=True)
        except Exception as e:
            st.error(f"Could not generate Grad-CAM: {e}")

if __name__ == "__main__":
    main()
