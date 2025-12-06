import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from src.model import get_model
from src.gradcam import get_gradcam
import os
import numpy as np
import src.database as db
import pandas as pd

# Initialize DB on startup
db.init_db()

# Session State Initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None

# Load model
@st.cache_resource
def load_model():
    model = get_model(num_classes=7, pretrained=False)
    if os.path.exists("model.pth"):
        model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    else:
        st.warning("Model weights not found. Please train the model first.")
    model.eval()
    return model

def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities[0], image_tensor

def login_view():
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user_id = db.verify_user(username, password)
        if user_id:
            st.session_state.logged_in = True
            st.session_state.user_id = user_id
            st.session_state.username = username
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid username or password")

def register_view():
    st.header("Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Sign Up"):
        if username and password:
            if db.create_user(username, password):
                st.success("Account created! Please log in.")
            else:
                st.error("Username already exists.")
        else:
            st.error("Please fill in all fields.")

def scan_view():
    st.title("Skin Cancer Detection")
    st.write("Upload an image or use the camera to classify a skin lesion.")
    
    input_method = st.radio("Select Input Method:", ("Upload Image", "Camera"))
    image = None
    filename = "Camera Capture"
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            filename = uploaded_file.name
    elif input_method == "Camera":
        camera_file = st.camera_input("Take a picture")
        if camera_file is not None:
            image = Image.open(camera_file).convert('RGB')
    
    if image is not None:
        st.image(image, caption='Input Image', use_column_width=True)
        st.write("Classifying...")
        
        model = load_model()
        probabilities, image_tensor = predict(image, model)
        
        classes = ['Actinic keratoses (akiec)', 'Basal cell carcinoma (bcc)', 
                   'Benign keratosis-like lesions (bkl)', 'Dermatofibroma (df)', 
                   'Melanoma (mel)', 'Melanocytic nevi (nv)', 'Vascular lesions (vasc)']
        
        top_class_idx = torch.argmax(probabilities).item()
        top_label = classes[top_class_idx]
        top_score = probabilities[top_class_idx].item()
        
        st.success(f"Prediction: **{top_label}**")
        st.write(f"Confidence: {top_score*100:.2f}%")
        st.progress(top_score)
        
        # Save to History
        if st.session_state.logged_in:
            db.log_scan(st.session_state.user_id, filename, top_label, float(top_score))
            st.caption("✅ Result saved to history.")
        
        dangerous_classes = ['Actinic keratoses (akiec)', 'Basal cell carcinoma (bcc)', 'Melanoma (mel)']
        if top_label in dangerous_classes:
            st.error("⚠️ **High Risk Alert**: Consult a dermatologist immediately.")
        else:
            st.info("ℹ️ **Note**: Monitor for changes. Consult a doctor if concerned.")
            
        st.subheader("Explainability (Grad-CAM):")
        try:
            cam_image = get_gradcam(model, image_tensor, target_class=top_class_idx)
            st.image(cam_image, caption='Grad-CAM Heatmap', use_column_width=True)
        except Exception as e:
            st.error(f"Could not generate Grad-CAM: {e}")

def history_view():
    st.title("My Scan History")
    history = db.get_user_history(st.session_state.user_id)
    
    if not history:
        st.info("No scan history found.")
        return

    # Convert to DataFrame for nicer display
    df = pd.DataFrame(history, columns=['Filename', 'Prediction', 'Confidence', 'Timestamp'])
    df['Confidence'] = df['Confidence'].apply(lambda x: f"{x*100:.1f}%")
    st.dataframe(df, use_container_width=True)

def main():
    st.sidebar.title("Skin Cancer AI")
    
    if st.session_state.logged_in:
        st.sidebar.write(f"Welcome, **{st.session_state.username}**!")
        page = st.sidebar.radio("Navigation", ["Scan", "History"])
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()
        
        if page == "Scan":
            scan_view()
        elif page == "History":
            history_view()
            
    else:
        st.sidebar.info("Please Login to save your scans.")
        page = st.sidebar.radio("Navigation", ["Login", "Register"])
        
        if page == "Login":
            login_view()
        elif page == "Register":
            register_view()

if __name__ == "__main__":
    main()
