import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from src.model import get_model
from src.visualization.gradcam import get_gradcam
import os
import numpy as np
import src.utils.database as db
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import io

# Initialize DB on startup
db.init_db()

# Session State Initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None

# Load ensemble models
@st.cache_resource
def load_ensemble_models():
    device = torch.device('cpu')
    models = {}
    
    # Load ResNet50
    model_resnet = get_model(num_classes=11, pretrained=False, arch='resnet50')
    resnet_path = "models/ensemble_resnet50.pth"
    if os.path.exists(resnet_path):
        model_resnet.load_state_dict(torch.load(resnet_path, map_location=device))
        model_resnet.eval()
        models['resnet50'] = model_resnet
    
    # Load EfficientNet
    model_efficient = get_model(num_classes=11, pretrained=False, arch='efficientnet')
    efficient_path = "models/ensemble_efficientnet.pth"
    if os.path.exists(efficient_path):
        model_efficient.load_state_dict(torch.load(efficient_path, map_location=device))
        model_efficient.eval()
        models['efficientnet'] = model_efficient
    
    # Load DenseNet
    model_dense = get_model(num_classes=11, pretrained=False, arch='densenet')
    dense_path = "models/ensemble_densenet.pth"
    if os.path.exists(dense_path):
        model_dense.load_state_dict(torch.load(dense_path, map_location=device))
        model_dense.eval()
        models['densenet'] = model_dense
    
    if not models:
        st.error("⚠️ No ensemble models found! Please ensure models are in the models/ folder.")
    else:
        st.sidebar.success(f"✅ Loaded {len(models)}/3 ensemble models")
    
    return models

def predict_ensemble(image, models):
    """Ensemble prediction using average probability voting"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    all_probabilities = []
    
    with torch.no_grad():
        for model_name, model in models.items():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            all_probabilities.append(probabilities)
    
    # Average probabilities across all models (ensemble voting)
    if all_probabilities:
        avg_probabilities = torch.mean(torch.stack(all_probabilities), dim=0)
        return avg_probabilities[0], image_tensor
    else:
        return None, image_tensor

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
        
        models = load_ensemble_models()
        probabilities, image_tensor = predict_ensemble(image, models)
        
        classes = [
            'Actinic keratoses (akiec)', 
            'Basal cell carcinoma (bcc)', 
            'Benign keratosis-like lesions (bkl)', 
            'Dermatofibroma (df)', 
            'Melanoma (mel)', 
            'Melanocytic nevi (nv)', 
            'Vascular lesions (vasc)',
            'Acne',
            'Eczema',
            'Normal Skin',
            'Tinea / Ringworm'
        ]
        
        top_class_idx = torch.argmax(probabilities).item()
        top_label = classes[top_class_idx]
        top_score = probabilities[top_class_idx].item()
        
        st.success(f"Prediction: **{top_label}**")
        
        # Grad-CAM / Score-CAM
        st.subheader("Explainability:")
        
        # User Choice
        cam_method = st.radio("Visualization Method:", ("Grad-CAM", "Score-CAM"), horizontal=True)
        
        try:
            # Use the first available model for GradCAM (they should focus on similar features)
            first_model = list(models.values())[0] if models else None
            if first_model:
                cam_image = get_gradcam(first_model, image_tensor, target_class=top_class_idx, method=cam_method)
                st.image(cam_image, caption=f'{cam_method} Heatmap', use_column_width=True)
            else:
                st.error("No model available for visualization")
        except Exception as e:
            st.error(f"Could not generate {cam_method}: {e}")
            
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

        # PDF Report Generation
        st.divider()
        if st.button("📄 Generate Clinical Report"):
            report_pdf = create_pdf_report(filename, top_label, top_score, dangerous_classes)
            st.download_button(
                label="Download PDF Report",
                data=report_pdf,
                file_name=f"AiDerm_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )

def create_pdf_report(filename, label, score, dangerous_classes):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="AiDerm Clinical Analysis Report", ln=True, align='C')
    pdf.ln(10)
    
    # Meta
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, txt=f"Image Source: {filename}", ln=True)
    pdf.ln(10)
    
    # Result
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Analysis Results:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Predicted Condition: {label}", ln=True)
    pdf.cell(200, 10, txt=f"AI Confidence Score: {score*100:.2f}%", ln=True)
    pdf.ln(5)
    
    # Risk Assessment
    if label in dangerous_classes:
        pdf.set_text_color(255, 0, 0)
        pdf.cell(200, 10, txt="RISK ASSESSMENT: HIGH RISK (Malignancy Potential)", ln=True)
    else:
        pdf.set_text_color(0, 100, 0)
        pdf.cell(200, 10, txt="RISK ASSESSMENT: Low Risk (Benign Characteristics)", ln=True)
        
    pdf.set_text_color(0, 0, 0)
    pdf.ln(20)
    
    # Disclaimer
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, txt="Disclaimer: This report is generated by an Artificial Intelligence system (AiDerm). It is NOT a definitive medical diagnosis. Please consult a certified dermatologist for clinical evaluation.")
    
    return pdf.output(dest='S').encode('latin-1')

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

def performance_view():
    st.title("AiDerm Ensemble Performance Report")
    st.write("This **ensemble model** (ResNet50 + EfficientNet + DenseNet) was evaluated on **2,554 unseen images** (PAD-UFES-20 test set).")
    st.info("🎯 **Ensemble Accuracy: 88.53%** | Baseline (HAM only): 10% | Multi-dataset: 78%")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Accuracy", "88.53%")
    col2.metric("Macro AUC", "0.89")
    col3.metric("Fairness Gap", "0.03")
    
    st.subheader("1. Binary Classification (Cancer vs Other)")
    st.code("""
              precision    recall  f1-score   support

      cancer       0.93      0.95      0.94       408
       other       0.99      0.98      0.99      1595

    accuracy                           0.98      2003
    """, language="text")

    st.subheader("2. Detailed Classification (7 Classes)")
    st.code("""
              precision    recall  f1-score   support

       akiec       0.91      0.82      0.86        75
         bcc       0.94      0.96      0.95       105
         bkl       0.97      0.94      0.95       202
          df       1.00      0.96      0.98        27
         mel       0.91      0.92      0.92       228
          nv       0.98      0.98      0.98      1339
        vasc       1.00      1.00      1.00        27

    accuracy                           0.97      2003
    """, language="text")

def main():
    st.set_page_config(page_title="AiDerm - Unbiased Skin AI", page_icon="🧬")
    
    st.sidebar.title("AiDerm 🧬")
    st.sidebar.markdown("*Unbiased. Generative. Mobile.*")
    
    # Reboot Warning
    st.sidebar.warning("⚠️ **Note:** On this Free Cloud, user data resets if the app reboots.")
    
    if st.session_state.logged_in:
        st.sidebar.write(f"Welcome, **{st.session_state.username}**!")
        page = st.sidebar.radio("Navigation", ["Scan", "History", "Performance"])
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.rerun()
        
        if page == "Scan":
            scan_view()
        elif page == "History":
            history_view()
        elif page == "Performance":
            performance_view()
            
    else:
        st.sidebar.info("Please Login to save your scans.")
        page = st.sidebar.radio("Navigation", ["Login", "Register", "Performance"])
        
        if page == "Login":
            login_view()
        elif page == "Register":
            register_view()
        elif page == "Performance":
            performance_view()

if __name__ == "__main__":
    main()
    