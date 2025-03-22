import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime
import tensorflow as tf

st.set_page_config(
    page_title="Stroke Segmentation App",
    page_icon="🧠",
    layout="wide"
)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        curr_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_channels, feature))
            curr_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)

        return self.final_conv(x)

def preprocess_image(image_data, target_size=(256, 256)):
    try:
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)
    except Exception as e:
        st.error(f"Error opening image: {e}")
        return None, None
    
    original_image = image.copy()
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] > 3:
        image = image[:, :, :3]
    
    image_resized = cv2.resize(image, target_size)
    
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_tensor = torch.tensor(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return original_image, image_tensor

def preprocess_image_tf(image_data, target_size=(256, 256)):
    try:
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)
    except Exception as e:
        st.error(f"Error opening image: {e}")
        return None
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] > 3:
        image = image[:, :, :3]
    
    image_resized = cv2.resize(image, target_size)
    
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    image_tensor = np.expand_dims(image_normalized, axis=0)
    
    return image_tensor

def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        prediction = model(image_tensor)
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > 0.5).float()
    return prediction

def predict_stroke(model, image_tensor):
    prediction = model.predict(image_tensor)
    return prediction

@st.cache_resource
def load_segmentation_model(model_file):
    model = UNet(in_channels=3, out_channels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_bytes = io.BytesIO(model_file.getvalue())
    model.load_state_dict(torch.load(model_bytes, map_location=device))
    model.to(device)
    return model, device

@st.cache_resource
def load_prediction_model(model_file):
    model_bytes = io.BytesIO(model_file.getvalue())
    
    model = tf.keras.models.load_model(model_bytes)
    return model

def overlay_mask(original_image, mask, alpha=0.5):
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
    if original_image.max() > 1.0:
        original_image = original_image.astype(np.float32) / 255.0
    
    overlay_img = original_image.copy()
    
    mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
    
    colored_mask = np.zeros_like(overlay_img)
    if colored_mask.dtype != np.float32:
        colored_mask = colored_mask.astype(np.float32)
    
    colored_mask[mask_resized > 0] = [1.0, 0, 0]
    
    result = cv2.addWeighted(overlay_img, 1, colored_mask, alpha, 0)
    
    result = np.clip(result, 0, 1)
    
    return (result * 255).astype(np.uint8)

def prepare_for_display(image):
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    return image

st.title("🧠 Stroke Segmentation")
st.markdown("""
This application helps segment strokes in brain scans. Upload images and perform detailed segmentation analysis.
""")

st.sidebar.header("Model Setup")

segmentation_model = st.sidebar.file_uploader("Upload Segmentation Model (.pth file)", type=["pth"])
prediction_model = st.sidebar.file_uploader("Upload Stroke Prediction Model (.h5 file)", type=["h5"])

segmentation_model_loaded = False
prediction_model_loaded = False

if segmentation_model is not None:
    try:
        seg_model, seg_device = load_segmentation_model(segmentation_model)
        st.sidebar.success("Segmentation model loaded successfully!")
        segmentation_model_loaded = True
    except Exception as e:
        st.sidebar.error(f"Failed to load segmentation model: {e}")
        segmentation_model_loaded = False

if prediction_model is not None:
    try:
        pred_model = load_prediction_model(prediction_model)
        st.sidebar.success("Stroke prediction model loaded successfully!")
        prediction_model_loaded = True
    except Exception as e:
        st.sidebar.error(f"Failed to load prediction model: {e}")
        prediction_model_loaded = False

st.header("Patient Information")
col1, col2 = st.columns(2)

with col1:
    patient_name = st.text_input("Patient Name")
    patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=45)

with col2:
    patient_sex = st.selectbox("Patient Sex", ["Male", "Female", "Other"])
    patient_id = st.text_input("Patient ID")

st.header("Upload Brain Scans")
uploaded_files = st.file_uploader("Upload brain scan images", type=["jpg", "jpeg", "png", "tif", "tiff"], accept_multiple_files=True)

if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []

if uploaded_files:
    st.write("Processing uploaded images...")
    
    if len(st.session_state.processed_images) != len(uploaded_files):
        st.session_state.processed_images = []
        
    with st.spinner("Processing images..."):
        for file in uploaded_files:
            if any(img['filename'] == file.name for img in st.session_state.processed_images):
                continue
                
            file_bytes = file.getvalue()
            original_image, image_tensor = preprocess_image(file_bytes)
            
            if original_image is not None and image_tensor is not None:
                st.session_state.processed_images.append({
                    'filename': file.name,
                    'original_image': original_image,
                    'image_tensor': image_tensor,
                    'file_bytes': file_bytes
                })
    
    st.header("Image Analysis Results")
    
    st.subheader("Scan Images")
    
    cols = st.columns(4)
    selected_images = []
    
    for idx, img_data in enumerate(st.session_state.processed_images):
        original = prepare_for_display(img_data['original_image'])
        
        col_idx = idx % 4
        with cols[col_idx]:
            st.image(original, caption=f"{img_data['filename']}", width=150)
            if st.checkbox(f"Select #{idx}", key=f"select_{idx}"):
                selected_images.append(idx)
    
    if selected_images:
        st.header("Analysis of Selected Images")
        
        for img_idx in selected_images:
            img_data = st.session_state.processed_images[img_idx]
            st.subheader(f"Analysis for {img_data['filename']}")
            
            tab1, tab2 = st.tabs(["Segmentation Analysis", "Stroke Prediction"])
            
            with tab1:
                if segmentation_model_loaded:
                    prediction = predict(seg_model, img_data['image_tensor'], seg_device)
                    
                    mask = prediction[0, 0].cpu().numpy()
                    
                    display_original = prepare_for_display(img_data['original_image'])
                    
                    overlay = overlay_mask(img_data['original_image'], mask)
                    
                    mask_display = (mask * 255).astype(np.uint8)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Original Scan**")
                        st.image(display_original, use_column_width=True)
                    
                    with col2:
                        st.markdown("**Segmentation Mask**")
                        st.image(mask_display, caption="White indicates stroke regions", use_column_width=True)
                    
                    with col3:
                        st.markdown("**Overlay Result**")
                        st.image(overlay, caption="Stroke highlighted in red", use_column_width=True)
                    
                    stroke_percentage = (mask.sum() / mask.size) * 100
                    st.metric("Affected Area", f"{stroke_percentage:.2f}%")
                    
                    st.markdown("### Download Options")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.download_button(
                            label="Download Original",
                            data=cv2.imencode('.png', cv2.cvtColor(display_original, cv2.COLOR_RGB2BGR))[1].tobytes(),
                            file_name=f"{img_data['filename'].split('.')[0]}_original.png",
                            mime="image/png"
                        )
                    
                    with col2:
                        st.download_button(
                            label="Download Mask",
                            data=cv2.imencode('.png', mask_display)[1].tobytes(),
                            file_name=f"{img_data['filename'].split('.')[0]}_mask.png",
                            mime="image/png"
                        )
                    
                    with col3:
                        st.download_button(
                            label="Download Overlay",
                            data=cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))[1].tobytes(),
                            file_name=f"{img_data['filename'].split('.')[0]}_overlay.png",
                            mime="image/png"
                        )
                else:
                    st.warning("Please upload a segmentation model to perform segmentation analysis.")
            
            with tab2:
                if prediction_model_loaded:
                    tf_tensor = preprocess_image_tf(img_data['file_bytes'])
                    
                    if tf_tensor is not None:
                        prediction = predict_stroke(pred_model, tf_tensor)
                        
                        display_original = prepare_for_display(img_data['original_image'])
                        st.image(display_original, caption="Original Image", width=300)
                        
                        if len(prediction.shape) > 2:
                            pred_display = prediction[0]
                            if len(pred_display.shape) > 2:
                                if pred_display.shape[-1] == 1:
                                    pred_display = pred_display[:, :, 0]
                                else:
                                    pred_display = np.mean(pred_display, axis=-1)
                                    
                            pred_display = (pred_display - pred_display.min()) / (pred_display.max() - pred_display.min() + 1e-8)
                            pred_display = (pred_display * 255).astype(np.uint8)
                            
                            pred_display = cv2.resize(pred_display, (display_original.shape[1], display_original.shape[0]))
                            
                            st.image(pred_display, caption="Prediction Heatmap", width=300)
                            
                            heatmap = cv2.applyColorMap(pred_display, cv2.COLORMAP_JET)
                            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                            overlay_heat = cv2.addWeighted(display_original, 0.7, heatmap, 0.3, 0)
                            
                            st.image(overlay_heat, caption="Prediction Overlay", width=300)
                            
                        else:
                            pred_value = float(prediction[0][0])
                            st.metric("Stroke Probability", f"{pred_value:.2%}")
                            
                            progress_color = "red" if pred_value > 0.5 else "green"
                            st.progress(pred_value)
                            
                            if pred_value > 0.7:
                                st.error("High risk of stroke detected")
                            elif pred_value > 0.3:
                                st.warning("Moderate risk of stroke detected")
                            else:
                                st.success("Low risk of stroke detected")
                            
                            st.write("### Interpretation")
                            if pred_value > 0.5:
                                st.write("This scan shows features consistent with stroke pathology.")
                            else:
                                st.write("This scan does not show strong features of stroke pathology.")
                else:
                    st.warning("Please upload a stroke prediction model to perform prediction analysis.")
    
    else:
        st.info("Please select images to analyze.")
    
else:
    st.info("Please upload brain scan images and ensure at least one model is loaded.")

with st.expander("About the System"):
    st.write("""
    This application provides a workflow for stroke analysis:
    
    **Workflow:**
    1. Upload multiple brain scan images
    2. Select specific images for detailed analysis
    3. View and download segmentation results
    4. Check stroke prediction for selected images
    
    **Models Required:**
    - Segmentation Model (.pth format): Provides detailed pixel-level segmentation of stroke areas
    - Stroke Prediction Model (.h5 format): Provides stroke likelihood prediction
    """)

with st.expander("Troubleshooting"):
    st.write("""
    **Common Issues:**
    
    1. **Model loading error:** 
       - For segmentation: Make sure your .pth file is a valid PyTorch U-Net model trained for stroke segmentation
       - For prediction: Make sure your .h5 file is a valid TensorFlow/Keras model trained for stroke prediction
    
    2. **Image loading error:** The app supports common image formats including JPEG, PNG, and TIFF.
    
    3. **Performance issues:** Processing multiple large images may require more time.
    
    4. **Memory issues:** If encountering memory errors, try processing fewer images at once.
    
    If you continue to experience problems, try using standard image formats without any special encoding.
    """)

st.markdown("---")
st.markdown("Stroke Segmentation App | Created with Streamlit")
