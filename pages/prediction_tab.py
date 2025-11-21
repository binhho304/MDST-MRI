"""Code for the predcition tab for our streamlit UI should go here"""
import streamlit as st
import model_utils
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import gradcam


def show_prediction_tab():
    """Define the Streamlit Prediction Tab UI"""
    st.title("Prediction Tab")
    st.subheader("Upload your MRI scan for prediction")
    
    uploaded_file = st.file_uploader("Choose a file (jpg)", type=["jpg"])
    
    # Model options
    model_options = ["single-model", "multi-model"]
    model_variants = ["Resnet50", "VGG16"]
    multi_models = ["Ensemble"]
    
    model_selected = st.radio("Select Model Option: ", model_options)
    
    if model_selected == "single-model":
        selected_model = st.radio("Select which model you want to use: ", model_variants)
    else:
        selected_model = st.radio("Select which model you want to use (only Ensemble available): ", multi_models)
    
    st.button("Get Prediction", on_click=fetch_prediction, args=(uploaded_file, model_selected, selected_model))


def fetch_prediction(uploaded_file, model_option, selected_model):
    if uploaded_file is None:
        st.write("Please upload an MRI scan to get a prediction.")
        return

    # Load and transform image
    pil_img = Image.open(uploaded_file).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(pil_img).unsqueeze(0)
    
    st.image(pil_img, caption='Uploaded MRI Scan.', use_container_width=True)
    st.write("")

    final_output = None
    superimposed_img = None
    predicted_class = None
    
    with st.spinner(text="Predicting in progress...", show_time=True):
        if model_option == "single-model":
            if selected_model == "Resnet50":
                final_output = model_utils.generate_softmax_outputs(model_utils.RESNET50_MODEL, tensor)
                superimposed_img = gradcam.generate_gradcam(model_utils.RESNET50_MODEL, pil_img, tensor)
            elif selected_model == "VGG16":
                final_output = model_utils.generate_softmax_outputs(model_utils.VGG16_MODEL, tensor)
                superimposed_img = gradcam.generate_gradcam(model_utils.VGG16_MODEL, pil_img, tensor)
            
            if final_output is not None:
                confidence = final_output.max().item() * 100
                predicted_class = model_utils.CLASS_NAMES[final_output.argmax()]
                st.success(f"Prediction: {predicted_class} with {confidence:.2f}% confidence.")
                st.image(superimposed_img, caption='GradCAM output.', use_container_width=True)
            else:
                st.error("Prediction failed. Please try again.")
        
        else:  # Ensemble
            predicted_class, _ = model_utils.weighted_average_ensemble(tensor)
            st.success(f"Prediction: {predicted_class}")


# Run the tab
show_prediction_tab()