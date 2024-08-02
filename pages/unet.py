import streamlit as st
import torch
from torchvision import transforms as T
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO


st.title('Unet forest segmentation')
st.header('Example of prediction')
st.image('images/example_unet.png')

def load_image_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image
    else:
        st.error("Failed to load image from URL")
        return None

device='cpu'

model = smp.Unet(encoder_name="mobilenet_v2", encoder_weights="imagenet", classes=1, activation="sigmoid", encoder_depth=5, decoder_channels=[512, 256, 64, 32, 16]).to(device)
model.load_state_dict(torch.load('models/best_weights_unet_second.pt', map_location=torch.device('cpu')))

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


st.header('Load your own image or give url')
uploaded_file = st.file_uploader("Upload image file", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
image_url = st.text_input("Enter image URL")
if uploaded_file is not None:
    for file in uploaded_file:    
        image = Image.open(file)
        col1, col2 = st.columns(2)
        col1.image(image, caption='Uploaded image file',width=315)
        transformed_img  = transform(image)
        model.eval()
        with torch.no_grad():
            pred = (model(transformed_img.unsqueeze(0).to(device))).float().cpu()
            pred = pred.squeeze().numpy()
        col2.image(pred, width=315, caption='Predicted mask')

if image_url:
        image = load_image_from_url(image_url)
        if image:
             col3, col4 = st.columns(2)
             col3.image(image, caption='Uploaded image url',width=315)
             transformed_img  = transform(image)
             model.eval()
             with torch.no_grad():
                pred = (model(transformed_img.unsqueeze(0).to(device))).float().cpu()
                pred = pred.squeeze().numpy()
             col4.image(pred, width=315, caption='Predicted mask')
             
