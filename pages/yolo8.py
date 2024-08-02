import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO

device = 'cpu'
st.title('Yolov8-turbine|cable tower-segmentation')

model = YOLO("models/best_yolo8_weights.pt")
model.to(device)
image = Image.open('images/example_image.png')
res = model.predict(image)
res_plotted = res[0].plot()
st.header('Example of prediction')
st.image(res_plotted)


def load_image_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image
    else:
        st.error("Failed to load image from URL")
        return None
    

st.header('Load your own image or give url')
uploaded_file = st.file_uploader("Upload image file", type=["jpg", "jpeg", "png"])
image_url = st.text_input("Enter image URL")



if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded image file')
        confidence_file = st.slider('Choose confidence', 0, 100)
        res_file = model.predict(image, conf=confidence_file/100)
        res_plotted_file = res_file[0].plot()
            
        st.image(res_plotted_file, caption='Predicted objects')

if image_url:
        image = load_image_from_url(image_url)
        if image:
             st.image(image, caption='Uploaded image url')
             confidence_url = st.slider('Choose your confidence', 0, 100)
             res_url = model.predict(image, conf=confidence_url/100)
             res_plotted_url = res_url[0].plot()
            
             st.image(res_plotted_url, caption='Predicted objects')
