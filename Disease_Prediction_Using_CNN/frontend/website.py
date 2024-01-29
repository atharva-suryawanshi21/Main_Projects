import streamlit as st
import requests
from PIL import Image
import main_component

st.title("Potato Disease Prediction")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])


def predict_class(uploaded_file):
    url = "http://127.0.0.1:8000/predict"
    files = {'file': uploaded_file}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        return response.json()
    else:
        return {'error': 'Prediction failed'}


if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        st.write("File sucessfully uploaded !")

        prediction_result = predict_class(uploaded_file)
        disease = prediction_result['class']
        confidence = prediction_result['confidence']

        image = Image.open(uploaded_file)
        st.header('Prediction')
        main_component.create_container(image, disease, confidence)
