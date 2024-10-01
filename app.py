import streamlit as st
import requests
from PIL import Image

st.title("Brain MRI Metastasis Segmentation")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    if st.button("Segment Metastasis"):
        # Sending the image to FastAPI for inference
        files = {'file': uploaded_file.getvalue()}
        response = requests.post("http://localhost:8000/predict/", files=files)
        result = response.json()
        st.write(result)
