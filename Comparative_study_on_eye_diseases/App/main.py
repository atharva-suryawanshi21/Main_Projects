import streamlit as st
from intro_components import intro_container
from output_component import output_container

st.set_page_config(layout="wide")


intro_container()
st.markdown("---")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])


st.markdown("---")


output_container(uploaded_file)
