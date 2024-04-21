import streamlit as st


def display(image, size):
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(image, width=size)
