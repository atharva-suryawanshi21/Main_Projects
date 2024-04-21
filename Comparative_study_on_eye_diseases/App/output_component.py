import streamlit as st
from image_processing import get_image
from accuracies import get_predictions
from io import BytesIO
from PIL import Image
import numpy as np
from display_image import display
import cv2


def heading():
    # For titles
    heading = st.container()
    col1, col2, col3, col4 = heading.columns(4)
    col1.markdown(
        "<p style='text-align: center;font-size: 40px;'>Gray Scale</p>", unsafe_allow_html=True)
    col2.markdown(
        "<p style='text-align: center;font-size: 40px;'>Lab + HSV</p>", unsafe_allow_html=True)
    col3.markdown(
        "<p style='text-align: center;font-size: 40px;'>Lab + CLAHECC</p>", unsafe_allow_html=True)
    col4.markdown(
        "<p style='text-align: center;font-size: 40px;'>HSV + Gabor</p>", unsafe_allow_html=True)


def column(col, img, num):
    with col:
        image = get_image(img, num)
        display(image, 200)
        # st.write(image1.shape)
        pred_cnn, pred_svm, pred_knn, pred_adaboost, pred_xgboost = get_predictions(
            image, num)

        st.markdown(
            "<p style='text-align: center;font-size: 35px;'>Predictions</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: center;font-size: 20px;'>SVM :- {pred_cnn}</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: center;font-size: 20px;'>CNN :- {pred_svm}</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: center;font-size: 20px;'>KNN :- {pred_knn}</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: center;font-size: 20px;'>Adaboost :- {pred_adaboost}</p>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align: center;font-size: 20px;'>Xgboost :- {pred_xgboost}</p>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            "<p style='text-align: center;font-size: 40px;'>Preferred Model</p>", unsafe_allow_html=True)
        if num == 4:
            st.markdown(
                f"<p style='text-align: center;font-size: 25px;'>Convolutional Neual Network</p>", unsafe_allow_html=True)
            st.markdown(
                f"<p style='text-align: center;font-size: 25px;'>{pred_cnn}</p>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<p style='text-align: center;font-size: 25px;'>Support Vector Machines</p>", unsafe_allow_html=True)
            st.markdown(
                f"<p style='text-align: center;font-size: 25px;'>{pred_svm}</p>", unsafe_allow_html=True)


def output_container(uploaded_file):
    if uploaded_file is None:
        st.error("Please upload an image.")
        return None
    heading()

    main_container = st.container()
    col1, col2, col3, col4 = main_container.columns(4)

    file_contents = uploaded_file.getvalue()
    image = np.array(Image.open(BytesIO(file_contents)))

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3:
        rgb_image = image[:, :, : 3]
        image = rgb_image.astype(np.uint8)

    # for Images
    column(col1, image, 1)
    column(col2, image, 2)
    column(col3, image, 3)
    column(col4, image, 4)
