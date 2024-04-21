import streamlit as st


def intro_container():

    col1, col2 = st.columns([1, 3])

    # Add an image to the first column
    with col1:
        st.image("images.jpeg", width=250)
    with col2:
        st.markdown(
            "<p style='text-align: center;font-size: 70px;'>Comparative prediction of Eye Disease</p>", unsafe_allow_html=True)
    st.markdown("---")

    heading = st.container()
    col1, col2, col3 = heading.columns(3)
    col1.header("Eye Disease Types")
    col2.header("Machine Learning Models")
    col3.header("Image Enchancement")

    intro = st.container()
    col1, col2, col3 = intro.columns(3)
    with col1:
        # st.write("We classify eye images into following 4 classes:")
        st.write("1. CNV")
        st.write("2. DME")
        st.write("3. Drusen")
        st.write("4. Healthy")

    with col2:

        # st.write("We classify eye images by following 5 models:")
        st.write("1. Convolution Neural Network")
        st.write("2. Support Vector Machine")
        st.write("3. K Nearest Neighbours ")
        st.write("4. Adaboost")
        st.write("5. XGBoost")

    with col3:

        # st.write("We classify eye images into following 4 classes:")
        st.write("1. Gray Scale")
        st.write("2. Lab + HSV ")
        st.write("3. Lab + CLAHECC")
        st.write("4. HSV + Gabor")
