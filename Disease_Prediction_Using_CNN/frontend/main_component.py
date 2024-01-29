import streamlit as st

disease_class = {
    "early_blight": "Early Blight",
    "late_blight": "Late Blight",
    "healthy": "Healthy Leaf"
}


def info_class(disease):
    if disease == "early_blight":
        st.write("Early blight, caused by the fungus Alternaria solani, is a common fungal disease affecting potato plants. It typically appears as dark brown to black lesions with concentric rings on the lower leaves of the plant. These lesions can expand and merge, eventually causing the leaves to wither and die. Early blight is favored by warm and humid conditions and can significantly reduce yield if left untreated. Control measures include crop rotation, fungicide application, and removal of infected plant debris to prevent the spread of the disease.")
    elif disease == "late_blight":
        st.write("Late blight, caused by the oomycete pathogen Phytophthora infestans, is one of the most devastating diseases affecting potato plants worldwide. It is characterized by rapidly spreading lesions that appear as dark, water-soaked patches on the leaves, stems, and tubers of the plant. Under favorable conditions of high humidity and cool temperatures, late blight can spread rapidly, leading to complete crop loss within a short period. Historically, late blight was responsible for the Irish potato famine in the 19th century. Control measures include the use of resistant potato varieties, fungicide application, and cultural practices such as crop rotation and proper plant spacing.")
    else:
        st.write("Normal or healthy potato plants exhibit vigorous growth with green, lush foliage. The leaves are typically free from any visible signs of disease, such as lesions, spots, or discoloration. Healthy potato plants produce abundant foliage and develop tubers of uniform size and shape. They thrive in well-drained soil with adequate moisture and sunlight. Proper cultural practices, including regular irrigation, fertilization, and weed control, help maintain the health and vigor of potato plants. Additionally, timely harvesting and storage of potatoes contribute to preserving their quality and preventing post-harvest diseases. Regular monitoring and early detection of any signs of disease are essential for maintaining the health and productivity of potato crops.")


def info_col2(col2, disease, confidence):
    col2.header(disease_class[disease])
    col2.write(f"Confidence level : {confidence}")


def create_container(image, disease, confidence):
    image_container = st.container()
    col1, col2 = image_container.columns(2)
    col1.image(image)
    info_col2(col2, disease, confidence)

    expander2 = st.expander(disease_class[disease])
    with expander2:
        info_class(disease)
