import streamlit as st
from PIL import Image


def upload_image():

    st.subheader("📤 Upload Chest X-ray Image")

    uploaded_file = st.file_uploader(
        "Choose a Chest X-ray Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        return None, None, None

    image = Image.open(uploaded_file)

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### 🖼 Uploaded Image")
        st.image(
            image,
            caption="Chest X-ray",
            width="stretch"
        )

    return uploaded_file, image, right