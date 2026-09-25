import os
import streamlit as st

from components.sidebar import show_sidebar
from components.patient_form import patient_information
from components.upload import upload_image
from components.prediction import prediction_section
from components.history import show_history
from components.brain_prediction import brain_prediction_section
from components.skin_prediction import skin_prediction_section


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="DiagNova",
    page_icon="🩺",
    layout="wide"
)


# --------------------------------------------------
# Custom Styling
# --------------------------------------------------

st.markdown(
    """
<style>

/* Main page */
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
}

/* Reduce default vertical spacing */
div[data-testid="stVerticalBlock"] {
    gap: 0.45rem;
}

/* Reduce spacing around headings */
h1, h2, h3 {
    margin-top: 0.25rem !important;
    margin-bottom: 0.35rem !important;
}

/* Reduce spacing around horizontal lines */
hr {
    margin-top: 0.35rem !important;
    margin-bottom: 0.45rem !important;
}

/* Keep the main header compact */
.diag-header h3 {
    margin-top: 0 !important;
    margin-bottom: 0.15rem !important;
}

.diag-header p {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

</style>
""",
    unsafe_allow_html=True
)


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

selected_module = show_sidebar()


# --------------------------------------------------
# Logo / Header
# --------------------------------------------------

logo_path = os.path.join(
    os.path.dirname(__file__),
    "assets",
    "logo.png"
)

col1, col2 = st.columns(
    [1, 5],
    vertical_alignment="center"
)

with col1:

    st.image(
        logo_path,
        width=125
    )

with col2:

    st.markdown(
        '<div class="diag-header">'
        '<h3 style="color:#3B82F6;">'
        'AI-Powered Medical Image Analysis Platform'
        '</h3>'
        '<p style="color:#888888;">'
        'Medical Image Intelligence • '
        'Deep Learning • '
        'Computer Vision'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )


st.divider()


# ==================================================
# PNEUMONIA DETECTION
# ==================================================

if selected_module == "🫁 Pneumonia Detection":

    st.subheader(
        "🫁 Pneumonia Detection"
    )

    patient_name, patient_age, patient_gender = (
        patient_information()
    )

    uploaded_file, image, right = upload_image()

    if uploaded_file is not None:

        with right:

            prediction_section(
                uploaded_file,
                patient_name,
                patient_age,
                patient_gender
            )


# ==================================================
# BRAIN TUMOR DETECTION
# ==================================================

elif selected_module == "🧠 Brain Tumor Detection":

    st.subheader(
        "🧠 Brain Tumor Detection"
    )

    patient_name, patient_age, patient_gender = (
        patient_information()
    )

    uploaded_file = st.file_uploader(
        "Upload Brain MRI Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="brain_mri_upload"
    )

    if uploaded_file is not None:

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### 🖼 Brain MRI"
            )

            st.image(
                uploaded_file,
                caption="Uploaded Brain MRI",
                width="stretch"
            )

        with col2:

            brain_prediction_section(
                uploaded_file,
                patient_name,
                patient_age,
                patient_gender
            )


# ==================================================
# SKIN DISEASE DETECTION
# ==================================================

elif selected_module == "🧴 Skin Disease Detection":

    st.subheader(
        "🧴 Skin Disease Detection"
    )

    patient_name, patient_age, patient_gender = (
        patient_information()
    )

    uploaded_file = st.file_uploader(
        "Upload Skin Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="skin_image_upload"
    )

    if uploaded_file is not None:

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### 🖼 Skin Image"
            )

            st.image(
                uploaded_file,
                caption="Uploaded Skin Image",
                width="stretch"
            )

        with col2:

            skin_prediction_section(
                uploaded_file,
                patient_name,
                patient_age,
                patient_gender
            )


# ==================================================
# UNKNOWN MODULE
# ==================================================

else:

    st.info(
        "Please select a medical image analysis "
        "module from the sidebar."
    )


# ==================================================
# Medical History
# ==================================================

show_history()


# ==================================================
# Project Highlights
# ==================================================

st.subheader(
    "🚀 Project Highlights"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.info(
        """
        ### 🧠 AI Models

        • Custom CNN

        • Pneumonia Detection

        • Brain Tumor Classification

        • Skin Disease Classification
        """
    )


with col2:

    st.info(
        """
        ### 💻 Technologies

        • Python

        • PyTorch

        • Streamlit

        • OpenCV

        • Computer Vision
        """
    )


with col3:

    st.info(
        """
        ### ⭐ Features

        • Real-time Prediction

        • Confidence Score

        • Medical History

        • PDF Medical Reports
        """
    )


# ==================================================
# About DiagNova
# ==================================================

st.divider()

st.subheader(
    "ℹ️ About DiagNova"
)

st.write(
    """
    DiagNova is an AI-powered medical image analysis
    platform designed to assist in the analysis of
    medical images using Deep Learning and Computer Vision.

    The platform currently supports three AI modules:

    • Pneumonia Detection from Chest X-ray images

    • Brain Tumor Classification from Brain MRI images

    • Skin Disease Classification from skin images

    The project demonstrates the complete AI workflow
    including image preprocessing, CNN model training,
    evaluation, real-time prediction, confidence scoring,
    medical history tracking, and PDF report generation
    through Streamlit.

    Its modular architecture allows future expansion into
    additional medical imaging applications.
    """
)


# ==================================================
# Disclaimer
# ==================================================

st.divider()

st.warning(
    """
    **Disclaimer**

    DiagNova is developed for educational and research
    purposes only.

    The predictions generated by this application are
    AI-assisted results and should not be considered a
    substitute for professional medical diagnosis.

    Always consult a qualified healthcare professional
    before making clinical decisions.
    """
)


# ==================================================
# Footer
# ==================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#888888;">
        © 2026 Pateel Meghana • DiagNova
    </div>
    """,
    unsafe_allow_html=True
)